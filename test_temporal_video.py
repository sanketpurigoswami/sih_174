import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Reuse the EXACT feature extractor and 1D CNN definition used for training.
from train_temporal_1dcnn_sliding_window_per_clip_mediapipe import (
    CLASS_NAMES,
    DEFAULT_HAND_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_YOLO_MODEL,
    FEATURE_FPS,
    HAND_DETECTION_CONF,
    HAND_PRESENCE_CONF,
    HAND_TRACKING_CONF,
    InteractionFeatureExtractor,
    TemporalCNN,
    standardize,
)

DEFAULT_VIDEO = r"F:\sih\testvideo.mp4"


def load_artifacts(output_dir, yolo_model):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    scaler = np.load(output_dir / "feature_standardizer.npz")
    checkpoint = torch.load(output_dir / "temporal_1dcnn_best.pt", map_location="cpu")

    object_names = [str(yolo_model.names[i]) for i in range(len(yolo_model.names))]
    if object_names != metadata["object_class_names"]:
        raise RuntimeError(
            "YOLO class mismatch.\n"
            f"Training: {metadata['object_class_names']}\n"
            f"Current:  {object_names}"
        )

    model = TemporalCNN(int(metadata["input_features"]), len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device, metadata, scaler["mean"], scaler["std"], object_names


def draw_hands(frame, hands):
    for hand in hands:
        if not hand.present:
            continue
        pts = hand.landmarks_px[:, :2].astype(np.int32)
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
        ]
        for a, b in connections:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0,220,0), 1)
        for p in pts:
            cv2.circle(frame, tuple(p), 2, (0,255,0), -1)


def draw_objects(frame, objects, names):
    for cls_id, item in objects.items():
        if item is None:
            continue
        x1, y1, x2, y2 = map(int, item["box"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,150,0), 2)
        cv2.putText(
            frame,
            f"{names[cls_id]} {item['confidence']:.2f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            2,
        )


def draw_probabilities(frame, probabilities, action, confidence, buffer_size, window_size):
    cv2.putText(
        frame,
        f"ACTION: {action}  {confidence:.1%}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2,
    )
    cv2.putText(
        frame,
        f"BUFFER: {buffer_size}/{window_size}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255,255,255),
        1,
    )
    if probabilities is None:
        return
    y = 100
    for name, p in zip(CLASS_NAMES, probabilities):
        cv2.putText(
            frame,
            f"{name:10s} {p:6.1%}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255,255,255),
            1,
        )
        y += 24


def main():
    parser = argparse.ArgumentParser(description="Test trained SIH temporal 1D CNN on a video")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    args = parser.parse_args()

    print("Loading YOLO...")
    yolo = YOLO(args.yolo_model)
    model, device, metadata, mean, std, object_names = load_artifacts(args.output, yolo)

    print(f"Device: {device}")
    print(f"Opening: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    interval = 1.0 / float(metadata["feature_fps"])
    last_sample_time = -1e9
    window_size = int(metadata["seq_len"])
    buffer = deque(maxlen=window_size)
    ema = None
    ema_alpha = 0.25

    base_options = python.BaseOptions(model_asset_path=args.hand_model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=HAND_DETECTION_CONF,
        min_hand_presence_confidence=HAND_PRESENCE_CONF,
        min_tracking_confidence=HAND_TRACKING_CONF,
    )

    hand_detector = vision.HandLandmarker.create_from_options(options)
    extractor = InteractionFeatureExtractor(yolo, hand_detector, object_names)

    cv2.namedWindow("SIH Temporal 1D CNN - Video Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SIH Temporal 1D CNN - Video Test", 1280, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_time = frame_idx / float(fps)
            debug = None

            if video_time - last_sample_time >= interval:
                timestamp_ms = int(round(video_time * 1000.0))
                dt = 0.0 if last_sample_time < 0 else max(1e-6, video_time - last_sample_time)

                vec, hands, objects = extractor.extract(
                    frame,
                    timestamp_ms,
                    0.0,
                    dt,
                )
                buffer.append(vec)
                debug = (hands, objects)
                last_sample_time = video_time

                if len(buffer) == window_size:
                    sequence = standardize(np.stack(buffer), mean, std)
                    x = torch.from_numpy(sequence).float().unsqueeze(0).transpose(1, 2).to(device)
                    with torch.no_grad():
                        probabilities = torch.softmax(model(x), dim=1)[0].cpu().numpy()

                    ema = probabilities if ema is None else (1.0 - ema_alpha) * ema + ema_alpha * probabilities

            if debug is not None:
                draw_hands(frame, debug[0])
                draw_objects(frame, debug[1], object_names)

            if ema is None:
                action = "WARMING UP"
                confidence = 0.0
            else:
                idx = int(np.argmax(ema))
                confidence = float(ema[idx])
                action = CLASS_NAMES[idx] if confidence >= 0.45 else "UNCERTAIN"

            draw_probabilities(frame, ema, action, confidence, len(buffer), window_size)

            cv2.imshow("SIH Temporal 1D CNN - Video Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1
    finally:
        cap.release()
        hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
