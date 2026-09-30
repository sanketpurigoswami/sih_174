"""
Live action recognition using:
    Camera -> YOLO object detection (best.pt)
           -> ResNet18 CNN frame features
           -> 2-layer BiLSTM
           -> action prediction (best_har_model.pt)

IMPORTANT:
The provided train_har.py trains CNN+BiLSTM on the ORIGINAL VIDEO FRAMES,
not on YOLO feature vectors/crops. Therefore this script keeps the CNN input
identical to training (224x224 camera frames) and runs YOLO in parallel for
object awareness/visualization. Feeding YOLO crops/features directly into the
existing checkpoint would be a training/inference mismatch.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

YOLO_MODEL_PATH = Path(r"F:\sih\runs\detect\train\weights\best.pt")
HAR_MODEL_PATH = Path(r"F:\sih\har_model\best_har_model.pt")
STATE_LOG_PATH = Path(r"F:\sih\object_states.txt")

NUM_FRAMES = 48
IMAGE_SIZE = 224

# Predict after this many frames are available.
# 1 = every frame; higher = less compute.
PREDICT_EVERY_N_FRAMES = 1

# A small prediction smoothing window avoids one-frame flicker.
PREDICTION_SMOOTHING = 5

YOLO_CONFIDENCE = 0.35
YOLO_IOU = 0.50

# Camera
CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Right-side panel
PANEL_WIDTH = 360

# YOLO class ids from your trained detector.
OBJECT_CLASS_IDS = {0, 1, 2, 3}
PLACE_CLASS_IDS = {4, 5, 6, 7}
PERSON_CLASS_ID = 8
HAND_CLASS_ID = 9

# ============================================================
# DEVICE
# ============================================================


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple Metal (MPS).")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using NVIDIA CUDA.")
        return torch.device("cuda")
    print("Using CPU.")
    return torch.device("cpu")


# ============================================================
# CNN + BiLSTM: EXACT ARCHITECTURE FROM train_har.py
# ============================================================

class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        # Exactly as in training: CNN was frozen and used only as a feature
        # extractor. The checkpoint is therefore expected to contain the CNN
        # weights too.
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape

        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        # Match train_har.py exactly: CNN runs without gradients.
        with torch.no_grad():
            x = self.cnn(x)

        x = x.flatten(start_dim=1)
        x = x.reshape(batch_size, time_steps, -1)

        lstm_output, _ = self.lstm(x)
        features = lstm_output[:, -1, :]
        logits = self.classifier(features)
        return logits


# ============================================================
# IMAGE PREPROCESSING: EXACT RESNET18 IMAGENET TRANSFORM
# ============================================================

weights = ResNet18_Weights.DEFAULT
TRANSFORM = weights.transforms()


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(
        frame_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    image = torch.from_numpy(frame_rgb.copy())
    image = image.permute(2, 0, 1).float() / 255.0
    return TRANSFORM(image)


# ============================================================
# YOLO HELPERS
# ============================================================


def get_yolo_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def center(box: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = box.astype(int)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def find_current_object(
    detections: list[dict[str, Any]],
    hand_box: np.ndarray | None,
) -> dict[str, Any] | None:
    """Choose only the object nearest to / overlapping the detected hand."""
    objects = [d for d in detections if d["class_id"] in OBJECT_CLASS_IDS]
    if not objects or hand_box is None:
        return None

    hand_cx, hand_cy = center(hand_box)
    best = None
    best_score = -1e18

    for obj in objects:
        box = obj["box"]
        obj_cx, obj_cy = center(box)

        dx = float(obj_cx - hand_cx)
        dy = float(obj_cy - hand_cy)
        distance = float((dx * dx + dy * dy) ** 0.5)

        overlap = box_iou(hand_box, box)
        # Strongly prefer overlap, then proximity, then confidence.
        score = overlap * 1000.0 - distance + obj["confidence"] * 100.0

        if score > best_score:
            best_score = score
            best = obj

    return best


# ============================================================
# UI HELPERS
# ============================================================


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    # White text with black background for readability.
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = org
    cv2.rectangle(
        image,
        (x - 5, y - th - baseline - 5),
        (x + tw + 5, y + 5),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    frame: np.ndarray,
    class_names: list[str],
    current_action: str,
    current_object_name: str,
    action_confidence: float,
    sequence_ready: bool,
    yolo_object_count: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    panel = np.full((h, PANEL_WIDTH, 3), 245, dtype=np.uint8)

    def ptext(text: str, y: int, scale: float = 0.62, bold: bool = False):
        thickness = 2 if bold else 1
        cv2.putText(
            panel,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    ptext("LIVE ACTION RECOGNITION", 42, 0.70, True)
    ptext("YOLO + CNN + BiLSTM", 72, 0.55, False)

    cv2.line(panel, (20, 92), (PANEL_WIDTH - 20, 92), (80, 80, 80), 1)

    ptext("CURRENT OBJECT", 125, 0.55, True)
    ptext(current_object_name if current_object_name else "None", 158, 0.70, True)

    ptext("CURRENT ACTION", 205, 0.55, True)
    ptext(current_action, 238, 0.75, True)
    ptext(f"Confidence: {action_confidence * 100:.1f}%", 268, 0.52)

    cv2.line(panel, (20, 292), (PANEL_WIDTH - 20, 292), (80, 80, 80), 1)

    ptext("6 ACTION CLASSES", 325, 0.55, True)
    y = 358
    for idx, name in enumerate(class_names):
        ptext(f"{idx}: {name}", y, 0.53)
        y += 28

    ptext("SYSTEM", y + 10, 0.55, True)
    ptext(
        "Sequence: READY" if sequence_ready else "Sequence: WAITING",
        y + 42,
        0.52,
    )
    ptext(f"YOLO objects: {yolo_object_count}", y + 70, 0.52)
    ptext("Press Q to quit", h - 25, 0.50)

    return np.hstack([frame, panel])


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    state_log_file = open(STATE_LOG_PATH, "a", encoding="utf-8")

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH.resolve()}"
        )
    if not HAR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"HAR model not found: {HAR_MODEL_PATH.resolve()}"
        )

    device = get_device()

    print("Loading YOLO...")
    yolo = YOLO(str(YOLO_MODEL_PATH))
    yolo_names = get_yolo_names(yolo)
    print("YOLO classes:")
    for class_id, name in sorted(yolo_names.items()):
        print(f"  {class_id}: {name}")

    print("Loading HAR checkpoint...")
    checkpoint = torch.load(
        HAR_MODEL_PATH,
        map_location="cpu",
    )

    class_names = checkpoint["class_names"]
    hidden_size = int(checkpoint.get("lstm_hidden_size", 256))
    lstm_layers = int(checkpoint.get("lstm_layers", 2))
    dropout = float(checkpoint.get("dropout", 0.30))
    checkpoint_num_frames = int(checkpoint.get("num_frames", 48))
    checkpoint_image_size = int(checkpoint.get("image_size", 224))

    if checkpoint_num_frames != NUM_FRAMES:
        raise RuntimeError(
            f"Checkpoint expects {checkpoint_num_frames} frames, but script is set to {NUM_FRAMES}."
        )
    if checkpoint_image_size != IMAGE_SIZE:
        raise RuntimeError(
            f"Checkpoint expects image size {checkpoint_image_size}, but script is set to {IMAGE_SIZE}."
        )

    print("HAR classes:", class_names)
    print(f"HAR sequence length: {checkpoint_num_frames}")

    model = CNNBiLSTM(
        num_classes=len(class_names),
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    sequence = deque(maxlen=NUM_FRAMES)
    prediction_history = deque(maxlen=PREDICTION_SMOOTHING)

    frame_counter = 0
    current_action = "WAITING"
    current_confidence = 0.0
    current_object_name = "None"

    print("Camera started. Press Q to quit.")

    WINDOW_NAME = "Live YOLO + CNN + BiLSTM Action Recognition"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)


    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed.")
            break

        frame_counter += 1
        display = frame.copy()

        # ----------------------------------------------------
        # YOLO perception
        # ----------------------------------------------------
        result = yolo.predict(
            source=frame,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []
        hand_box = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs = result.boxes.conf.detach().cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confs):
                d = {
                    "box": box.astype(np.float32),
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                }
                detections.append(d)
                if cls_id == HAND_CLASS_ID and (
                    hand_box is None or conf > next(
                        (x["confidence"] for x in detections if x["class_id"] == HAND_CLASS_ID),
                        -1.0,
                    )
                ):
                    hand_box = box.astype(np.float32)

        current_obj = find_current_object(detections, hand_box)
                # ----------------------------------------------------
        # Log state of all detected objects
        # ----------------------------------------------------
        for d in detections:
            if d["class_id"] not in OBJECT_CLASS_IDS:
                continue

            object_name = yolo_names.get(
                d["class_id"],
                f"class_{d['class_id']}"
            )

            if current_obj is not None and d is current_obj:
                state = "INTERACTING"
            else:
                state = "NOT_INTERACTING"

            state_log_file.write(
                f"{object_name} : {state}\n"
            )

        state_log_file.write("\n")
        state_log_file.flush()


        if current_obj is not None:
            current_object_name = yolo_names.get(
                current_obj["class_id"],
                f"class_{current_obj['class_id']}",
            )
        else:
            current_object_name = "None"

        # ----------------------------------------------------
        # Draw YOLO boxes.
        # Current hand-interacted object gets a thicker box.
        # No target is displayed.
        # ----------------------------------------------------
        for d in detections:
            box = d["box"].astype(int)
            cls_id = d["class_id"]
            conf = d["confidence"]
            name = yolo_names.get(cls_id, f"class_{cls_id}")

            is_current = (
                current_obj is not None
                and d is current_obj
            )

            thickness = 4 if is_current else 2
            cv2.rectangle(
                display,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 0, 0),
                thickness,
            )

            label = f"CURRENT: {name} {conf:.2f}" if is_current else f"{name} {conf:.2f}"
            draw_text(display, label, (box[0], max(25, box[1] - 8)), 0.52, 2)

        # ----------------------------------------------------
        # CNN + BiLSTM sequence
        # IMPORTANT: Full frame, exactly like training.
        # ----------------------------------------------------
        try:
            frame_tensor = frame_to_tensor(frame)
            sequence.append(frame_tensor)
        except Exception as exc:
            print(f"Frame preprocessing error: {exc}")
            continue

        if (
            len(sequence) == NUM_FRAMES
            and frame_counter % PREDICT_EVERY_N_FRAMES == 0
        ):
            batch = torch.stack(list(sequence), dim=0)
            batch = batch.unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probabilities).item())
                pred_conf = float(probabilities[pred_idx].item())

            prediction_history.append(pred_idx)

            # Majority vote over recent predictions.
            counts = np.bincount(
                np.asarray(prediction_history, dtype=np.int64),
                minlength=len(class_names),
            )
            smooth_idx = int(np.argmax(counts))

            current_action = class_names[smooth_idx]
            # Show confidence of the actual predicted class.
            current_confidence = float(probabilities[smooth_idx].item())

        # ----------------------------------------------------
        # Right-side information panel
        # ----------------------------------------------------
        object_count = sum(
            1 for d in detections if d["class_id"] in OBJECT_CLASS_IDS
        )

        output = draw_panel(
            display,
            class_names,
            current_action,
            current_object_name,
            current_confidence,
            len(sequence) == NUM_FRAMES,
            object_count,
        )

        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    state_log_file.close()

    print("Camera stopped.")


if __name__ == "__main__":
    main()
