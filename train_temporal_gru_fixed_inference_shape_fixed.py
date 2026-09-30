
"""
SIH 2026 - Temporal GRU for Human-Object Interaction
=====================================================

Classes:
    0 GRASP
    1 IDLE
    2 PICKUP
    3 PLACE
    4 TRANSPORT
    5 RELEASE

Training data:
    F:\sih\temporal_clips\
        0_GRASP\*.mp4
        1_IDLE\*.mp4
        2_PICKUP\*.mp4
        3_PLACE\*.mp4
        4_TRANSPORT\*.mp4
        5_RELEASE\*.mp4

IMPORTANT:
    - Clips are assumed to be clean, single-action clips.
    - Labels come ONLY from the parent folder.
    - Sliding windows are made from feature sequences.
    - No raw videos are cut or rewritten.
    - Train/validation/test split happens BEFORE sliding windows, so windows
      from the same original clip cannot leak across splits.
    - MediaPipe is created separately for EVERY clip. This prevents the
      "Input timestamp must be monotonically increasing" error between clips.
    - Short clips with fewer than WINDOW_SIZE sampled timesteps are skipped.
    - Validation is checked explicitly and per-class counts are printed.

MODEL:
    YOLO + MediaPipe -> per-timestep feature vector
                     -> 20-timestep sliding windows
                     -> GRU
                     -> 6-class action prediction

Usage:
    TRAIN:
        python train_temporal_gru_fixed_inference_shape_fixed.py --mode train --rebuild-cache

    TRAIN using an existing cache:
        python train_temporal_gru_fixed_inference_shape_fixed.py --mode train

    LIVE CAMERA:
        python train_temporal_gru_fixed_inference_shape_fixed.py --mode infer --source 1

    VIDEO:
        python train_temporal_gru_fixed_inference_shape_fixed.py --mode infer --source "F:/sih/testvideo.mp4"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = [
    "GRASP",
    "IDLE",
    "PICKUP",
    "PLACE",
    "TRANSPORT",
    "RELEASE",
]

CLASS_FOLDERS = [
    "0_GRASP",
    "1_IDLE",
    "2_PICKUP",
    "3_PLACE",
    "4_TRANSPORT",
    "5_RELEASE",
]

DEFAULT_CLIPS = r"F:\sih\temporal_clips"
DEFAULT_YOLO = r"F:\sih\runs\detect\train\weights\best.pt"
DEFAULT_HAND = r"F:\sih\models\hand_landmarker.task"
DEFAULT_OUT = r"F:\sih\temporal_gru"

WINDOW_SIZE = 20          # timesteps per training sample
WINDOW_STRIDE = 2         # sliding-window stride
FEATURE_FPS = 20.0        # feature sampling rate

YOLO_CONF = 0.25
HAND_DET_CONF = 0.50
HAND_PRESENCE_CONF = 0.50
HAND_TRACKING_CONF = 0.50

BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.25
PATIENCE = 10

SEED = 42

HAND_TIP_IDS = [4, 8, 12, 16, 20]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GEOMETRY
# ============================================================

def point_to_box_distance(
    px: float,
    py: float,
    box: Tuple[float, float, float, float],
) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(math.hypot(dx, dy))


def box_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_diagonal(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


# ============================================================
# HAND STATE
# ============================================================

class HandState:
    def __init__(
        self,
        present: float = 0.0,
        landmarks_px: Optional[np.ndarray] = None,
        wrist_norm: Tuple[float, float] = (0.0, 0.0),
        scale_px: float = 1.0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
    ):
        self.present = present
        self.landmarks_px = (
            landmarks_px
            if landmarks_px is not None
            else np.zeros((21, 3), dtype=np.float32)
        )
        self.wrist_norm = wrist_norm
        self.scale_px = scale_px
        self.bbox = bbox
        self.velocity = velocity


def hand_scale(points_xy: np.ndarray) -> float:
    return max(
        1.0,
        float(np.linalg.norm(points_xy[9] - points_xy[0]))
    )


def hand_bbox(points_xy: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(points_xy[:, 0].min()),
        float(points_xy[:, 1].min()),
        float(points_xy[:, 0].max()),
        float(points_xy[:, 1].max()),
    )


def parse_hands(
    hand_result,
    frame_w: int,
    frame_h: int,
    previous_wrists: List[Optional[Tuple[float, float]]],
    dt_seconds: float,
) -> List[HandState]:

    detected = []

    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            pts = np.array(
                [
                    [
                        p.x * frame_w,
                        p.y * frame_h,
                        p.z * frame_w,
                    ]
                    for p in hand_landmarks
                ],
                dtype=np.float32,
            )

            scale = hand_scale(pts[:, :2])

            wrist_norm = (
                float(pts[0, 0] / max(1, frame_w)),
                float(pts[0, 1] / max(1, frame_h)),
            )

            detected.append(
                (
                    pts,
                    hand_bbox(pts[:, :2]),
                    scale,
                    wrist_norm,
                )
            )

    # Deterministic left-to-right slots.
    detected.sort(key=lambda item: item[0][0, 0])

    hands: List[HandState] = []

    for slot in range(2):
        if slot >= len(detected):
            hands.append(HandState())
            continue

        pts, bbox, scale, wrist_norm = detected[slot]

        previous = previous_wrists[slot]

        if previous is None or dt_seconds <= 0.0:
            velocity = (0.0, 0.0)
        else:
            velocity = (
                (wrist_norm[0] - previous[0]) / dt_seconds,
                (wrist_norm[1] - previous[1]) / dt_seconds,
            )

        hands.append(
            HandState(
                present=1.0,
                landmarks_px=pts,
                wrist_norm=wrist_norm,
                scale_px=scale,
                bbox=bbox,
                velocity=velocity,
            )
        )

    return hands


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class FeatureExtractor:
    """
    The SAME feature definition is used for training and inference.

    Per timestep:
      HAND:
        2 hands:
          presence
          wrist x/y
          hand scale
          wrist vx/vy
          21 landmarks x/y/z relative to wrist, normalized by hand scale

      OBJECTS:
        Every YOLO class:
          presence
          center x/y
          width/height
          confidence
          minimum fingertip->box distance
          hand/object IoU
          object vx/vy
          close-contact flag
    """

    def __init__(
        self,
        yolo_model: YOLO,
        hand_model_path: str,
        object_names: List[str],
    ):
        self.yolo = yolo_model
        self.hand_model_path = hand_model_path
        self.object_names = object_names
        self.num_objects = len(object_names)

        self.hand_detector = None
        self.reset()

    def reset(self):
        self.previous_object_centers = {
            i: None for i in range(self.num_objects)
        }
        self.previous_wrists = [None, None]

    def create_hand_detector(self):
        base = python.BaseOptions(
            model_asset_path=self.hand_model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=HAND_DET_CONF,
            min_hand_presence_confidence=HAND_PRESENCE_CONF,
            min_tracking_confidence=HAND_TRACKING_CONF,
        )

        self.hand_detector = (
            vision.HandLandmarker.create_from_options(options)
        )

    def close_hand_detector(self):
        if self.hand_detector is not None:
            self.hand_detector.close()
            self.hand_detector = None

    def reset_clip(self):
        """
        Start a completely independent MediaPipe sequence for a clip.
        """
        self.close_hand_detector()
        self.create_hand_detector()
        self.reset()

    def detect_objects(self, frame, hands):
        result = self.yolo.predict(
            source=frame,
            conf=YOLO_CONF,
            verbose=False,
        )[0]

        by_class: Dict[int, List[dict]] = {
            i: [] for i in range(self.num_objects)
        }

        if result.boxes is not None:
            for b in result.boxes:
                cls_id = int(b.cls[0].item())
                if 0 <= cls_id < self.num_objects:
                    box = tuple(
                        float(v)
                        for v in b.xyxy[0].tolist()
                    )
                    by_class[cls_id].append(
                        {
                            "box": box,
                            "confidence": float(
                                b.conf[0].item()
                            ),
                        }
                    )

        fingertips = [
            (
                float(hand.landmarks_px[t, 0]),
                float(hand.landmarks_px[t, 1]),
            )
            for hand in hands
            if hand.present
            for t in HAND_TIP_IDS
        ]

        chosen = {}

        for cls_id, candidates in by_class.items():
            if not candidates:
                chosen[cls_id] = None
                continue

            if fingertips:
                candidates.sort(
                    key=lambda item: (
                        min(
                            point_to_box_distance(
                                px, py, item["box"]
                            )
                            for px, py in fingertips
                        ),
                        -item["confidence"],
                    )
                )
            else:
                candidates.sort(
                    key=lambda item: -item["confidence"]
                )

            chosen[cls_id] = candidates[0]

        return chosen

    def hand_features(
        self,
        hands: List[HandState],
        frame_w: int,
        frame_h: int,
    ) -> List[float]:

        out = []

        for hand in hands:
            out.append(float(hand.present))

            # 2 wrist coords + scale + 2 velocity + 21*3 landmarks = 68
            if not hand.present:
                out.extend([0.0] * 68)
                continue

            out.extend(
                [
                    hand.wrist_norm[0],
                    hand.wrist_norm[1],
                    hand.scale_px / max(frame_w, frame_h),
                    hand.velocity[0],
                    hand.velocity[1],
                ]
            )

            wrist = hand.landmarks_px[0]
            scale = max(1.0, hand.scale_px)

            for p in hand.landmarks_px:
                out.extend(
                    [
                        float((p[0] - wrist[0]) / scale),
                        float((p[1] - wrist[1]) / scale),
                        float(np.clip(p[2] / scale, -3.0, 3.0)),
                    ]
                )

        return out

    def object_features(
        self,
        objects,
        hands,
        frame_w: int,
        frame_h: int,
        dt_seconds: float,
    ) -> List[float]:

        out = []

        for cls_id in range(self.num_objects):
            item = objects.get(cls_id)

            if item is None:
                # presence,cx,cy,w,h,conf,dist,iou,vx,vy,close
                out.extend([0.0] * 11)
                self.previous_object_centers[cls_id] = None
                continue

            box = item["box"]
            conf = item["confidence"]

            x1, y1, x2, y2 = box
            cx, cy = box_center(box)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)

            min_distance = 1.0
            max_iou = 0.0

            for hand in hands:
                if not hand.present or hand.bbox is None:
                    continue

                distances = [
                    point_to_box_distance(
                        float(hand.landmarks_px[t, 0]),
                        float(hand.landmarks_px[t, 1]),
                        box,
                    )
                    for t in HAND_TIP_IDS
                ]

                min_distance = min(
                    min_distance,
                    min(distances) / box_diagonal(box),
                )

                max_iou = max(
                    max_iou,
                    box_iou(hand.bbox, box),
                )

            previous = self.previous_object_centers[cls_id]

            if previous is None or dt_seconds <= 0.0:
                vx, vy = 0.0, 0.0
            else:
                vx = (
                    ((cx - previous[0]) / frame_w)
                    / dt_seconds
                )
                vy = (
                    ((cy - previous[1]) / frame_h)
                    / dt_seconds
                )

            self.previous_object_centers[cls_id] = (cx, cy)

            out.extend(
                [
                    1.0,
                    cx / frame_w,
                    cy / frame_h,
                    bw / frame_w,
                    bh / frame_h,
                    conf,
                    float(np.clip(min_distance, 0.0, 5.0)),
                    max_iou,
                    float(np.clip(vx, -1.0, 1.0)),
                    float(np.clip(vy, -1.0, 1.0)),
                    1.0 if min_distance < 0.15 else 0.0,
                ]
            )

        return out

    def extract(
        self,
        frame,
        timestamp_ms: int,
        dt_seconds: float,
    ):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.hand_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hands = parse_hands(
            result,
            w,
            h,
            self.previous_wrists,
            dt_seconds,
        )

        self.previous_wrists = [
            hand.wrist_norm if hand.present else None
            for hand in hands
        ]

        objects = self.detect_objects(
            frame,
            hands,
        )

        features = (
            self.hand_features(hands, w, h)
            + self.object_features(
                objects,
                hands,
                w,
                h,
                dt_seconds,
            )
        )

        return (
            np.asarray(features, dtype=np.float32),
            hands,
            objects,
        )


# ============================================================
# VIDEO -> FEATURE SEQUENCE
# ============================================================

def extract_clip_features(
    video_path: Path,
    extractor: FeatureExtractor,
) -> np.ndarray:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    stride = max(
        1,
        int(round(fps / FEATURE_FPS)),
    )

    # New MediaPipe detector for THIS clip.
    extractor.reset_clip()

    sequence = []
    frame_idx = 0
    previous_sample_time = None

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_idx % stride == 0:
                sample_time = frame_idx / fps

                if previous_sample_time is None:
                    dt = 0.0
                else:
                    dt = max(
                        1e-6,
                        sample_time - previous_sample_time,
                    )

                vec, _, _ = extractor.extract(
                    frame,
                    int(round(sample_time * 1000.0)),
                    dt,
                )

                sequence.append(vec)
                previous_sample_time = sample_time

            frame_idx += 1

    finally:
        cap.release()
        extractor.close_hand_detector()

    if not sequence:
        raise RuntimeError(
            f"No sampled frames in {video_path}"
        )

    return np.stack(sequence).astype(np.float32)


# ============================================================
# SLIDING WINDOWS
# ============================================================

def make_windows(
    sequence: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:

    t = sequence.shape[0]

    if t < window_size:
        return (
            np.empty(
                (0, window_size, sequence.shape[1]),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
        )

    starts = list(
        range(
            0,
            t - window_size + 1,
            stride,
        )
    )

    windows = np.stack(
        [
            sequence[s:s + window_size]
            for s in starts
        ],
        axis=0,
    )

    return windows.astype(np.float32), np.asarray(
        starts,
        dtype=np.int64,
    )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_clips(root: Path):
    clips = []

    extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    for class_id, folder_name in enumerate(CLASS_FOLDERS):
        folder = root / folder_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {folder}"
            )

        for path in sorted(folder.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in extensions
            ):
                clips.append(
                    (path, class_id)
                )

    if not clips:
        raise RuntimeError(
            f"No clips found under {root}"
        )

    return clips


def split_by_clip(
    clips,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):

    rng = random.Random(seed)

    by_class = {
        i: []
        for i in range(len(CLASS_NAMES))
    }

    for item in clips:
        by_class[item[1]].append(item)

    train = []
    val = []
    test = []

    for class_id in range(len(CLASS_NAMES)):
        items = by_class[class_id]

        if len(items) < 3:
            raise ValueError(
                f"{CLASS_NAMES[class_id]} has only "
                f"{len(items)} clips. Need at least 3."
            )

        rng.shuffle(items)

        n = len(items)

        n_train = max(
            1,
            int(round(n * train_ratio)),
        )

        n_val = max(
            1,
            int(round(n * val_ratio)),
        )

        # Guarantee at least one test clip.
        n_train = min(
            n_train,
            n - 2,
        )

        n_val = min(
            n_val,
            n - n_train - 1,
        )

        train.extend(items[:n_train])

        val.extend(
            items[n_train:n_train + n_val]
        )

        test.extend(
            items[n_train + n_val:]
        )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ============================================================
# CACHE
# ============================================================

def signature(path: Path) -> str:
    st = path.stat()

    text = (
        f"{path.resolve()}|"
        f"{st.st_size}|"
        f"{st.st_mtime_ns}"
    )

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


def cache_path(
    cache_dir: Path,
    video_path: Path,
) -> Path:
    sig = signature(video_path)
    return cache_dir / f"{sig}.npy"


def build_split_features(
    items,
    split_name: str,
    extractor: FeatureExtractor,
    cache_dir: Path,
    rebuild_cache: bool,
):

    X_list = []
    y_list = []
    clip_names = []

    print(
        f"\n=== {split_name.upper()} ==="
    )

    for index, (path, label) in enumerate(
        items,
        start=1,
    ):

        print(
            f"[{index}/{len(items)}] "
            f"{CLASS_NAMES[label]:10s} "
            f"{path.name}"
        )

        cpath = cache_path(
            cache_dir,
            path,
        )

        try:
            if (
                cpath.exists()
                and not rebuild_cache
            ):
                sequence = np.load(
                    cpath
                ).astype(np.float32)
            else:
                sequence = extract_clip_features(
                    path,
                    extractor,
                )

                np.save(
                    cpath,
                    sequence,
                )

            windows, starts = make_windows(
                sequence,
                WINDOW_SIZE,
                WINDOW_STRIDE,
            )

            if len(windows) == 0:
                print(
                    f"  WARNING: only "
                    f"{len(sequence)} sampled timesteps; "
                    f"need {WINDOW_SIZE}. SKIPPED."
                )
                continue

            X_list.append(windows)

            y_list.append(
                np.full(
                    len(windows),
                    label,
                    dtype=np.int64,
                )
            )

            clip_names.extend(
                [
                    str(path)
                    for _ in range(len(windows))
                ]
            )

            print(
                f"  sampled timesteps: "
                f"{len(sequence)} | "
                f"windows: {len(windows)}"
            )

        except Exception as exc:
            print(
                f"  ERROR: {exc}"
            )

    if not X_list:
        raise RuntimeError(
            f"No usable windows in {split_name}."
        )

    X = np.concatenate(
        X_list,
        axis=0,
    ).astype(np.float32)

    y = np.concatenate(
        y_list,
        axis=0,
    ).astype(np.int64)

    print(
        f"{split_name}: "
        f"{len(X)} windows from "
        f"{len(set(clip_names))} clips"
    )

    return X, y, clip_names


# ============================================================
# NORMALIZATION
# ============================================================

def fit_standardizer(
    X_train: np.ndarray,
):

    flat = X_train.reshape(
        -1,
        X_train.shape[-1],
    )

    mean = flat.mean(
        axis=0
    ).astype(np.float32)

    std = flat.std(
        axis=0
    ).astype(np.float32)

    std[std < 1e-6] = 1.0

    return mean, std


def standardize(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
):
    return (
        (X - mean[None, None, :])
        / std[None, None, :]
    ).astype(np.float32)


# ============================================================
# TORCH DATASET
# ============================================================

class TemporalDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        training: bool,
    ):
        self.X = X
        self.y = y
        self.training = training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.from_numpy(
            self.X[index]
        ).float()

        y = torch.tensor(
            self.y[index]
        ).long()

        # Small feature noise only.
        # No time reversal because these are directional actions.
        if self.training:
            if torch.rand(()) < 0.30:
                x = (
                    x
                    + 0.005
                    * torch.randn_like(x)
                )

        # GRU expects [batch, time, features].
        return x, y


# ============================================================
# GRU MODEL
# ============================================================

class TemporalGRU(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = 6,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(
            input_features
        )

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.input_norm(x)

        output, _ = self.gru(x)

        # Last temporal position.
        final = output[:, -1, :]

        return self.head(final)


# ============================================================
# METRICS
# ============================================================

def confusion_matrix(
    y_true,
    y_pred,
    num_classes=6,
):
    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for t, p in zip(
        y_true,
        y_pred,
    ):
        cm[int(t), int(p)] += 1

    return cm


def macro_f1(cm):
    scores = []

    for i in range(
        cm.shape[0]
    ):

        tp = cm[i, i]

        fp = (
            cm[:, i].sum()
            - tp
        )

        fn = (
            cm[i, :].sum()
            - tp
        )

        precision = (
            tp / max(1, tp + fp)
        )

        recall = (
            tp / max(1, tp + fn)
        )

        if precision + recall == 0:
            score = 0.0
        else:
            score = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        scores.append(score)

    return float(
        np.mean(scores)
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    true_values = []
    predicted_values = []

    for X, y in loader:
        X = X.to(
            device,
            non_blocking=True,
        )

        logits = model(X)

        pred = logits.argmax(
            dim=1
        ).cpu().numpy()

        predicted_values.extend(
            pred.tolist()
        )

        true_values.extend(
            y.numpy().tolist()
        )

    if not true_values:
        raise RuntimeError(
            "Validation/test loader is empty."
        )

    y_true = np.asarray(
        true_values,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        predicted_values,
        dtype=np.int64,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    accuracy = float(
        (y_true == y_pred).mean()
    )

    f1 = macro_f1(cm)

    return (
        accuracy,
        f1,
        cm,
    )


def print_split_distribution(
    name,
    y,
):
    counts = np.bincount(
        y,
        minlength=len(CLASS_NAMES),
    )

    print(
        f"\n{name} windows:"
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name:10s}: "
            f"{counts[i]}"
        )


def print_confusion_matrix(cm):
    print(
        "\nConfusion matrix "
        "(rows=true, cols=predicted):"
    )

    print(
        "            "
        + " ".join(
            f"{name[:8]:>9s}"
            for name in CLASS_NAMES
        )
    )

    for i, row in enumerate(cm):
        print(
            f"{CLASS_NAMES[i]:>10s} "
            + " ".join(
                f"{int(v):9d}"
                for v in row
            )
        )


# ============================================================
# TRAIN
# ============================================================

def train(args):

    seed_everything(args.seed)

    clips_root = Path(args.clips)
    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir = (
        output_dir
        / "clip_feature_cache"
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("Loading YOLO...")
    yolo = YOLO(
        args.yolo_model
    )

    object_names = [
        str(
            yolo.names[i]
        )
        for i in range(
            len(yolo.names)
        )
    ]

    print(
        f"YOLO classes "
        f"({len(object_names)}): "
        f"{object_names}"
    )

    if len(object_names) != 8:
        print(
            "WARNING: Expected 8 YOLO "
            "object classes."
        )

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    extractor = FeatureExtractor(
        yolo_model=yolo,
        hand_model_path=args.hand_model,
        object_names=object_names,
    )

    # --------------------------------------------------------
    # Discover and split CLIPS FIRST
    # --------------------------------------------------------

    all_clips = discover_clips(
        clips_root
    )

    print(
        f"\nTotal original clips: "
        f"{len(all_clips)}"
    )

    train_clips, val_clips, test_clips = (
        split_by_clip(
            all_clips,
            args.seed,
        )
    )

    print(
        f"Original clip split: "
        f"train={len(train_clips)} "
        f"val={len(val_clips)} "
        f"test={len(test_clips)}"
    )

    # Save the exact split.
    split_metadata = {
        "train": [
            str(p)
            for p, _ in train_clips
        ],
        "validation": [
            str(p)
            for p, _ in val_clips
        ],
        "test": [
            str(p)
            for p, _ in test_clips
        ],
    }

    (
        output_dir
        / "clip_split.json"
    ).write_text(
        json.dumps(
            split_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Build each split independently.
    # This is the important validation fix.
    # --------------------------------------------------------

    X_train, y_train, _ = build_split_features(
        train_clips,
        "train",
        extractor,
        cache_dir,
        args.rebuild_cache,
    )

    X_val, y_val, _ = build_split_features(
        val_clips,
        "validation",
        extractor,
        cache_dir,
        args.rebuild_cache,
    )

    X_test, y_test, _ = build_split_features(
        test_clips,
        "test",
        extractor,
        cache_dir,
        args.rebuild_cache,
    )

    print(
        f"\nTensor shapes:"
        f"\n  Train: {X_train.shape}"
        f"\n  Val:   {X_val.shape}"
        f"\n  Test:  {X_test.shape}"
    )

    print_split_distribution(
        "TRAIN",
        y_train,
    )

    print_split_distribution(
        "VALIDATION",
        y_val,
    )

    print_split_distribution(
        "TEST",
        y_test,
    )

    # Explicit validation sanity check.
    if len(X_val) == 0:
        raise RuntimeError(
            "Validation set has zero windows."
        )

    if len(
        np.unique(y_val)
    ) != len(CLASS_NAMES):
        missing = [
            CLASS_NAMES[i]
            for i in range(len(CLASS_NAMES))
            if i not in set(
                y_val.tolist()
            )
        ]

        raise RuntimeError(
            "Validation split is missing "
            f"classes: {missing}"
        )

    # --------------------------------------------------------
    # Standardization: TRAIN ONLY
    # --------------------------------------------------------

    mean, std = fit_standardizer(
        X_train
    )

    X_train = standardize(
        X_train,
        mean,
        std,
    )

    X_val = standardize(
        X_val,
        mean,
        std,
    )

    X_test = standardize(
        X_test,
        mean,
        std,
    )

    np.savez_compressed(
        output_dir
        / "feature_standardizer.npz",
        mean=mean,
        std=std,
    )

    (
        output_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            {
                "class_names": CLASS_NAMES,
                "object_class_names": object_names,
                "window_size": WINDOW_SIZE,
                "window_stride": WINDOW_STRIDE,
                "feature_fps": FEATURE_FPS,
                "input_features": int(
                    X_train.shape[-1]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nTraining device: "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    train_loader = DataLoader(
        TemporalDataset(
            X_train,
            y_train,
            training=True,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        TemporalDataset(
            X_val,
            y_val,
            training=False,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        TemporalDataset(
            X_test,
            y_test,
            training=False,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TemporalGRU(
        input_features=X_train.shape[-1]
    ).to(device)

    # Class-balanced loss.
    train_counts = np.bincount(
        y_train,
        minlength=len(CLASS_NAMES),
    ).astype(np.float32)

    weights = (
        train_counts.sum()
        / np.maximum(
            train_counts,
            1.0,
        )
    )

    weights = (
        weights
        / weights.mean()
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        ),
        label_smoothing=0.03,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    )

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0

    checkpoint_path = (
        output_dir
        / "temporal_gru_best.pt"
    )

    print(
        "\nStarting training..."
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        losses = []

        for X, y in train_loader:

            X = X.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        train_loss = float(
            np.mean(losses)
        )

        val_acc, val_f1, _ = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step(
            val_f1
        )

        score = val_f1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "input_features":
                        int(
                            X_train.shape[-1]
                        ),
                    "hidden_size":
                        HIDDEN_SIZE,
                    "num_layers":
                        NUM_LAYERS,
                    "dropout":
                        DROPOUT,
                    "class_names":
                        CLASS_NAMES,
                    "window_size":
                        WINDOW_SIZE,
                    "window_stride":
                        WINDOW_STRIDE,
                },
                checkpoint_path,
            )

        else:
            stale_epochs += 1

            if (
                stale_epochs
                >= PATIENCE
            ):
                print(
                    "\nEarly stopping."
                )
                break

    # --------------------------------------------------------
    # Final test using BEST validation checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_acc, test_f1, test_cm = (
        evaluate(
            model,
            test_loader,
            device,
        )
    )

    print(
        "\n=============================="
    )
    print(
        "FINAL TEST RESULTS"
    )
    print(
        "=============================="
    )

    print(
        f"Best validation macro-F1: "
        f"{best_score:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Test accuracy: "
        f"{test_acc:.4f}"
    )

    print(
        f"Test macro-F1: "
        f"{test_f1:.4f}"
    )

    print_confusion_matrix(
        test_cm
    )

    np.savetxt(
        output_dir
        / "test_confusion_matrix.csv",
        test_cm,
        fmt="%d",
        delimiter=",",
    )

    extractor.close_hand_detector()

    print(
        f"\nSaved model:"
        f"\n{checkpoint_path}"
    )


# ============================================================
# LIVE / VIDEO INFERENCE
# ============================================================

def load_artifacts(
    output_dir: Path,
    yolo: YOLO,
):

    metadata = json.loads(
        (
            output_dir
            / "metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    scaler = np.load(
        output_dir
        / "feature_standardizer.npz"
    )

    trained_object_names = (
        metadata[
            "object_class_names"
        ]
    )

    current_object_names = [
        str(
            yolo.names[i]
        )
        for i in range(
            len(yolo.names)
        )
    ]

    if (
        current_object_names
        != trained_object_names
    ):
        raise RuntimeError(
            "YOLO class mismatch.\n"
            f"Training: {trained_object_names}\n"
            f"Current:  {current_object_names}"
        )

    model = TemporalGRU(
        input_features=int(
            metadata["input_features"]
        )
    )

    checkpoint = torch.load(
        output_dir
        / "temporal_gru_best.pt",
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    return (
        model,
        device,
        metadata,
        scaler["mean"],
        scaler["std"],
    )


def open_source(
    source: str,
):
    if source.isdigit():
        return cv2.VideoCapture(
            int(source)
        )

    return cv2.VideoCapture(
        source
    )


def draw_hands(
    frame,
    hands,
):

    connections = [
        (0, 1), (1, 2),
        (2, 3), (3, 4),
        (0, 5), (5, 6),
        (6, 7), (7, 8),
        (0, 9), (9, 10),
        (10, 11), (11, 12),
        (0, 13), (13, 14),
        (14, 15), (15, 16),
        (0, 17), (17, 18),
        (18, 19), (19, 20),
    ]

    for hand in hands:

        if not hand.present:
            continue

        pts = (
            hand.landmarks_px[:, :2]
            .astype(np.int32)
        )

        for a, b in connections:
            cv2.line(
                frame,
                tuple(pts[a]),
                tuple(pts[b]),
                (0, 220, 0),
                1,
            )

        for point in pts:
            cv2.circle(
                frame,
                tuple(point),
                2,
                (0, 255, 0),
                -1,
            )


def draw_objects(
    frame,
    objects,
    object_names,
):

    for cls_id, item in objects.items():

        if item is None:
            continue

        x1, y1, x2, y2 = map(
            int,
            item["box"],
        )

        conf = item["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 150, 0),
            2,
        )

        cv2.putText(
            frame,
            f"{object_names[cls_id]} "
            f"{conf:.2f}",
            (
                x1,
                max(
                    20,
                    y1 - 7,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
        )


def run_inference(args):

    output_dir = Path(
        args.output
    )

    yolo = YOLO(
        args.yolo_model
    )

    (
        model,
        device,
        metadata,
        mean,
        std,
    ) = load_artifacts(
        output_dir,
        yolo,
    )

    object_names = (
        metadata[
            "object_class_names"
        ]
    )

    extractor = FeatureExtractor(
        yolo,
        args.hand_model,
        object_names,
    )

    # One independent MediaPipe detector for this
    # live/video inference stream.
    extractor.reset_clip()

    cap = open_source(
        args.source
    )

    if not cap.isOpened():
        extractor.close_hand_detector()
        raise RuntimeError(
            f"Could not open "
            f"{args.source}"
        )

    source_fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    sample_period = (
        1.0
        / float(
            metadata[
                "feature_fps"
            ]
        )
    )

    last_sample_time = -1e9
    frame_number = 0

    buffer = deque(
        maxlen=int(
            metadata[
                "window_size"
            ]
        )
    )

    ema = None
    EMA_ALPHA = 0.25

    previous_wall_time = (
        time.perf_counter()
    )

    display_fps = 0.0

    print(
        "\nInference started. "
        "Press Q to quit."
    )

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        video_time = (
            frame_number
            / source_fps
        )

        if (
            video_time
            - last_sample_time
            >= sample_period
        ):

            if last_sample_time < 0:
                dt = 0.0
            else:
                dt = max(
                    1e-6,
                    video_time
                    - last_sample_time,
                )

            vec, hands, objects = (
                extractor.extract(
                    frame,
                    int(
                        round(
                            video_time
                            * 1000.0
                        )
                    ),
                    dt,
                )
            )

            buffer.append(vec)
            last_sample_time = (
                video_time
            )

            if (
                len(buffer)
                == buffer.maxlen
            ):

                seq = np.stack(
                    buffer
                )

                seq = standardize(
                    seq,
                    mean,
                    std,
                )

                X = (
                    torch.from_numpy(
                        seq
                    )
                    .float()
                    .reshape(1, int(metadata["window_size"]), int(metadata["input_features"]))
                    .to(device)
                )

                with torch.no_grad():
                    probabilities = (
                        torch.softmax(
                            model(X),
                            dim=1,
                        )[0]
                        .cpu()
                        .numpy()
                    )

                if ema is None:
                    ema = probabilities
                else:
                    ema = (
                        (1.0 - EMA_ALPHA)
                        * ema
                        + EMA_ALPHA
                        * probabilities
                    )

            current_debug = (
                hands,
                objects,
            )

        else:
            current_debug = None

        if current_debug is not None:
            draw_hands(
                frame,
                current_debug[0],
            )

            draw_objects(
                frame,
                current_debug[1],
                object_names,
            )

        if ema is None:
            label = "WARMING UP"
            confidence = 0.0
        else:
            pred = int(
                np.argmax(ema)
            )

            confidence = float(
                ema[pred]
            )

            label = (
                CLASS_NAMES[pred]
                if confidence >= 0.45
                else "UNCERTAIN"
            )

        cv2.putText(
            frame,
            f"ACTION: {label} "
            f"{confidence:.1%}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"BUFFER: "
            f"{len(buffer)}/"
            f"{buffer.maxlen}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        y = 100

        if ema is None:
            probabilities_to_show = (
                np.zeros(
                    len(CLASS_NAMES)
                )
            )
        else:
            probabilities_to_show = ema

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities_to_show,
        ):

            cv2.putText(
                frame,
                f"{class_name:9s} "
                f"{probability:6.1%}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            y += 24

        now = time.perf_counter()

        instant_fps = (
            1.0
            / max(
                1e-6,
                now
                - previous_wall_time,
            )
        )

        display_fps = (
            instant_fps
            if display_fps == 0.0
            else (
                0.9 * display_fps
                + 0.1 * instant_fps
            )
        )

        previous_wall_time = now

        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f}",
            (20, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            "SIH 2026 - Temporal GRU",
            frame,
        )

        if (
            cv2.waitKey(1)
            & 0xFF
        ) == ord("q"):
            break

        frame_number += 1

    cap.release()
    extractor.close_hand_detector()
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "infer",
        ],
        required=True,
    )

    parser.add_argument(
        "--clips",
        default=DEFAULT_CLIPS,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO,
    )

    parser.add_argument(
        "--hand-model",
        default=DEFAULT_HAND,
    )

    parser.add_argument(
        "--source",
        default="1",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.mode == "train":
        train(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
