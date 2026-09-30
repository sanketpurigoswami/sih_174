from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("temporal_clips")
OUTPUT_DIR = Path("har_model")

SEED = 42

# Every video is uniformly sampled to this many frames.
#
# Your clips are 60-180 frames, so 48 is a good balance.
NUM_FRAMES = 48

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 25

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LSTM_HIDDEN_SIZE = 256

LSTM_LAYERS = 2

DROPOUT = 0.30

NUM_WORKERS = 0

# Minimum confidence isn't relevant here because we're training
# on videos rather than YOLO detections.

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

# Train / validation / test split.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# CLASS DISCOVERY
# ============================================================

def discover_classes(
    dataset_dir: Path,
) -> List[Tuple[int, str, Path]]:

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{dataset_dir.resolve()}"
        )

    folders = [
        p
        for p in dataset_dir.iterdir()
        if p.is_dir()
    ]

    discovered = []

    for folder in folders:

        # Expected:
        #
        # 0_GRASP
        # 1_IDLE
        # 2_PICKUP
        #
        match = re.match(
            r"^(\d+)_(.+)$",
            folder.name,
        )

        if match is None:

            print(
                f"WARNING: Ignoring folder "
                f"with unexpected name: {folder.name}"
            )

            continue

        class_id = int(match.group(1))
        class_name = match.group(2)

        discovered.append(
            (
                class_id,
                class_name,
                folder,
            )
        )

    discovered.sort(
        key=lambda x: x[0]
    )

    if len(discovered) != 6:

        raise RuntimeError(
            "\nExpected exactly 6 action folders.\n"
            f"Found {len(discovered)}:\n"
            + "\n".join(
                f"  {x[0]}_{x[1]}"
                for x in discovered
            )
        )

    expected_ids = list(range(6))

    actual_ids = [
        x[0]
        for x in discovered
    ]

    if actual_ids != expected_ids:

        raise RuntimeError(
            "Action folders must use IDs "
            "0 through 5.\n"
            f"Found IDs: {actual_ids}"
        )

    return discovered


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def collect_videos(
    classes,
) -> List[Tuple[str, int]]:

    videos = []

    for class_id, class_name, folder in classes:

        class_videos = [
            p
            for p in folder.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]

        class_videos.sort()

        if not class_videos:

            raise RuntimeError(
                f"No videos found in: {folder}"
            )

        print(
            f"{class_id}: {class_name:12s} "
            f"{len(class_videos):3d} videos"
        )

        for video in class_videos:

            videos.append(
                (
                    str(video),
                    class_id,
                )
            )

    return videos


# ============================================================
# CHECK VIDEO
# ============================================================

def get_video_frame_count(
    video_path: str,
) -> int:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        cap.release()

        return 0

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return frame_count


# ============================================================
# UNIFORM FRAME SAMPLING
# ============================================================

def load_sampled_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            f"Video contains no frames: "
            f"{video_path}"
        )

    # Uniformly sample frames.
    #
    # Example:
    # 60-frame video -> 48 selected frames
    # 180-frame video -> 48 selected frames
    #
    # Temporal order is preserved.
    indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames,
        dtype=np.int64,
    )

    frames = []

    for frame_index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = cap.read()

        if not success:

            # Retry by using the last successfully decoded
            # frame instead of crashing.
            if frames:

                frame = frames[-1].copy()

            else:

                cap.release()

                raise RuntimeError(
                    f"Could not read frame "
                    f"{frame_index} from "
                    f"{video_path}"
                )

        # OpenCV gives BGR.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    cap.release()

    return np.stack(
        frames,
        axis=0,
    )


# ============================================================
# FRAME PREPROCESSING
# ============================================================

class FramePreprocessor:

    def __init__(self):

        weights = ResNet18_Weights.DEFAULT

        self.transform = weights.transforms()

    def process(
        self,
        frames: np.ndarray,
    ) -> torch.Tensor:

        processed = []

        for frame in frames:

            image = torch.from_numpy(
                frame.copy()
            )

            # H x W x C -> C x H x W
            image = image.permute(
                2,
                0,
                1,
            )

            image = image.float() / 255.0

            image = self.transform(
                image
            )

            processed.append(
                image
            )

        return torch.stack(
            processed
        )


# ============================================================
# DATASET
# ============================================================

class VideoDataset(Dataset):

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        preprocessor: FramePreprocessor,
    ):

        self.samples = samples
        self.preprocessor = preprocessor

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):

        video_path, label = self.samples[index]

        try:

            frames = load_sampled_frames(
                video_path,
                NUM_FRAMES,
            )

            frames = self.preprocessor.process(
                frames
            )

            return (
                frames,
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
            )

        except Exception as error:

            raise RuntimeError(
                f"\nFailed processing video:\n"
                f"{video_path}\n"
                f"Error: {error}"
            ) from error


# ============================================================
# HAR MODEL
# ============================================================

class CNNBiLSTM(nn.Module):

    def __init__(
        self,
        num_classes: int,
    ):

        super().__init__()

        # ----------------------------------------------------
        # CNN
        #
        # Pretrained ResNet18 extracts visual features
        # from every frame.
        # ----------------------------------------------------

        weights = ResNet18_Weights.DEFAULT

        backbone = resnet18(
            weights=weights
        )

        # Remove ImageNet classification layer.
        self.cnn = nn.Sequential(
            *list(
                backbone.children()
            )[:-1]
        )

        # ResNet18 output:
        # 512 features per frame.
        feature_size = 512

        # Freeze CNN initially.
        #
        # This is important because the dataset is relatively
        # small (~300-360 clips).
        for parameter in self.cnn.parameters():

            parameter.requires_grad = False

        # ----------------------------------------------------
        # TEMPORAL MODEL
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=(
                DROPOUT
                if LSTM_LAYERS > 1
                else 0.0
            ),
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                LSTM_HIDDEN_SIZE * 2,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Input:
        #
        # B x T x C x H x W
        #
        # Example:
        # 8 x 48 x 3 x 224 x 224

        batch_size, time_steps, channels, height, width = (
            x.shape
        )

        # ----------------------------------------------------
        # Flatten time dimension so CNN sees each frame.
        # ----------------------------------------------------

        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        # ----------------------------------------------------
        # CNN feature extraction
        # ----------------------------------------------------

        with torch.no_grad():

            x = self.cnn(
                x
            )

        # B*T x 512 x 1 x 1
        x = x.flatten(
            start_dim=1
        )

        # ----------------------------------------------------
        # Restore temporal dimension.
        # ----------------------------------------------------

        x = x.reshape(
            batch_size,
            time_steps,
            -1,
        )

        # B x T x 512
        #       ↓
        # BiLSTM
        #       ↓
        # B x T x 512
        lstm_output, _ = self.lstm(
            x
        )

        # We use the final temporal representation.
        features = lstm_output[:, -1, :]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            features
        )

        return logits


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    predictions = []
    targets = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for frames, labels in progress:

        frames = frames.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            frames
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predicted = torch.argmax(
            logits,
            dim=1,
        )

        predictions.extend(
            predicted.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        targets,
        predictions,
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    predictions = []
    targets = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False,
        )

        for frames, labels in progress:

            frames = frames.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                frames
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(
                predicted.cpu()
                .numpy()
                .tolist()
            )

            targets.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

    loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    return (
        loss,
        accuracy,
        targets,
        predictions,
    )


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def save_training_curves(
    history,
    output_dir: Path,
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HAR Training Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "loss_curve.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("HAR Training Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "accuracy_curve.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    targets,
    predictions,
    class_names,
    output_dir,
):

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(
                len(class_names)
            )
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        cm
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        range(len(class_names))
    )

    ax.set_yticks(
        range(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "HAR Confusion Matrix"
    )

    for row in range(cm.shape[0]):

        for col in range(cm.shape[1]):

            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "confusion_matrix.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("             TEMPORAL HUMAN ACTIVITY RECOGNITION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # DISCOVER CLASSES
    # --------------------------------------------------------

    classes = discover_classes(
        DATASET_DIR
    )

    class_names = [
        class_name
        for _, class_name, _
        in classes
    ]

    print()
    print("Actions:")

    for class_id, class_name, _ in classes:

        print(
            f"  {class_id}: {class_name}"
        )

    print()

    # --------------------------------------------------------
    # COLLECT ALL VIDEOS
    # --------------------------------------------------------

    samples = collect_videos(
        classes
    )

    print()
    print(
        f"Total videos: {len(samples)}"
    )

    print()

    # --------------------------------------------------------
    # DATA DISTRIBUTION
    # --------------------------------------------------------

    labels = np.array(
        [
            label
            for _, label
            in samples
        ]
    )

    # First split:
    #
    # 70% train
    # 30% temporary
    #

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(
            VAL_RATIO + TEST_RATIO
        ),
        random_state=SEED,
        stratify=labels,
    )

    # --------------------------------------------------------
    # Second split temporary into validation/test.
    # --------------------------------------------------------

    temp_labels = np.array(
        [
            label
            for _, label
            in temp_samples
        ]
    )

    relative_test_ratio = (
        TEST_RATIO
        /
        (VAL_RATIO + TEST_RATIO)
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_ratio,
        random_state=SEED,
        stratify=temp_labels,
    )

    print(
        f"Train      : {len(train_samples)}"
    )

    print(
        f"Validation : {len(val_samples)}"
    )

    print(
        f"Test       : {len(test_samples)}"
    )

    print()

    # --------------------------------------------------------
    # Save split information.
    #
    # This is useful so you know exactly which videos were
    # used for training/validation/test.
    # --------------------------------------------------------

    split_data = {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples,
    }

    with open(
        OUTPUT_DIR / "dataset_split.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            split_data,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    print(
        "Loading ImageNet preprocessing..."
    )

    preprocessor = FramePreprocessor()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = VideoDataset(
        train_samples,
        preprocessor,
    )

    val_dataset = VideoDataset(
        val_samples,
        preprocessor,
    )

    test_dataset = VideoDataset(
        test_samples,
        preprocessor,
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Creating CNN + BiLSTM model..."
    )

    model = CNNBiLSTM(
        num_classes=len(class_names)
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    #
    # CNN is frozen, so only LSTM/classifier train.
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # LR scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0

    best_epoch = -1

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_accuracy
        )

        # ----------------------------------------------------
        # Store history.
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        print()
        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.7f}"
        )

        # ----------------------------------------------------
        # Save best model.
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_frames": NUM_FRAMES,
                "image_size": IMAGE_SIZE,
                "lstm_hidden_size": LSTM_HIDDEN_SIZE,
                "lstm_layers": LSTM_LAYERS,
                "dropout": DROPOUT,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
            }

            torch.save(
                checkpoint,
                OUTPUT_DIR
                / "best_har_model.pt",
            )

            print(
                "✓ New best model saved."
            )

    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    # --------------------------------------------------------
    # Save curves
    # --------------------------------------------------------

    save_training_curves(
        history,
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        OUTPUT_DIR
        / "best_har_model.pt",
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(
        device
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print(
        "Evaluating best model on TEST set..."
    )

    (
        test_loss,
        test_accuracy,
        test_targets,
        test_predictions,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Test Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        test_targets,
        test_predictions,
        labels=list(
            range(
                len(class_names)
            )
        ),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        report
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        test_targets,
        test_predictions,
        class_names,
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Save class names.
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "class_names.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            class_names,
            file,
            indent=2,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 70
    )

    print(
        OUTPUT_DIR
        / "best_har_model.pt"
    )

    print(
        OUTPUT_DIR
        / "class_names.json"
    )

    print(
        OUTPUT_DIR
        / "dataset_split.json"
    )

    print(
        OUTPUT_DIR
        / "training_history.json"
    )

    print(
        OUTPUT_DIR
        / "classification_report.txt"
    )

    print(
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    print(
        OUTPUT_DIR
        / "loss_curve.png"
    )

    print(
        OUTPUT_DIR
        / "accuracy_curve.png"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()