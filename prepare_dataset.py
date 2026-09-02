from pathlib import Path
import random
import shutil


# Original folder containing your images and YOLO label files
source_folder = Path("F:/sih/unannotated images")

# New folder where YOLO will read the training dataset
dataset_folder = Path("F:/sih/yolo_dataset")


# Create the folders required by YOLO
train_images = dataset_folder / "images" / "train"
val_images = dataset_folder / "images" / "val"

train_labels = dataset_folder / "labels" / "train"
val_labels = dataset_folder / "labels" / "val"

for folder in [train_images, val_images, train_labels, val_labels]:
    folder.mkdir(parents=True, exist_ok=True)


# Find all images in the original folder
images = [
    path for path in source_folder.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

print(f"Found {len(images)} images.")


# Shuffle the images so the train/validation split is random
random.seed(42)
random.shuffle(images)


# Use 80% for training and 20% for validation
split_index = int(len(images) * 0.8)

train_set = images[:split_index]
val_set = images[split_index:]


# Copy images and their matching label files
def copy_files(image_list, image_destination, label_destination):

    for image_path in image_list:

        # The YOLO label must have the same filename as the image
        # but with a .txt extension.
        label_path = image_path.with_suffix(".txt")

        # Don't copy an image if it has no label
        if not label_path.exists():
            print(f"WARNING: No label for {image_path.name}")
            continue

        shutil.copy2(
            image_path,
            image_destination / image_path.name
        )

        shutil.copy2(
            label_path,
            label_destination / label_path.name
        )


# Copy training files
copy_files(
    train_set,
    train_images,
    train_labels
)


# Copy validation files
copy_files(
    val_set,
    val_images,
    val_labels
)


print(f"Training images: {len(train_set)}")
print(f"Validation images: {len(val_set)}")
print("Dataset preparation finished!")