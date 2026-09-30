from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor


# Set up SAM 3
overrides = {
    "model": "sam3.pt",
    "task": "segment",
    "mode": "predict",
    "imgsz": 644,
    "quantize": 16,
    "conf": 0.40,
    "save": True
}

predictor = SAM3SemanticPredictor(overrides=overrides)


# Folder containing the images
image_folder = Path("F:/sih/unannotated images")


# Each prompt corresponds to a class ID
prompts = [
    "orange circular cap",
    "red computer mouse",
    "small black pebble shaped object except on red computer mouse",
    "dark blue square",
    "yellow square",
    "pink square",
    "green square",
    "white box",
    "person",
    "hand"
]


# Process every image
for image_path in image_folder.iterdir():

    # Ignore files that are not images
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    print(f"Processing: {image_path.name}")

    # Give the image to SAM 3
    predictor.set_image(str(image_path))

    # Ask SAM 3 to find our objects
    results = predictor(text=prompts)

    # Get the first result
    result = results[0]

    # Get original image height and width
    image_height, image_width = result.orig_shape

    # Create a label file with the same name as the image
    # Example:
    # photo_1.jpg → photo_1.txt
    label_path = image_path.with_suffix(".txt")

    # Open the file so we can write the detected boxes into it
    with open(label_path, "w") as file:

        # Go through every detected bounding box
        for box, class_id in zip(
            result.boxes.xyxy,
            result.boxes.cls
        ):

            # SAM 3 gives the box as:
            # x1, y1, x2, y2
            x1, y1, x2, y2 = box.tolist()

            # Convert SAM 3 class ID to an integer
            class_id = int(class_id.item())

            # Convert the box to YOLO format
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height

            box_width = (x2 - x1) / image_width
            box_height = (y2 - y1) / image_height

            # Write the bounding box into the .txt file
            #
            # YOLO format:
            # class_id center_x center_y width height
            file.write(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

    print(f"Saved labels: {label_path.name}")


print("Finished!")