from ultralytics import YOLO


# Load the YOLO model that we trained on our dataset
model = YOLO("F:/sih/runs/detect/train/weights/best.pt")


# Run object detection on a test image
# YOLO will draw bounding boxes around detected objects.
results = model.predict(
    source="F:/sih/unannotated images/test.jpg",
    save=True,
    conf=0.25
)


print("Detection finished!")