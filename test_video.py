from ultralytics import YOLO


# Load our trained YOLO model
model = YOLO("F:/sih/runs/detect/train/weights/best.pt")


# Run detection on the video.
# YOLO processes the video one frame at a time
# and places bounding boxes around recognized objects.
results = model.predict(
    source="F:/sih/unannotated images/test_video.mp4",
    save=True,
    conf=0.25
)


print("Video detection finished!")