from ultralytics import YOLO


# Load YOLO11 nano with pretrained weights
model = YOLO("yolo11n.pt")


# Fine-tune YOLO on our custom dataset
model.train(
    data="F:/sih/yolo_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    device=0,
    workers=0
)


print("Training finished!")