from ultralytics import YOLO

# Load the default model
# model = YOLO("F:/sih/yolo11n.pt")
model = YOLO("F:/sih/runs/detect/train/weights/best.pt")

# Run detection with a much lower confidence threshold to catch weak predictions
results = model.predict(
    source="F:/sih/testfull.jpg",
    save=True,
    conf=0.5,  # Lowered from 0.25 to 0.05 to catch top-down objects
)

# Print out exactly what boxes (if any) were found to your terminal
for r in results:
    if len(r.boxes) == 0:
        print("❌ YOLO still found 0 objects at this confidence level.")
    for box in r.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"🎯 Found: {label} with {confidence*100:.1f}% confidence!")
