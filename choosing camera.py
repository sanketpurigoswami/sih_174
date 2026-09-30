import cv2

print("Opening camera 1 with Media Foundation...")

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

print("isOpened:", cap.isOpened())

if not cap.isOpened():
    print("Could not open camera 1.")
    raise SystemExit

print("Camera 1 opened. Trying to read frames...")

for i in range(100):
    ret, frame = cap.read()

    print(f"Frame {i}: ret={ret}", end="")

    if frame is not None:
        print(f", shape={frame.shape}")
    else:
        print(", frame=None")

    if ret and frame is not None:
        cv2.imshow("GlideX Camera 1", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()