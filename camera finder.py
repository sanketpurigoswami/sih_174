import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera index {i}: WORKING")
        else:
            print(f"Camera index {i}: detected but no frame")

        cap.release()
    else:
        print(f"Camera index {i}: unavailable")