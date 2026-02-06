from ultralytics import YOLO
import cv2
import cvzone
import torch
import os

MODEL_PATH = r"/runs/detect/train5/weights/best.pt"
CONFIDENCE = 0.15
IOU = 0.45
IMG_SIZE = 640
WINDOW_SIZE = (960, 720)
CAMERA_ID = 0


assert os.path.exists(MODEL_PATH), "Model file not found"

print("CUDA Available:", torch.cuda.is_available())


model = YOLO(MODEL_PATH)
class_names = model.names

current_device = 0 if torch.cuda.is_available() else "cpu"
print(f"Starting on device: {current_device}")


cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=current_device,
        verbose=False
    )

    detected_items = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls_id]

            detected_items.append(f"{class_name} ({conf:.2f})")

            label = f"{class_name} {conf:.2f}"

            cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2)
            cvzone.putTextRect(
                frame, label,
                (x1, y1 - 10),
                scale=1,
                thickness=2,
                offset=4
            )

    # Display detected items (top-left)
    y_offset = 30
    for item in detected_items:
        cv2.putText(
            frame,
            item,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y_offset += 30

    # Device info overlay
    device_text = f"DEVICE: {'GPU' if current_device == 0 else 'CPU'}"
    cv2.putText(
        frame,
        device_text,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    frame = cv2.resize(frame, WINDOW_SIZE)
    cv2.imshow("YOLO Food Detection (Webcam)", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Switch to GPU
    elif key == ord('g') and torch.cuda.is_available():
        current_device = 0
        print("Switched to GPU")

    # Switch to CPU
    elif key == ord('c'):
        current_device = "cpu"
        print("Switched to CPU")

cap.release()
cv2.destroyAllWindows()
