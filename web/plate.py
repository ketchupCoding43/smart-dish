from ultralytics import YOLO
import cv2
import cvzone
from collections import defaultdict
import os

MODEL_PATH = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\runs\detect\train5\weights\best.pt"
CONFIDENCE = 0.15
IOU = 0.45
IMG_SIZE = 640
DEVICE = 0
WINDOW_SIZE = (720, 720)

assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
model = YOLO(MODEL_PATH)
class_names = model.names

def detect_plate(img_path):
    img = cv2.imread(img_path)
    results = model.predict(
        source=img,
        conf=CONFIDENCE,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    detected_items = []
    class_counter = defaultdict(int)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = class_names[cls_id]

            # Save detection info
            detected_items.append({
                "class": class_name,
                "confidence": round(conf_score, 2)
            })
            class_counter[class_name] += 1

            # Draw on image
            label = f"{class_name} {conf_score:.2f}"
            cvzone.cornerRect(img, (x1, y1, w, h), l=12, rt=2)
            cvzone.putTextRect(img, label, (x1, y1 - 8), scale=1, thickness=2, offset=4)

    return detected_items, class_counter, img
