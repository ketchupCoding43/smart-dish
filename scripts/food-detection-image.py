from ultralytics import YOLO
import cv2
import cvzone
import torch
import os
from collections import defaultdict

MODEL_PATH = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\runs\detect\train5\weights\best.pt"
IMAGE_PATH = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\data\raw-data\T1_aug_247.jpg"
OUTPUT_PATH = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\data\raw-data\predicted_plate.jpg"

CONFIDENCE = 0.15
IOU = 0.45
IMG_SIZE = 640
DEVICE = 0            # 0 = GPU, 'cpu' for CPU
WINDOW_SIZE = (720, 720)

assert os.path.exists(MODEL_PATH), "Model file not found"
assert os.path.exists(IMAGE_PATH), "Image file not found"

print("CUDA available:", torch.cuda.is_available())

model = YOLO(MODEL_PATH)
class_names = model.names

img = cv2.imread(IMAGE_PATH)

results = model.predict(
    source=img,
    conf=CONFIDENCE,
    iou=IOU,
    imgsz=IMG_SIZE,
    device=DEVICE,
    verbose=False
)

detections = defaultdict(list)

for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names[cls_id]

        # store confidence per class
        detections[class_name].append(conf)

        label = f"{class_name} {conf:.2f}"

        cvzone.cornerRect(img, (x1, y1, w, h), l=12, rt=2)
        cvzone.putTextRect(
            img,
            label,
            (x1, y1 - 8),
            scale=1,
            thickness=2,
            offset=4
        )

print("\nDetected Food Items Summary")
print("-" * 40)

if len(detections) == 0:
    print("No food items detected.")
else:
    for item, confs in detections.items():
        quantity = len(confs)
        avg_conf = sum(confs) / quantity

        print(f"Item        : {item}")
        print(f"Quantity    : {quantity}")
        print(f"Confidence  : {[f'{c:.2f}' for c in confs]}")
        print(f"Avg Conf    : {avg_conf:.2f}")
        print("-" * 40)

img = cv2.resize(img, WINDOW_SIZE)
cv2.imshow("Food Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(OUTPUT_PATH, img)
print(f"\nSaved output → {OUTPUT_PATH}")
