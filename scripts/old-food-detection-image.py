from ultralytics import YOLO
import cv2
import cvzone
import os


MODEL_PATH = r"/runs/detect/train4/weights/best.pt"
IMAGE_PATH = r"/data/raw-data/T1_aug_241.jpg"
OUTPUT_PATH = r"/data/raw-data/predicted_plate.jpg"
CONFIDENCE = 0.01  # minimum confidence to display detection
WINDOW_SIZE = (720, 720)  # resize display window


if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


model = YOLO(MODEL_PATH)
class_names = model.names


img = cv2.imread(IMAGE_PATH)

# Optional: resize for faster detection (YOLO input)
# img = cv2.resize(img, (640, 640))


results = model(img, conf=CONFIDENCE)


for r in results:
    if r.boxes is None:
        continue
    for box in r.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w, h = x2 - x1, y2 - y1
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls]} {conf:.2f}"


        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(0, 255, 0))
        cvzone.putTextRect(img, label, (x1, y1 - 10), scale=1, thickness=2, offset=3)

# ====== RESIZE AND DISPLAY ======
img_resized = cv2.resize(img, WINDOW_SIZE)
cv2.namedWindow("Food Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Food Detection", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ====== SAVE OUTPUT ======
cv2.imwrite(OUTPUT_PATH, img)
print(f"Prediction saved to {OUTPUT_PATH}")
