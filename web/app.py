# web/app.py
import sys
import os
from flask import Flask, render_template, request, url_for, send_from_directory


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate import MODEL_PATH, IMG_SIZE, CONFIDENCE, IOU, DEVICE, WINDOW_SIZE  # Your YOLO script
from invoice import Invoice  # Make sure your invoice.py has Invoice class

from ultralytics import YOLO
import cv2
import cvzone
from collections import defaultdict

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
model = YOLO(MODEL_PATH)
class_names = model.names

PRICE_MAP = {
    "Roti": 10,
    "Dal": 30,
    "Rice": 25,
    "Curd": 15,
    "Vegetable": 40,
    "Salad": 20,
    "Sweet": 25,
    "Pickle": 5
}

@app.route("/", methods=["GET", "POST"])
def index():
    invoice_data = None
    image_filename = None

    if request.method == "POST":
        file = request.files.get("food_image")
        if file:
            # Save uploaded image
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Run YOLO detection
            img = cv2.imread(file_path)
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

                    detected_items.append({
                        "class": class_name,
                        "confidence": round(conf_score, 2),
                        "price": PRICE_MAP.get(class_name, 0)
                    })
                    class_counter[class_name] += 1

                    label = f"{class_name} {conf_score:.2f}"
                    cvzone.cornerRect(img, (x1, y1, w, h), l=12, rt=2)
                    cvzone.putTextRect(
                        img, label,
                        (x1, y1 - 8),
                        scale=1,
                        thickness=2,
                        offset=4
                    )

            image_filename = f"pred_{file.filename}"
            output_path = os.path.join(UPLOAD_FOLDER, image_filename)
            img = cv2.resize(img, WINDOW_SIZE)
            cv2.imwrite(output_path, img)

            # Generate invoice
            invoice = Invoice(user_id="USER_001")  # Fixed user for now
            for item in detected_items:
                for _ in range(class_counter[item["class"]]):
                    invoice.add_item(item["class"], item["price"])
            invoice_data = invoice

    return render_template(
        "index.html",
        invoice=invoice_data,
        image_filename=image_filename
    )


if __name__ == "__main__":
    app.run(debug=True)
