from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    model = YOLO("../yolov8n.pt")

    model.train(
        data=r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\data\data.yaml",
        epochs=50,
        imgsz=512,
        batch=4,
        device=0,
        workers=2,
        amp=False,
        plots=False,
        cache=False,
        val=False
    )

if __name__ == "__main__":
    main()
