from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch

MODEL_PATH = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\runs\detect\train5\weights\best.pt"
DATA_YAML = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\data\data.yaml"
DEVICE = 0  # 0 = GPU, use 'cpu' if needed


def main():
    print("CUDA available:", torch.cuda.is_available())

    print("\nLoading model...")
    model = YOLO(MODEL_PATH)

    print("\nRunning validation to compute metrics and generate graphs...\n")

    # 🔥 This generates PR curve, F1 curve, confusion matrix, etc.
    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        conf=0.001,
        iou=0.6,
        device=DEVICE,
        plots=True,
        project=r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\evaluation_results",
        name="food_model_eval"
    )

    # 📊 Extract overall metrics
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    print("\n📈 MODEL PERFORMANCE SUMMARY")
    print(f"Precision      : {precision:.3f}")
    print(f"Recall         : {recall:.3f}")
    print(f"mAP@0.5        : {map50:.3f}")
    print(f"mAP@0.5:0.95   : {map5095:.3f}")

    # 📉 Simple bar chart (for report/presentation)
    labels = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    values = [precision, recall, map50, map5095]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("YOLO Food Detection Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis='y')

    graph_path = r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\evaluation_results\food_model_eval\performance_summary.png"
    plt.savefig(graph_path)
    plt.show()

    print(f"\nSummary graph saved at → {graph_path}")
    print("\nAll advanced YOLO graphs saved in:")
    print(r"C:\Users\Soham Sathe\PycharmProjects\YOLO-food-project\evaluation_results\food_model_eval")


# ✅ Required on Windows to prevent multiprocessing crash
if __name__ == "__main__":
    main()
