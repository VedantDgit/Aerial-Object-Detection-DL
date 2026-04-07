from ultralytics import YOLO
import os

# 🔹 Ensure working directory (important if running from root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Path to data.yaml
data_yaml = os.path.join(BASE_DIR, "data.yaml")

# 🔹 Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")   # lightweight model

print("🚀 Starting YOLOv8 Training...")
print(f"Using data config: {data_yaml}")

# 🔹 Train model
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo_bird_drone",
    device="cpu"   # change to 'cuda' if GPU available
)

print("✅ Training Completed!")

# 🔹 Validate model
metrics = model.val()
print("📊 Validation Metrics:")
print(metrics)

# 🔹 Save best model path
print("\n💾 Best model saved at:")
print("runs/detect/yolo_bird_drone3/weights/best.pt")