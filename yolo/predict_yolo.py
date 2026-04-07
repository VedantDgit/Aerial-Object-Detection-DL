from ultralytics import YOLO
import os

model = YOLO("runs/detect/yolo_bird_drone3/weights/best.pt")

# Use an existing image from the dataset (path built relative to this script)
base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, "object_detection_Dataset", "test", "images", "pic_1044_jpg.rf.eb4fcaf978190d647cd6fbaa4bf605e1.jpg")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"{image_path} does not exist")

results = model(image_path, save=True)

for r in results:
    r.show()