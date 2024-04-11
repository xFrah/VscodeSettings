from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

help(results)
