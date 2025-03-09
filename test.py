from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on the source
results = model(source=0, stream=True)  # generator of Results objects``