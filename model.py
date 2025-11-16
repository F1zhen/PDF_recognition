from ultralytics import YOLO

model = YOLO("models/signature/yolov8s.pt")

print("Wrapper names:", model.names)
print("Inner model names:", getattr(model.model, 'names', None))
print("Overrides:", getattr(model, 'overrides', None))
