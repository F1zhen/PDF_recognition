from typing import List, Dict
from ultralytics import YOLO


class StampDetector:
    def __init__(self, model_path: str, img_size: int, conf_threshold: float, iou_threshold: float):
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.conf = conf_threshold
        self.iou = iou_threshold

      
        names = {int(k): v for k, v in self.model.names.items()}
        self.stamp_id = None
        for idx, name in names.items():
            if name.lower() == "stamp":
                self.stamp_id = idx
                break

        if self.stamp_id is None:
            raise ValueError(f"'stamp' class not found in model.names: {names}")

    def predict(self, image_path: str) -> List[Dict]:
        results = self.model.predict(
            source=image_path,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            classes=[self.stamp_id],  # <<< ТОЛЬКО печати
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1

                detections.append(
                    {
                        "category": "stamp",
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(box.conf[0]),
                    }
                )

        return detections
