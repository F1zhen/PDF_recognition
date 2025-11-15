from typing import List, Dict
from pathlib import Path

from ultralytics import YOLO


class QrDetector:
    def __init__(self, model_path: str, img_size: int = 1024,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.3):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image_path: str) -> List[Dict]:
        results = self.model.predict(
            source=image_path,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), score in zip(xyxy, confs):
                w = x2 - x1
                h = y2 - y1
                detections.append(
                    {
                        "category": "qr",
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )

        return detections
