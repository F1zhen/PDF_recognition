from typing import List, Dict
from pathlib import Path

from ultralytics import YOLO
import numpy as np


class SignatureDetector:
    """
    Детектор подписей на основе YOLO-модели.
    Использует только класс 0 (signature), даже если модель обучена на нескольких классах.
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = 1024,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image_path: str) -> List[Dict]:
        """
        Запускает детекцию на изображении и возвращает только bounding box'ы класса 0 (signature).

        :param image_path: путь до изображения
        :return: список словарей с ключами:
                 - "category": "signature"
                 - "bbox": [x, y, w, h]
                 - "score": float
        """
        # Фильтрация по классам уже на этапе предсказания (только класс 0)
        results = self.model.predict(
            source=image_path,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # брать только класс 0
            verbose=False,
        )

        detections: List[Dict] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            # Координаты боксов
            xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) -> x1, y1, x2, y2
            confs = boxes.conf.cpu().numpy()  # (N,)
            classes = boxes.cls.cpu().numpy()  # (N,)

            for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, classes):
                # На всякий случай дополнительно фильтруем по классу 0
                if int(cls) != 0:
                    continue

                w = x2 - x1
                h = y2 - y1

                detections.append(
                    {
                        "category": "signature",
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )

        return detections
