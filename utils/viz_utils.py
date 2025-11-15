from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np


# простая цветовая схема по категориям
COLOR_MAP = {
    "signature": (0, 255, 0),   # зелёный
    "stamp": (255, 0, 0),       # синий
    "qr": (0, 0, 255),          # красный
}


def draw_boxes(
    image_path: str,
    detections: List[Dict],
    output_path: str,
    thickness: int = 2,
):
    """
    detections: список словарей
        {
          "category": "signature"/"stamp"/"qr",
          "bbox": [x, y, w, h],
          "score": 0.87
        }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    for det in detections:
        x, y, w, h = det["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        category = det["category"]
        score = det.get("score", 0.0)

        color = COLOR_MAP.get(category, (255, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"{category} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
