from typing import List, Dict
import numpy as np

from .signature_detector import SignatureDetector
# from .stamp_detectop import StampDetector
from .qr_detector import QrDetector


def bbox_iou_xywh(b1, b2) -> float:
    # b = [x, y, w, h]
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa1, ya1 = x1, y1
    xa2, ya2 = x1 + w1, y1 + h1
    xb1, yb1 = x2, y2
    xb2, yb2 = x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


def nms_per_class(dets: List[Dict], iou_thresh: float) -> List[Dict]:
    if not dets:
        return []

    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []

    while dets_sorted:
        best = dets_sorted.pop(0)
        kept.append(best)
        dets_sorted = [
            d
            for d in dets_sorted
            if bbox_iou_xywh(best["bbox"], d["bbox"]) < iou_thresh
        ]
    return kept


class EnsembleDetector:
    def __init__(self, cfg):
        paths = cfg["paths"]["models"]
        inf_cfg = cfg["inference"]

        self.signature = SignatureDetector(
            paths["signature"],
            img_size=inf_cfg["img_size"],
            conf_threshold=inf_cfg["conf_threshold"]["signature"],
            iou_threshold=inf_cfg["iou_nms"]["signature"],
        )
        # self.stamp = StampDetector(
        #     paths["stamp"],
        #     img_size=inf_cfg["img_size"],
        #     conf_threshold=inf_cfg["conf_threshold"]["stamp"],
        #     iou_threshold=inf_cfg["iou_nms"]["stamp"],
        # )
        self.qr = QrDetector(
            paths["qr"],
            img_size=inf_cfg["img_size"],
            conf_threshold=inf_cfg["conf_threshold"]["qr"],
            iou_threshold=inf_cfg["iou_nms"]["qr"],
        )

        self.stamp_with_signature_iou = inf_cfg["stamp_with_signature_iou"]
        self.iou_nms = inf_cfg["iou_nms"]

    def detect_on_image(self, image_path: str) -> List[Dict]:
        sigs = self.signature.predict(image_path)
        # stamps = self.stamp.predict(image_path)
        qrs = self.qr.predict(image_path)

        # NMS по классам
        sigs = nms_per_class(sigs, self.iou_nms["signature"])
        # stamps = nms_per_class(stamps, self.iou_nms["stamp"])
        qrs = nms_per_class(qrs, self.iou_nms["qr"])

        all_dets = sigs + qrs
        all_dets = self._add_stamp_with_signature_flag(all_dets)

        return all_dets

    def _add_stamp_with_signature_flag(self, dets: List[Dict]) -> List[Dict]:
        # сейчас печатей нет, но код оставляем на будущее
        stamps = [d for d in dets if d["category"] == "stamp"]
        sigs = [d for d in dets if d["category"] == "signature"]

        for st in stamps:
            for sg in sigs:
                if bbox_iou_xywh(st["bbox"], sg["bbox"]) > self.stamp_with_signature_iou:
                    st["stamp_with_signature"] = True
                    break
        return dets
