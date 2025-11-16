from typing import List, Dict
from pathlib import Path
import os
import tempfile

from PIL import Image

from .signature_detector import SignatureDetector
from .stamp_detectop import StampDetector
from .qr_detector import QrDetector


def bbox_iou_xywh(b1, b2) -> float:
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

        self.iou_nms = inf_cfg["iou_nms"]
        self.stamp_with_signature_iou = inf_cfg["stamp_with_signature_iou"]

        #ДЕТЕКТОР ПОДПИСИ НА ВСЕЙ СТРАНИЦЕ
        self.signature_global = SignatureDetector(
            paths["signature"],
            img_size=inf_cfg["img_size"],
            conf_threshold=inf_cfg["conf_threshold"]["signature_global"],
            iou_threshold=inf_cfg["iou_nms"]["signature"],
        )

        #ДЕТЕКТОР ПОДПИСИ НА CROP ПЕЧАТИ
        # можно уменьшить img_size и порог
        self.signature_in_stamp = SignatureDetector(
            paths["signature"],
            img_size=inf_cfg.get("img_size_stamp", 512),
            conf_threshold=inf_cfg["conf_threshold"]["signature_in_stamp"],
            iou_threshold=inf_cfg["iou_nms"]["signature"],
        )

        #ДЕТЕКТОР ПЕЧАТЕЙ
        self.stamp = StampDetector(
            paths["stamp"],
            img_size=inf_cfg["img_size"],
            conf_threshold=inf_cfg["conf_threshold"]["stamp"],
            iou_threshold=inf_cfg["iou_nms"]["stamp"],
        )

        #ДЕТЕКТОР QR
        self.qr = QrDetector(
            paths["qr"],
            img_size=inf_cfg["img_size"],
            conf_threshold=inf_cfg["conf_threshold"]["qr"],
            iou_threshold=inf_cfg["iou_nms"]["qr"],
        )

    #ВСПОМОГАТЕЛЬНОЕ: ДЕТЕКТ ПОДПИСЕЙ ВНУТРИ PEЧАТЕЙ

    def _detect_signatures_inside_stamps(
        self,
        image_path: str,
        stamps: List[Dict],
    ) -> List[Dict]:
        """
        Для каждой печати:
          - делаем crop
          - прогоняем signature_in_stamp
          - переносим bbox подписи обратно в координаты страницы
        """
        if not stamps:
            return []

        img = Image.open(image_path).convert("RGB")
        h_img, w_img = img.size[1], img.size[0]  # (w, h) -> (h, w) если надо

        sigs_from_crops: List[Dict] = []

        for idx, st in enumerate(stamps):
            x, y, w, h = st["bbox"]
            #аккуратно приводим к int и обрезаем границы
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(int(x + w), w_img)
            y2 = min(int(y + h), h_img)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            #временный файл для вторичного детектора
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp:
                crop_path = tmp.name
                crop.save(crop_path)

            local_sigs = self.signature_in_stamp.predict(crop_path)

            #очистим временный файл
            try:
                os.remove(crop_path)
            except OSError:
                pass

            #переносим координаты в глобальные
            for sg in local_sigs:
                lx, ly, lw, lh = sg["bbox"]
                gx = lx + x1
                gy = ly + y1
                sg_global = sg.copy()
                sg_global["bbox"] = [gx, gy, lw, lh]
                sg_global["source"] = "signature_in_stamp"
                sigs_from_crops.append(sg_global)

        return sigs_from_crops

    #ОСНОВНОЙ МЕТОД 

    def detect_on_image(self, image_path: str) -> List[Dict]:
        #базовые детекциями
        sigs_global = self.signature_global.predict(image_path)
        stamps = self.stamp.predict(image_path)
        qrs = self.qr.predict(image_path)

        #подписи внутри печатей (второй проход)
        sigs_from_crops = self._detect_signatures_inside_stamps(
            image_path, stamps
        )

        #склеиваем все подписи и делаем NMS
        sigs_all_raw = sigs_global + sigs_from_crops
        sigs = nms_per_class(sigs_all_raw, self.iou_nms["signature"])

        stamps = nms_per_class(stamps, self.iou_nms["stamp"])
        qrs = nms_per_class(qrs, self.iou_nms["qr"])

        all_dets = sigs + stamps + qrs
        all_dets = self._add_stamp_with_signature_flag(all_dets)

        return all_dets

    def _add_stamp_with_signature_flag(self, dets: List[Dict]) -> List[Dict]:
        stamps = [d for d in dets if d["category"] == "stamp"]
        sigs = [d for d in dets if d["category"] == "signature"]

        for st in stamps:
            for sg in sigs:
                if bbox_iou_xywh(st["bbox"], sg["bbox"]) > self.stamp_with_signature_iou:
                    st["stamp_with_signature"] = True
                    break
        return dets
