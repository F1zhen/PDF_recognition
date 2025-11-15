from pathlib import Path
import shutil

from ultralytics import YOLO


def download_and_save(model_name: str, output_path: str):
    print(f"Downloading {model_name} → {output_path}")
    model = YOLO(model_name)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # сохраняем best веса
    shutil.copy(model.ckpt_path, out_path)


def main():
    # примеры, подставь реальные имена моделей с HF/Ultralytics Hub
    download_and_save(
        "keremberke/yolov8m-signature-detection",
        "models/signature/yolov8m-signature-detection.pt",
    )
    download_and_save(
        "keremberke/yolov8s-forge-stamp-detection",
        "models/stamp/yolov8s-forge-stamp-detection.pt",
    )
    download_and_save(
        "ultralytics/yolov8n-qr",
        "models/qr/yolov8n-qr.pt",
    )


if __name__ == "__main__":
    main()
