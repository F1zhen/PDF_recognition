import argparse
import os
from pathlib import Path
import yaml

from utils.pds_utils import pdf_to_images
from utils.viz_utils import draw_boxes
from utils.json_utils import save_results_json
from src.detectors.ensemble import EnsembleDetector


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_pdf(pdf_path: str, cfg) -> dict:
    paths_cfg = cfg["paths"]
    page_images_dir = paths_cfg["page_images"]
    viz_dir = paths_cfg["output_viz"]

    pages_info = pdf_to_images(pdf_path, page_images_dir)
    ensemble = EnsembleDetector(cfg)

    doc_predictions = {}

    for page_num, info in pages_info.items():
        img_path = info["image_path"]
        width = info["width"]
        height = info["height"]

        detections = ensemble.detect_on_image(img_path)

        # сохраним для JSON
        doc_predictions[page_num] = {
            "size": (width, height),
            "detections": detections,
        }

        # визуализация
        out_viz_path = Path(viz_dir) / f"{Path(pdf_path).stem}_page_{page_num:04d}_viz.png"
        draw_boxes(img_path, detections, str(out_viz_path))

    return doc_predictions


def main():
    parser = argparse.ArgumentParser(description="Digital Inspector inference")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file or directory with PDFs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to output JSON file (if None, will use paths.output_json)",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    pdf_input = Path(args.pdf)
    all_docs_predictions = {}

    if pdf_input.is_file():
        docs = [pdf_input]
    else:
        docs = sorted(pdf_input.glob("*.pdf"))

    for pdf in docs:
        doc_pred = process_pdf(str(pdf), cfg)
        all_docs_predictions[pdf.name] = doc_pred

    output_json_dir = cfg["paths"]["output_json"]
    if args.output_json is None:
        output_path = Path(output_json_dir) / "predictions.json"
    else:
        output_path = Path(args.output_json)

    save_results_json(all_docs_predictions, str(output_path))
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
