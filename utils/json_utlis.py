import json
from pathlib import Path
from typing import Dict, List


def build_results_dict(all_docs_predictions: Dict) -> Dict:
    """
    all_docs_predictions:
    {
      "file.pdf": {
        page_num(int): {
          "size": (width, height),
          "detections": [
            {
              "category": "signature"/"stamp"/"qr",
              "bbox": [x, y, w, h],
              "score": 0.87,
              # опционально: "stamp_with_signature": True
            }, ...
          ]
        }, ...
      }, ...
    }
    """
    result = {}

    for doc_name, pages in all_docs_predictions.items():
        doc_entry = {}
        annotation_counter = 1

        for page_num, page_info in pages.items():
            width, height = page_info["size"]
            detections = page_info["detections"]

            page_key = f"page_{page_num}"
            page_entry = {
                "page_size": {"width": width, "height": height},
                "annotations": [],
            }

            for det in detections:
                x, y, w, h = det["bbox"]
                area = float(w) * float(h)

                ann_key = f"annotation_{annotation_counter}"
                annotation_counter += 1

                ann_data = {
                    "category": det["category"],
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                    },
                    "area": area,
                }

                # если какие-то флаги/score нужны
                if "score" in det:
                    ann_data["score"] = float(det["score"])
                if det.get("stamp_with_signature"):
                    ann_data["stamp_with_signature"] = True

                page_entry["annotations"].append({ann_key: ann_data})

            doc_entry[page_key] = page_entry

        result[doc_name] = doc_entry

    return result


def save_results_json(all_docs_predictions: Dict, output_path: str):
    data = build_results_dict(all_docs_predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
