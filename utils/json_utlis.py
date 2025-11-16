import json
from pathlib import Path
from typing import Dict


def build_results_dict(all_docs_predictions: Dict) -> Dict:
    result = {}
    for doc_name, pages in all_docs_predictions.items():
        doc_entry = {}
        annotation_counter = 1   # счётчик как в примере selected_annotations

        for page_num, page_info in pages.items():
            width, height = page_info["size"]
            detections = page_info["detections"]

            # 🔴 ВАЖНО: если на странице нет детекций — вообще не добавляем этот page_X
            if not detections:
                continue

            page_key = f"page_{page_num}"

            page_entry = {
                "annotations": [],
                "page_size": {
                    "width": int(width),
                    "height": int(height),
                },
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

                # score, stamp_with_signature — не добавляем
                page_entry["annotations"].append({ann_key: ann_data})

            # Добавляем страницу только если есть аннотации (на всякий случай)
            if page_entry["annotations"]:
                doc_entry[page_key] = page_entry

        # Если у документа нет ни одной страницы с аннотациями — не включаем его
        if doc_entry:
            result[doc_name] = doc_entry

    return result


def save_results_json(all_docs_predictions: Dict, output_path: str):
    data = build_results_dict(all_docs_predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
