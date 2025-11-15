from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import yaml

from utils.pds_utils import pdf_to_images
from utils.viz_utils import draw_boxes
from utils.json_utlis import save_results_json
from src.detectors.ensemble import EnsembleDetector


app = FastAPI(title="Digital Inspector API")

CFG_PATH = Path("config/config.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ENSEMBLE = EnsembleDetector(CFG)


@app.post("/inspect_pdf")
async def inspect_pdf(file: UploadFile = File(...)):
    """
    Принимает PDF, возвращает JSON-аннотации для всех страниц.
    """
    # сохраним временно
    tmp_dir = Path("data/tmp_api")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / file.filename

    content = await file.read()
    pdf_path.write_bytes(content)

    pages_info = pdf_to_images(str(pdf_path), CFG["paths"]["page_images"])

    all_docs_predictions = {file.filename: {}}
    for page_num, info in pages_info.items():
        img_path = info["image_path"]
        width = info["width"]
        height = info["height"]

        detections = ENSEMBLE.detect_on_image(img_path)

        all_docs_predictions[file.filename][page_num] = {
            "size": (width, height),
            "detections": detections,
        }

        viz_name = f"{Path(file.filename).stem}_page_{page_num:04d}_viz.png"
        viz_path = Path(CFG["paths"]["output_viz"]) / viz_name
        draw_boxes(img_path, detections, str(viz_path))

    json_data = build_results_dict(all_docs_predictions)
    return JSONResponse(content=json_data)
