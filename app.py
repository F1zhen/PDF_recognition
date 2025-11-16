from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yaml

from utils.pds_utils import pdf_to_images
from utils.viz_utils import draw_boxes
from utils.json_utils import build_results_dict
from src.detectors.ensemble import EnsembleDetector

app = FastAPI(title="Digital Inspector API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG_PATH = Path("config/config.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ENSEMBLE = EnsembleDetector(CFG)


@app.post("/inspect_pdf")
async def inspect_pdf(file: List[UploadFile] = File(...)):
    tmp_dir = Path("data/tmp_api")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_docs_predictions = {}

    for file in file:
        pdf_path = tmp_dir / file.filename

        content = await file.read()
        pdf_path.write_bytes(content)

        pages_info = pdf_to_images(str(pdf_path), CFG["paths"]["page_images"])

        all_docs_predictions[file.filename] = {}

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

    result_dict = build_results_dict(all_docs_predictions)

    return JSONResponse(content=result_dict)