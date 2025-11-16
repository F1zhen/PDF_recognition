import os
from pathlib import Path
from typing import List, Tuple, Dict

import fitz  
from tqdm import tqdm


def pdf_to_images(pdf_path: str, output_dir: str) -> Dict[int, Dict]:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_info = {}

    for page_idx in tqdm(range(len(doc)), desc=f"PDF→IMG {pdf_path.name}"):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=72)  
        page_num = page_idx + 1

        img_name = f"{pdf_path.stem}_page_{page_num:04d}.png"
        img_path = output_dir / img_name
        pix.save(str(img_path))

        pages_info[page_num] = {
            "image_path": str(img_path),
            "width": pix.width,
            "height": pix.height,
        }

    doc.close()
    return pages_info
