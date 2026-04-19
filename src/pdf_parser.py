import json
import logging
import hashlib
import os
from pathlib import Path

import fitz
import numpy as np
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("/app/data/.ocr_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class PDFParserOCR:
    def __init__(self) -> None:
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            import torch
            import easyocr
            use_gpu = torch.cuda.is_available()
            model_dir = os.getenv("EASYOCR_MODULE_PATH", "/app/volumes/easyocr_models")
            logger.info(f"初始化 EasyOCR，GPU={'启用' if use_gpu else 'CPU'}，模型目录: {model_dir}")
            self._ocr = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=use_gpu,
                model_storage_directory=model_dir,
            )
            logger.info("EasyOCR 初始化完成")
        return self._ocr

    def _get_cache_path(self, pdf_path: str) -> Path:
        stat = Path(pdf_path).stat()
        key = f"{pdf_path}_{stat.st_size}_{stat.st_mtime}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return CACHE_DIR / f"{hash_key}.json"

    def _extract_text_with_pymupdf(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)
        result = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            result.append({"page": page_num, "content": text})
        doc.close()
        return result

    def _extract_text_with_ocr(self, pdf_path: str) -> list[dict]:
        ocr = self._get_ocr()
        # DPI 提高到 250，提升识别准确率
        images = convert_from_path(pdf_path, dpi=250)
        result = []
        for page_num, image in enumerate(images, start=1):
            logger.info(f"OCR 处理第 {page_num}/{len(images)} 页，尺寸: {image.size}")
            img_array = np.array(image)
            # detail=1 保留位置信息，paragraph=False 逐行识别更准确
            detections = ocr.readtext(img_array, detail=1, paragraph=False)
            # 按 Y 坐标排序，保证文字顺序正确
            detections.sort(key=lambda x: x[0][0][1])
            page_text = "\n".join([d[1] for d in detections if d[2] > 0.3])  # 置信度 > 0.3
            logger.info(f"第 {page_num} 页完成，识别字符数: {len(page_text)}")
            result.append({"page": page_num, "content": page_text})
        return result

    def parse_pdf(self, pdf_path: str) -> list[dict]:
        cache_path = self._get_cache_path(pdf_path)
        if cache_path.exists():
            logger.info(f"命中 OCR 缓存，直接加载: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info(f"开始解析: {pdf_path}")
        pages = self._extract_text_with_pymupdf(pdf_path)
        total_chars = sum(len(p["content"]) for p in pages)

        if total_chars > 100:
            logger.info(f"检测到内嵌文字（共 {total_chars} 字），跳过 OCR")
        else:
            logger.info("文字内容过少，启用 OCR...")
            pages = self._extract_text_with_ocr(pdf_path)
            # OCR 完成后立即释放模型，归还 GPU/CPU 内存
            self._release_ocr()

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
        logger.info(f"OCR 结果已缓存: {cache_path}")
        return pages

    def _release_ocr(self) -> None:
        """释放 EasyOCR 模型，归还内存"""
        if self._ocr is not None:
            del self._ocr
            self._ocr = None
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("EasyOCR 模型已释放")