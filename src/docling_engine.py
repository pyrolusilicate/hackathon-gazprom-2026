"""
Docling wrapper: layout + TableFormer + OCR (EasyOCR RU/EN).

Основной движок пайплайна. Обрабатывает PDF целиком: detects layout, parses
text, extracts tables via TableFormer, rasterizes figures, и OCR'ит сканы.

Singleton по умолчанию — первая инициализация качает модели (~400MB).
"""

from __future__ import annotations

import gc
from typing import Optional

from device import is_cuda_available


class DoclingEngine:
    """Singleton-обёртка над Docling DocumentConverter."""

    _instance: Optional["DoclingEngine"] = None

    @classmethod
    def get(cls) -> "DoclingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            EasyOcrOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        device = AcceleratorDevice.CUDA if is_cuda_available() else AcceleratorDevice.CPU

        opts = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=EasyOcrOptions(
                lang=["ru", "en"],
                use_gpu=is_cuda_available(),
            ),
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=4,
            ),
            generate_picture_images=True,
            images_scale=2.0,
        )
        opts.table_structure_options.do_cell_matching = True

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        print(f"  [Docling] ready ({device.value if hasattr(device, 'value') else device})")

    def convert(self, pdf_path: str):
        """Возвращает DoclingDocument — основной payload пайплайна."""
        result = self._converter.convert(pdf_path)
        return result.document

    @staticmethod
    def page_is_sparse(doc, page_num: int, *, min_chars: int = 30) -> bool:
        """
        True, если страница пустая/почти пустая — кандидат на olmOCR fallback.

        Считаем только text-items данной страницы; если на ней есть таблица
        или picture, не считаем sparse (TableFormer/layout уже поработал).
        """
        try:
            texts = []
            has_structure = False
            for item, _ in doc.iterate_items():
                prov = getattr(item, "prov", None)
                if not prov:
                    continue
                if not any(p.page_no == page_num for p in prov):
                    continue
                label = getattr(item, "label", "")
                label_str = str(label).lower()
                if "table" in label_str or "picture" in label_str:
                    has_structure = True
                    break
                text = getattr(item, "text", "") or ""
                if text.strip():
                    texts.append(text)
            if has_structure:
                return False
            total = sum(len(t) for t in texts)
            return total < min_chars
        except Exception:
            return False

    def release(self) -> None:
        """Освобождает VRAM после конвертации (перед загрузкой olmOCR)."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
