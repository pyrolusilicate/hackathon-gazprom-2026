"""
PaddleOCR CPU-обёртка для RASTER_TEXT.

Зачем: DeepSeek-OCR-2 — тяжёлый ~7B VLM, занимающий GPU. Простые текстовые
блоки (подпись, абзац, list-item) не требуют VLM — PaddleOCR на CPU
распознаёт их в 10–20x быстрее и специально обучен на кириллице.

GPU при этом остаётся свободен под RASTER_TABLE / VECTOR_TABLE-fallback /
SMART_FIGURE, где VLM действительно нужен.

Модель грузится лениво при первом вызове; singleton через `get()`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


class PaddleEngine:
    """Singleton-обёртка над PaddleOCR."""

    _instance: Optional["PaddleEngine"] = None

    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # noqa: WPS433 (ленивый импорт)

        # use_angle_cls исправляет повёрнутый текст; lang='ru' включает
        # русский словарь (PP-OCRv3 понимает русский из коробки).
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ru",
            use_gpu=False,
            show_log=False,
        )
        print("  [Paddle] PP-OCRv3 ready (CPU, ru)")

    @classmethod
    def get(cls) -> "PaddleEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ocr_text(self, img: Image.Image, *, min_conf: float = 0.5) -> str:
        """Plain-text OCR: строки в порядке top-down, через \\n."""
        if img is None:
            return ""
        arr = np.array(img.convert("RGB"))

        try:
            result = self._ocr.ocr(arr, cls=True)
        except Exception:
            return ""

        if not result or not result[0]:
            return ""

        lines: list[str] = []
        for entry in result[0]:
            if not entry or len(entry) < 2:
                continue
            text_block = entry[1]
            if not text_block or len(text_block) < 2:
                continue
            text, conf = text_block[0], text_block[1]
            if text and conf >= min_conf:
                lines.append(str(text).strip())
        return "\n".join(l for l in lines if l)
