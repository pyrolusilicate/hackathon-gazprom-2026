"""
EasyOCR-обёртка с предварительным шумоподавлением.

Примечания:
- EasyOCR задействует DataLoader с pin_memory=True, что даёт предупреждение
  на MPS. Подавляем через device.setup_environment().
- Реальный GPU-инференс EasyOCR поддерживает только на CUDA; на MPS и CPU
  работает на CPU.
"""
import ssl

import cv2
import numpy as np
import torch
from PIL import Image

from device import easyocr_uses_gpu, get_torch_device, setup_environment

setup_environment()

# macOS Python не поставляется с системными сертификатами.
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001


def _denoise_rgb(img: Image.Image) -> np.ndarray:
    """
    Шумоподавление перед OCR:
      1. fastNlMeansDenoising по яркостному каналу (сохраняет контуры букв),
      2. Лёгкая нормализация контраста (CLAHE),
      3. Возврат RGB-массива (EasyOCR ожидает RGB).
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # h=7 — мягкое шумоподавление; более агрессивно «съедает» тонкие шрифты.
    denoised = cv2.fastNlMeansDenoising(
        gray, h=7, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _lines_from_results(results) -> list[str]:
    """EasyOCR возвращает (bbox, text) в paragraph-режиме и (bbox, text, conf) — иначе."""
    lines: list[str] = []
    for r in results:
        if len(r) >= 3:
            _, text, conf = r[0], r[1], r[2]
            if conf is None or conf > 0.2:
                lines.append(str(text))
        elif len(r) == 2:
            lines.append(str(r[1]))
    return lines


class OCREngine:
    """EasyOCR singleton (ru + en)."""

    _reader = None

    def __init__(self):
        self.device = get_torch_device()
        self._ensure_reader()

    def _ensure_reader(self):
        if OCREngine._reader is None:
            import easyocr

            OCREngine._reader = easyocr.Reader(
                ["ru", "en"], gpu=easyocr_uses_gpu(self.device)
            )

    def read_image(self, image_path: str) -> str:
        try:
            img = Image.open(image_path)
            return self.read_pil(img)
        except Exception:
            return ""

    def read_pil(self, img: Image.Image) -> str:
        try:
            arr = _denoise_rgb(img)
            results = OCREngine._reader.readtext(arr, paragraph=True, width_ths=0.7)
            return "\n".join(_lines_from_results(results))
        except Exception:
            return ""
