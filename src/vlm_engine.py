"""
VLM (Qwen2.5-VL-3B-Instruct) для сложных блоков: растровые таблицы,
сканы, рукописный текст, rasterized PDF, и для описания обычных
изображений. Модель ленива: загружается только при первом вызове.
"""
from __future__ import annotations

import os
from typing import Optional, Union

from PIL import Image

from device import get_torch_device, get_torch_dtype, setup_environment

setup_environment()

_MODEL_ID_DEFAULT = "Qwen/Qwen2.5-VL-3B-Instruct"

# Промпты подобраны так, чтобы модель возвращала markdown без комментариев.
_PROMPTS = {
    "text": (
        "Извлеки весь текст с изображения на русском и английском языке. "
        "Сохрани абзацное деление, списки и заголовки. "
        "Нумерованные списки начинай с '1.', маркированные — с '- '. "
        "Если это страница документа — не добавляй колонтитулы, номера страниц и водяные знаки. "
        "Верни ТОЛЬКО извлечённый текст, без пояснений и без оформления ```markdown```."
    ),
    "table": (
        "На изображении таблица. Преобразуй её в Markdown-таблицу. "
        "Правила: (1) многоуровневые заголовки объединяй через нижнее подчёркивание "
        "(например, 'Q1_Выручка'); (2) объединённые ячейки дублируй содержимым; "
        "(3) пустые ячейки оставляй пустыми. "
        "Верни ТОЛЬКО markdown-таблицу, без пояснений и без оформления ```markdown```."
    ),
    "handwritten": (
        "На изображении рукописный текст на русском языке. "
        "Распознай его максимально точно, сохранив абзацное деление. "
        "Верни ТОЛЬКО распознанный текст, без пояснений."
    ),
    "caption": (
        "Опиши одной короткой фразой (до 10 слов), что изображено. "
        "Не используй символы ], [, ( и ), никаких markdown-конструкций. "
        "Отвечай на русском языке."
    ),
    "analyze": (
        "Что на изображении? Ответь ОДНИМ словом из списка: "
        "'text' — только текст/документ/скан; "
        "'table' — таблица; "
        "'handwritten' — рукописный текст; "
        "'picture' — фотография, схема, график, рисунок. "
        "Верни одно слово без пояснений."
    ),
}


class VLMEngine:
    """Singleton-обёртка над Qwen2.5-VL."""

    _instance: Optional["VLMEngine"] = None

    def __init__(self, model_id: str = _MODEL_ID_DEFAULT):
        self.model_id = model_id
        self.device = get_torch_device()
        self.dtype = get_torch_dtype(self.device)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    @classmethod
    def get(cls, model_id: str = _MODEL_ID_DEFAULT) -> "VLMEngine":
        if cls._instance is None:
            cls._instance = cls(model_id)
        return cls._instance

    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        print(f"  [VLM] Загрузка {self.model_id} на {self.device} ({self.dtype})...")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True
        )
        # На MPS device_map="auto" иногда падает — переносим вручную.
        self._model = model.to(self.device).eval()
        # Подрезаем число визуальных токенов — 1280 даёт хороший баланс скорость/качество.
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        print("  [VLM] Готово.")

    # ------------------------------------------------------------------
    def _generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        self._load()

        if isinstance(image, str) and not os.path.exists(image):
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------

    def extract_text(self, image: Union[str, Image.Image]) -> str:
        """Плотный текст / скан / rasterized_pdf."""
        return _strip_code_fence(self._generate(image, _PROMPTS["text"], 3072))

    def extract_table(self, image: Union[str, Image.Image]) -> str:
        """Растровая таблица → markdown."""
        return _strip_code_fence(self._generate(image, _PROMPTS["table"], 3072))

    def extract_handwritten(self, image: Union[str, Image.Image]) -> str:
        """Рукописный русский."""
        return _strip_code_fence(self._generate(image, _PROMPTS["handwritten"], 2048))

    def short_caption(self, image: Union[str, Image.Image]) -> str:
        """Короткий alt-текст (до ~10 слов)."""
        raw = self._generate(image, _PROMPTS["caption"], 64)
        raw = raw.replace("]", "").replace("[", "").replace("(", "").replace(")", "")
        return raw.splitlines()[0][:120].strip() if raw else "image"

    def classify(self, image: Union[str, Image.Image]) -> str:
        """Возвращает один из: 'text', 'table', 'handwritten', 'picture'."""
        raw = self._generate(image, _PROMPTS["analyze"], 8).strip().lower()
        for key in ("handwritten", "table", "text", "picture"):
            if key in raw:
                return key
        return "picture"


# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    """Убирает обёртки ```markdown ... ``` которые Qwen иногда добавляет."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # убираем первую строку ```lang
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # убираем закрывающую ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
