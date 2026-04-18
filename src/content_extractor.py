"""
Извлечение контента из PDF: текст (PyMuPDF) и таблицы (pdfplumber).
Все координаты из routing plan — в пикселях при 400 DPI.
"""
import re

import cv2
import fitz
import numpy as np
import pdfplumber
from PIL import Image

RENDER_DPI = 400
PDF_DPI = 72.0
_SCALE = PDF_DPI / RENDER_DPI  # pixel → PDF points


def _to_pdf_rect(coords: list) -> tuple:
    x1, y1, x2, y2 = coords
    return (x1 * _SCALE, y1 * _SCALE, x2 * _SCALE, y2 * _SCALE)


# ---------------------------------------------------------------------------
# Текст
# ---------------------------------------------------------------------------

def extract_text_block(pdf_doc: fitz.Document, page_num: int, coords: list) -> tuple[str, float]:
    """
    Возвращает (text, avg_font_size) для заданной области страницы.
    """
    page = pdf_doc[page_num]
    rect = fitz.Rect(*_to_pdf_rect(coords))
    text_dict = page.get_text("dict", clip=rect)

    lines, sizes = [], []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    line_parts.append(t)
                    sizes.append(span.get("size", 12.0))
            if line_parts:
                lines.append(" ".join(line_parts))

    text = "\n".join(lines)
    avg_size = sum(sizes) / len(sizes) if sizes else 12.0
    return text, avg_size


OCR_RENDER_DPI = 300  # Выше = лучше видны шрифты, но дороже по памяти.


def render_block_for_ocr(pdf_doc: fitz.Document, page_num: int, coords: list, dpi: int = OCR_RENDER_DPI):
    """Рендерит регион страницы в PIL Image для OCR (по умолчанию 300 DPI)."""
    from PIL import Image

    page = pdf_doc[page_num]
    scale = dpi / 72.0
    rect = fitz.Rect(*_to_pdf_rect(coords))
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def collect_heading_sizes(pdf_doc: fitz.Document, routing_plan: dict) -> list[float]:
    """
    Собирает уникальные размеры шрифтов из всех блоков title/section-header.
    Возвращает список, отсортированный по убыванию.
    """
    sizes: set[float] = set()
    for page_data in routing_plan.get("pages", []):
        pnum = page_data["page_num"] - 1
        page = pdf_doc[pnum]
        for block in page_data.get("blocks", []):
            if block["type"] not in ("title", "section-header"):
                continue
            rect = fitz.Rect(*_to_pdf_rect(block["coords"]))
            td = page.get_text("dict", clip=rect)
            for b in td.get("blocks", []):
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.add(round(span.get("size", 12.0), 1))
    return sorted(sizes, reverse=True)


def detect_heading_level(font_size: float, heading_sizes: list[float]) -> int:
    """1–4 по размеру шрифта относительно всех заголовков документа."""
    for i, s in enumerate(heading_sizes[:4]):
        if font_size >= s * 0.88:
            return i + 1
    return 4


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------

def extract_table(pdf_path: str, page_num: int, coords: list) -> list[list]:
    """
    Извлекает таблицу из PDF через pdfplumber.
    Возвращает сырые данные (list of rows) или [].
    """
    x1, y1, x2, y2 = [c * _SCALE for c in coords]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            cropped = page.crop((
                max(0, x1 - 5),
                max(0, y1 - 5),
                min(page.width, x2 + 5),
                min(page.height, y2 + 5),
            ))
            # Сначала пробуем строгое определение рамок
            for strategy in (
                {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
                {"vertical_strategy": "lines",        "horizontal_strategy": "lines"},
                {"vertical_strategy": "text",         "horizontal_strategy": "text"},
            ):
                table = cropped.extract_table({**strategy, "snap_tolerance": 5, "join_tolerance": 5})
                if table:
                    return table
    except Exception:
        pass
    return []


def _forward_fill(table: list[list]) -> list[list[str]]:
    """Дублирует содержимое объединённых ячеек (None → предыдущее значение)."""
    result = []
    for row in table:
        new_row, last = [], ""
        for cell in row:
            if cell is None or (isinstance(cell, str) and not cell.strip()):
                new_row.append(last)
            else:
                val = str(cell).strip().replace("\n", " ")
                last = val
                new_row.append(val)
        result.append(new_row)
    return result


def format_table_markdown(table: list[list]) -> str:
    """
    Преобразует данные таблицы в Markdown.
    Обрабатывает: объединённые ячейки, многоуровневые заголовки.
    """
    if not table:
        return ""

    filled = _forward_fill(table)
    if not filled or not filled[0]:
        return ""

    # Определяем, есть ли многоуровневый заголовок
    header_rows = 1
    if len(filled) >= 3:
        r0, r1 = filled[0], filled[1]
        # Если строка 1 не содержит «чисто числовых» ячеек
        # и часть её ячеек повторяет строку 0 → многоуровневый заголовок
        non_numeric = all(not re.fullmatch(r"[\d\s.,+\-/%]+", c) or c == "" for c in r1)
        repeats = sum(1 for c in r1 if c in r0 and c != "")
        if non_numeric and repeats > max(1, len(r1) * 0.25):
            header_rows = 2

    if header_rows == 2:
        headers = [
            f"{h1}_{h2}" if (h1 and h2 and h1 != h2) else (h1 or h2)
            for h1, h2 in zip(filled[0], filled[1])
        ]
        data_rows = filled[2:]
    else:
        headers = filled[0]
        data_rows = filled[1:]

    n = len(headers)
    if n == 0:
        return ""

    def pad(row: list, n: int) -> list[str]:
        row = list(row) + [""] * n
        return [str(c) for c in row[:n]]

    lines = [
        "| " + " | ".join(pad(headers, n)) + " |",
        "| " + " | ".join(["---"] * n) + " |",
    ]
    for row in data_rows:
        lines.append("| " + " | ".join(pad(row, n)) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Форматирование текстового блока
# ---------------------------------------------------------------------------

def format_text_markdown(text: str, block_type: str, heading_level: int = 0) -> str:
    """Применяет Markdown-разметку к извлечённому тексту."""
    text = text.strip()
    if not text:
        return ""

    if block_type in ("title", "section-header") and heading_level > 0:
        # Объединяем строки в одну — перенос строки в заголовке ломает Markdown
        single_line = " ".join(l.strip() for l in text.splitlines() if l.strip())
        return "#" * heading_level + " " + single_line

    if block_type == "list-item":
        result = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^[•▪▸\-\*]\s", line) or re.match(r"^\d+[.)]\s", line):
                result.append(line)
            else:
                result.append(f"- {line}")
        return "\n".join(result)

    return text


# ---------------------------------------------------------------------------
# Анализ изображения: определить, содержит ли оно плотный текст / таблицу
# ---------------------------------------------------------------------------

def estimate_text_density(img: Image.Image) -> float:
    """
    Оценка «текстовой плотности» изображения по доле тёмных пикселей
    после адаптивной бинаризации. Нужна, чтобы отличить фото/схему
    от скана или растровой таблицы.

    Возвращает float 0..1 — долю «чернильных» пикселей.
    """
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return float(binary.sum()) / (255.0 * binary.size)


def looks_like_table(text: str) -> bool:
    """Эвристика: есть ли в тексте табличные разделители."""
    if not text:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    col_like = sum(1 for l in lines if re.search(r"(\s{2,}|\|)\S+(\s{2,}|\|)", l))
    return col_like >= 2


def filter_noise_lines(text: str, min_chars: int = 3) -> str:
    """
    Убирает «строки-шум» — короткие капли символов ('К', 'Ч', '|'),
    которые PyMuPDF иногда выдаёт из-за буквиц и декоративных элементов.
    """
    if not text:
        return ""
    lines = [
        l for l in text.splitlines()
        if len(l.strip().replace(" ", "")) >= min_chars
    ]
    return "\n".join(lines).strip()
