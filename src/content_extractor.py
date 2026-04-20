"""
Вспомогательные валидаторы, форматтеры и crop-операции для пайплайна.

Используется в pipeline.py:
  - cyrillic_ratio, repetition_ratio, table_stats — анти-галлюцинационные метрики
  - format_table_markdown — unified pipe-markdown (forward-fill + multi-level)
  - format_text_markdown — применяет markdown к извлечённому тексту
  - filter_noise_lines — чистка штампов/коротких фрагментов
  - crop_pdf_bbox — рендер bbox страницы в PIL.Image для olmOCR
"""

from __future__ import annotations

import re
from typing import Optional

import fitz
from PIL import Image

from config import LAYOUT_DPI


# ---------------------------------------------------------------------------
# Crop bbox из PDF (для olmOCR fallback)
# ---------------------------------------------------------------------------


def crop_pdf_bbox(
    pdf_doc: fitz.Document,
    page_num: int,
    coords_px: list,
    *,
    max_side: int = 1288,
    pad_pts: float = 2.0,
) -> Optional[Image.Image]:
    """
    Рендерит bbox страницы PDF в PIL.Image.

    coords_px — координаты в пикселях YOLO-растра (LAYOUT_DPI).
    Результат масштабируется так, чтобы длинная сторона = max_side (требование olmOCR).
    """
    try:
        page = pdf_doc[page_num]
    except Exception:
        return None

    # px → PDF points
    s = 72.0 / LAYOUT_DPI
    x1, y1, x2, y2 = coords_px
    rect = fitz.Rect(
        x1 * s - pad_pts, y1 * s - pad_pts, x2 * s + pad_pts, y2 * s + pad_pts
    )

    # Проецируем масштаб рендера: получаем картинку с длинной стороной ≈ max_side
    bbox_w_pts = max(1.0, rect.width)
    bbox_h_pts = max(1.0, rect.height)
    longest_pts = max(bbox_w_pts, bbox_h_pts)
    # scale = max_side / (longest_pts в пикселях 72DPI) = max_side * 72 / (longest_pts * 72) = max_side / longest_pts
    scale = max_side / longest_pts

    mat = fitz.Matrix(scale, scale)
    try:
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    except Exception:
        return None
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------------------------------------------------------------------
# Markdown-таблицы: forward-fill + multi-level header
# ---------------------------------------------------------------------------


def _forward_fill(table: list[list]) -> list[list[str]]:
    """Дублирует значения объединённых ячеек (None/пусто → последнее в строке)."""
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


_NUMERIC_RE = re.compile(r"^[\d\s.,+\-/%№()]+$")


def _is_text_row(row: list[str]) -> bool:
    non_empty = [c for c in row if c.strip()]
    if not non_empty:
        return False
    return all(not _NUMERIC_RE.fullmatch(c) for c in non_empty)


def format_table_markdown(table: list[list], n_header_rows: int = 0) -> str:
    """Markdown-таблица с многоуровневыми заголовками (header1_header2)."""
    if not table:
        return ""

    filled = _forward_fill(table)
    if not filled or not filled[0]:
        return ""

    if n_header_rows >= 2:
        header_rows = 2
    elif n_header_rows == 1:
        header_rows = 1
    else:
        header_rows = 1
        if len(filled) >= 3 and _is_text_row(filled[0]) and _is_text_row(filled[1]):
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

    def pad(row: list, width: int) -> list[str]:
        row = list(row) + [""] * width
        return [str(c) for c in row[:width]]

    lines = [
        "| " + " | ".join(pad(headers, n)) + " |",
        "| " + " | ".join(["---"] * n) + " |",
    ]
    for row in data_rows:
        lines.append("| " + " | ".join(pad(row, n)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Форматирование текстовых блоков
# ---------------------------------------------------------------------------


def format_text_markdown(text: str, block_type: str, heading_level: int = 0) -> str:
    """Применяет Markdown-разметку к извлечённому тексту блока."""
    text = text.strip()
    if not text:
        return ""

    if block_type in ("title", "section-header") and heading_level > 0:
        single_line = " ".join(l.strip() for l in text.splitlines() if l.strip())
        return "#" * heading_level + " " + single_line

    if block_type == "list-item":
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^[•▪▸\-\*]\s", line) or re.match(r"^\d+[.)]\s", line):
                out.append(line)
            else:
                out.append(f"- {line}")
        return "\n".join(out)

    return text


# ---------------------------------------------------------------------------
# Фильтрация мусорных строк (штампы/водяные знаки)
# ---------------------------------------------------------------------------

_STAMP_FRAG_RE = re.compile(r"^[А-ЯЁA-Z]{2,6}$")
_WATERMARK_RE = re.compile(
    r"^(ЧЕРНОВИК|DRAFT|CONFIDENTIAL|КОНФИДЕНЦИАЛЬНО|НЕ\s+ДЛЯ\s+РАСПРОСТРАНЕНИЯ)[\s\d\W]*$",
    re.I | re.UNICODE,
)


def filter_noise_lines(text: str, min_chars: int = 3) -> str:
    """Убирает коротко-мусорные строки и фрагменты водяных знаков/штампов."""
    if not text:
        return ""
    lines = []
    for l in text.splitlines():
        stripped = l.strip()
        if len(stripped.replace(" ", "")) < min_chars:
            continue
        if _STAMP_FRAG_RE.match(stripped):
            continue
        if _WATERMARK_RE.match(stripped):
            continue
        lines.append(l)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Анти-галлюцинационные метрики
# ---------------------------------------------------------------------------

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


def cyrillic_ratio(text: str) -> float:
    """Доля кириллических букв среди всех букв. 0.0 если букв нет."""
    if not text:
        return 0.0
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    cyr = _CYRILLIC_RE.findall(text)
    return len(cyr) / len(letters)


def repetition_ratio(text: str) -> float:
    """Доля повторов самой частой непустой строки среди всех непустых строк."""
    if not text:
        return 0.0
    lines = [l.strip().lower() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    return max(counts.values()) / len(lines)


def table_stats(md: str) -> dict:
    """
    Структурный разбор pipe-markdown таблицы.
    Возвращает {'n_cols', 'n_rows', 'empty_ratio', 'max_cell', 'row_repeat_ratio'}.
    """
    stats = {
        "n_cols": 0,
        "n_rows": 0,
        "empty_ratio": 0.0,
        "max_cell": 0,
        "row_repeat_ratio": 0.0,
    }
    if not md:
        return stats

    rows: list[list[str]] = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s.startswith("|"):
            continue
        if re.match(r"^\|\s*:?-{2,}", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return stats

    n_cols = max(len(r) for r in rows)
    n_rows = len(rows)
    total = sum(len(r) for r in rows) or 1
    empty = sum(1 for r in rows for c in r if not c)
    max_cell = max((len(c) for r in rows for c in r), default=0)

    row_keys = ["|".join(r) for r in rows]
    row_counts: dict[str, int] = {}
    for k in row_keys:
        row_counts[k] = row_counts.get(k, 0) + 1
    row_repeat = (max(row_counts.values()) / len(row_keys)) if row_keys else 0.0

    stats.update(
        n_cols=n_cols,
        n_rows=n_rows,
        empty_ratio=empty / total,
        max_cell=max_cell,
        row_repeat_ratio=row_repeat,
    )
    return stats
