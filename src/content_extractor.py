"""
Извлечение контента из PDF.

Треки:
- VECTOR_TEXT  — PyMuPDF text dict по bbox
- RASTER_TABLE — DeepSeek-OCR-2 (все таблицы через VLM)
- RASTER_TEXT  — DeepSeek-OCR-2 (растровый текст / сканы)
- SMART_FIGURE — VLM-классификация → markdown/ocr/PNG
"""

from __future__ import annotations

import re
from typing import Optional

import cv2
import fitz
import numpy as np
from PIL import Image

from config import PDF_DPI, PDF_TO_LAYOUT


def _to_pdf_rect(coords_px: list) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = coords_px
    return (
        x1 * PDF_TO_LAYOUT,
        y1 * PDF_TO_LAYOUT,
        x2 * PDF_TO_LAYOUT,
        y2 * PDF_TO_LAYOUT,
    )


# ---------------------------------------------------------------------------
# Текст (PyMuPDF)
# ---------------------------------------------------------------------------


def extract_text_block(
    pdf_doc: fitz.Document, page_num: int, coords: list
) -> tuple[str, float]:
    """Возвращает (text, avg_font_size) для bbox страницы (coords в пикселях 300 DPI)."""
    page = pdf_doc[page_num]

    # 1. Получаем строгие координаты от YOLO
    x0, y0, x1, y1 = _to_pdf_rect(coords)

    # 2. ДОБАВЛЯЕМ PADDING (В PDF-пунктах)
    # Расширяем рамку по бокам (чтобы спасти длинные слова) и по вертикали (для букв "у", "р")
    pad_x = 5.0
    pad_y = 3.0

    rect = fitz.Rect(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)

    text_dict = page.get_text("dict", clip=rect)

    lines, sizes = [], []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    parts.append(t)
                    sizes.append(span.get("size", 12.0))
            if parts:
                lines.append(" ".join(parts))

    avg = sum(sizes) / len(sizes) if sizes else 12.0
    return "\n".join(lines), avg


def has_vector_text(
    pdf_doc: fitz.Document, page_num: int, coords: list, *, min_chars: int = 4
) -> bool:
    """True, если в bbox страницы есть текстовый слой PDF."""
    page = pdf_doc[page_num]

    x0, y0, x1, y1 = _to_pdf_rect(coords)
    # Здесь тоже добавляем padding, чтобы детектор вектора не сбоил на краях
    rect = fitz.Rect(x0 - 5.0, y0 - 3.0, x1 + 5.0, y1 + 3.0)

    text = page.get_text("text", clip=rect) or ""
    return len(text.strip()) >= min_chars


def render_block_image(
    pdf_doc: fitz.Document, page_num: int, coords: list, dpi: int = PDF_DPI
) -> Image.Image:
    """Рендерит bbox страницы в PIL Image с заданным DPI (для VLM-инференса)."""
    page = pdf_doc[page_num]
    scale = dpi / PDF_DPI
    rect = fitz.Rect(*_to_pdf_rect(coords))
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------------------------------------------------------------------
# Заголовки
# ---------------------------------------------------------------------------


def collect_heading_sizes(pdf_doc: fitz.Document, routing_plan: dict) -> list[float]:
    """Уникальные размеры шрифтов во всех блоках title/section-header, по убыванию."""
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
# Таблицы: PyMuPDF find_tables() + IoU-матчинг по bbox
# ---------------------------------------------------------------------------


class VectorTableStore:
    """
    Сканирует все страницы PDF через PyMuPDF `page.find_tables()` и кеширует
    найденные таблицы по номеру страницы. Для YOLO-bbox возвращает наиболее
    пересекающуюся таблицу (по IoU в PDF-points).

    Преимущества перед Docling:
      - ноль внешних neural-зависимостей, работает на голом transformers==4.46.3;
      - использует те же векторные линии и текстовые спаны, что и рендерит Acrobat;
      - rowspan/colspan восстанавливаются через forward_fill в format_table_markdown.
    """

    def __init__(self, pdf_doc: fitz.Document):
        self._tables_by_page: dict[int, list[tuple[tuple, list[list[str]]]]] = {}
        for page_idx, page in enumerate(pdf_doc):
            try:
                finder = page.find_tables()
            except Exception:
                continue
            tables = getattr(finder, "tables", None) or list(finder or [])
            for tbl in tables:
                grid = _pymupdf_table_to_grid(tbl)
                if not grid:
                    continue
                bbox = getattr(tbl, "bbox", None)
                if bbox is None:
                    continue
                # fitz.Rect → (l, t, r, b) в PDF points (top-left origin)
                rect = fitz.Rect(bbox)
                self._tables_by_page.setdefault(page_idx + 1, []).append(
                    ((rect.x0, rect.y0, rect.x1, rect.y1), grid)
                )

    def find(
        self, page_num_1based: int, coords_px: list, *, min_iou: float = 0.25
    ) -> Optional[str]:
        """Возвращает markdown таблицы с максимальным IoU с данным bbox, или None."""
        candidates = self._tables_by_page.get(page_num_1based, [])
        if not candidates:
            return None

        q = _to_pdf_rect(coords_px)
        best_grid, best_iou = None, 0.0
        for bbox, grid in candidates:
            iou = _iou(bbox, q)
            if iou > best_iou:
                best_iou = iou
                best_grid = grid
        if best_iou < min_iou or best_grid is None:
            return None
        return format_table_markdown(best_grid)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _pymupdf_table_to_grid(tbl) -> Optional[list[list[str]]]:
    """
    PyMuPDF `Table.extract()` возвращает list[list[str|None]] с готовой
    2D-структурой. Объединённые ячейки — None, forward_fill доделает
    format_table_markdown.
    """
    try:
        rows = tbl.extract()
    except Exception:
        return None
    if not rows:
        return None

    clean: list[list[str]] = []
    for row in rows:
        clean_row: list[str] = []
        for cell in row:
            if cell is None:
                clean_row.append("")
            else:
                clean_row.append(str(cell).strip().replace("\n", " "))
        clean.append(clean_row)

    # Отбрасываем полностью пустые таблицы — find_tables иногда ловит декор.
    if not any(any(c for c in row) for row in clean):
        return None
    return clean


# ---------------------------------------------------------------------------
# Markdown таблицы (объединённые шапки + forward-fill)
# ---------------------------------------------------------------------------


def _forward_fill(table: list[list]) -> list[list[str]]:
    """Дублирует значения объединённых ячеек (None/пусто → последнее значение в строке)."""
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
    """True если строка похожа на заголовок: все ячейки не числовые."""
    non_empty = [c for c in row if c.strip()]
    if not non_empty:
        return False
    return all(not _NUMERIC_RE.fullmatch(c) for c in non_empty)


def format_table_markdown(table: list[list], n_header_rows: int = 0) -> str:
    """
    Markdown-таблица с многоуровневыми заголовками (header1_header2).

    n_header_rows: если >0, явно задаёт количество строк-заголовков
                   (передаётся из HTML-парсера по <th> тегам).
    """
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
        # Авто-определение: две верхних строки — оба заголовка,
        # если обе полностью не числовые и минимум 3 строки данных.
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
# Форматирование текстового блока
# ---------------------------------------------------------------------------


def format_text_markdown(text: str, block_type: str, heading_level: int = 0) -> str:
    """Применяет Markdown-разметку к извлечённому тексту."""
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
# Утилиты
# ---------------------------------------------------------------------------


def estimate_text_density(img: Image.Image) -> float:
    """Доля 'чернильных' пикселей — для эвристической классификации."""
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return float(binary.sum()) / (255.0 * binary.size)


# Строки-фрагменты водяных знаков/штампов: только заглавные буквы, 2–6 символов.
# "DRAFT", "КОН", "ЕНЦ", "ИДЕ" и т.п. — артефакты OCR при разрыве страницы.
_STAMP_FRAG_RE = re.compile(r"^[А-ЯЁA-Z]{2,6}$")


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
        lines.append(l)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Анти-галлюцинационные валидаторы
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


def reference_text_for_bbox(
    pdf_doc: fitz.Document, page_num: int, coords: list, *, pad: float = 5.0
) -> str:
    """Текст PDF-слоя по bbox — база для language-drift проверки."""
    try:
        page = pdf_doc[page_num]
        x0, y0, x1, y1 = _to_pdf_rect(coords)
        rect = fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        return (page.get_text("text", clip=rect) or "").strip()
    except Exception:
        return ""


def table_stats(md: str) -> dict:
    """
    Структурный разбор pipe-markdown таблицы.
    Возвращает {'n_cols', 'n_rows', 'empty_ratio', 'max_cell', 'row_repeat_ratio'}.
    Для не-таблиц возвращает n_cols=0.
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

    # повторы строк (данные → один и тот же ряд дублируется — галлюцинация)
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
