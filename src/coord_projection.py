"""
Проекция координат между системами YOLO (пиксели страницы) и Docling (PDF points).

YOLO работает с отрендеренным растром страницы на LAYOUT_DPI (400).
Docling возвращает bbox в PDF points (72 DPI), оригинальная система координат PDF.

  S = LAYOUT_DPI / PDF_DPI  →  points → pixels
  1/S                        →  pixels → points

PDF-координаты в fitz имеют origin в верхнем левом углу (top-down Y),
что совпадает с пиксельной ориентацией YOLO → инверсия Y не нужна.

Docling может отдавать bbox в двух системах (BOTTOMLEFT / TOPLEFT);
см. `points_to_pixels` ниже — поддерживаем обе.

Матчинг блоков: IoM (Intersection over Min-Area) вместо IoU —
маленькое слово внутри большого блока при IoU даёт почти 0.
"""

from __future__ import annotations

from typing import Optional

from config import LAYOUT_DPI, PDF_DPI


Bbox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def points_to_pixels(
    bbox_pts: Bbox,
    page_height_pts: Optional[float] = None,
    *,
    origin: str = "top",
) -> Bbox:
    """
    Переводит bbox из PDF points (Docling) в пиксели YOLO-растра.

    origin="top"    — Y растёт вниз (fitz, Docling TOPLEFT)
    origin="bottom" — Y растёт вверх (Docling BOTTOMLEFT); нужна инверсия Y
    """
    s = LAYOUT_DPI / PDF_DPI
    x1, y1, x2, y2 = bbox_pts
    if origin == "bottom":
        if page_height_pts is None:
            raise ValueError("origin='bottom' требует page_height_pts")
        y1, y2 = page_height_pts - y2, page_height_pts - y1
    return (x1 * s, y1 * s, x2 * s, y2 * s)


def pixels_to_points(bbox_px: Bbox) -> Bbox:
    """Обратная проекция: YOLO-пиксели → PDF points (top-down)."""
    s = PDF_DPI / LAYOUT_DPI
    x1, y1, x2, y2 = bbox_px
    return (x1 * s, y1 * s, x2 * s, y2 * s)


# ---------------------------------------------------------------------------
# Метрики пересечения
# ---------------------------------------------------------------------------


def _area(bbox: Bbox) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _intersection(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def iou(a: Bbox, b: Bbox) -> float:
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def iom(a: Bbox, b: Bbox) -> float:
    """
    Intersection over Min-Area.
    Если маленький bbox полностью внутри большого — IoM=1.0 (а IoU ≈ 0).
    Используется для матчинга мелких Docling-элементов (слов, строк)
    с крупными YOLO-блоками (абзацами, таблицами).
    """
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    min_area = min(_area(a), _area(b))
    return inter / min_area if min_area > 0 else 0.0


def horizontal_overlap(a: Bbox, b: Bbox) -> float:
    """Доля горизонтального пересечения относительно меньшей ширины блока."""
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    ix1, ix2 = max(ax1, bx1), min(ax2, bx2)
    if ix2 <= ix1:
        return 0.0
    min_w = min(ax2 - ax1, bx2 - bx1)
    return (ix2 - ix1) / min_w if min_w > 0 else 0.0
