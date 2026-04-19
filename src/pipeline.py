"""
Главный пайплайн: PDF → Markdown + images/ → submission.zip

Архитектура:
    Layout Engine  : DocLayout-YOLO (src/layout_router.py)
    Vector Engine  : PyMuPDF (текст)
    Raster Engine  : DeepSeek-OCR-2 (src/vlm_engine.py) — все таблицы через VLM

Треки, в которые перераспределяются блоки YOLO после is_vector-анализа:
    VECTOR_TEXT  — текст/заголовок/список, поверх которого есть текстовый слой PDF
    RASTER_TEXT  — текст/заголовок/список без текстового слоя (скан)
    RASTER_TABLE — любая таблица (DeepSeek → markdown)
    SMART_FIGURE — figure/picture/image: VLM-классификация → markdown/ocr/PNG

Использование:
    python src/pipeline.py --all
    python src/pipeline.py --pdf data/raw/document_001.pdf
    python src/pipeline.py --all --no-vlm         # только векторные треки
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from content_extractor import (
    collect_heading_sizes,
    detect_heading_level,
    extract_text_block,
    filter_noise_lines,
    format_table_markdown,
    format_text_markdown,
    has_vector_text,
    render_block_image,
)
from device import setup_environment
from layout_router import LayoutRouter

setup_environment()

OUTPUT_DIR = "data/output"
IMAGE_MAX_SIDE = 150
VLM_RENDER_DPI = 300

FIGURE_LABELS = {"picture", "figure", "image", "missed_raster"}
TEXT_LABELS = {"text", "plain text", "title", "section-header", "list-item"}


# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        use_vlm: bool = True,
    ):
        self.output_dir = output_dir
        self.use_vlm = use_vlm

        self.router = LayoutRouter()

        self.vlm = None
        if use_vlm:
            try:
                from vlm_engine import VLMEngine

                self.vlm = VLMEngine.get()
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] VLM недоступен: {exc}")
                self.vlm = None

        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str) -> str:
        plan = self.router.build_routing_plan(pdf_path, self.output_dir)
        doc_name = Path(pdf_path).stem
        pdf_doc = fitz.open(pdf_path)

        self._enrich_plan_with_tracks(plan, pdf_doc)
        heading_sizes = collect_heading_sizes(pdf_doc, plan)

        # Собираем плоский список (track, md, is_last_on_page) для всех блоков.
        # is_last определяется по позиции среди непустых выводов страницы,
        # а не по позиции блока в исходном плане — блоки могут вернуть пустой md.
        flat: list[tuple[str, str, bool]] = []
        for page_data in plan["pages"]:
            page_num = page_data["page_num"] - 1
            page_items: list[tuple[str, str]] = []
            for block in page_data["blocks"]:
                md = self._process_block(block, page_num, pdf_doc, heading_sizes)
                if md and md.strip():
                    page_items.append((block["track"], md.strip()))
            n = len(page_items)
            for i, (track, md) in enumerate(page_items):
                flat.append((track, md, i == n - 1))
            if self.vlm is not None:
                self.vlm.release_page_cache()
            else:
                gc.collect()

        pdf_doc.close()

        parts = _merge_cross_page_tables(flat)
        markdown = "\n\n".join(parts)
        md_path = os.path.join(self.output_dir, f"{doc_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        self._cleanup_temp(plan)
        print(f"  ✓ {doc_name}.md")
        return md_path

    def process_all(self, raw_dir: str = "data/raw") -> str:
        pdfs = sorted(Path(raw_dir).glob("document_*.pdf"))
        print(f"Найдено {len(pdfs)} PDF файлов")

        for i, pdf_path in enumerate(pdfs, 1):
            print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
            try:
                self.process_pdf(str(pdf_path))
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ ОШИБКА: {exc}")

        zip_path = self._create_zip()
        print(f"\nГотово: {zip_path}")
        return zip_path

    # ------------------------------------------------------------------
    # Роутинг: добавление is_vector и правильных треков
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_plan_with_tracks(plan: dict, pdf_doc: fitz.Document) -> None:
        """
        Layout-роутер не знает про векторный слой PDF. Проходим по плану и:
          - помечаем каждый блок `is_vector`
          - перевешиваем `track` согласно ТЗ (5 треков)
        """
        for page_data in plan["pages"]:
            page_num = page_data["page_num"] - 1
            for block in page_data["blocks"]:
                btype = block["type"]
                coords = block["coords"]

                if btype in FIGURE_LABELS:
                    block["track"] = "SMART_FIGURE"
                    block["is_vector"] = False
                    continue

                is_vec = has_vector_text(pdf_doc, page_num, coords, min_chars=4)
                block["is_vector"] = is_vec

                if btype == "table":
                    block["track"] = "RASTER_TABLE"
                elif btype in TEXT_LABELS:
                    block["track"] = "VECTOR_TEXT" if is_vec else "RASTER_TEXT"
                else:
                    # Незнакомый label — обрабатываем как текст.
                    block["track"] = "VECTOR_TEXT" if is_vec else "RASTER_TEXT"

    # ------------------------------------------------------------------
    # Страница / блок
    # ------------------------------------------------------------------

    def _process_page(
        self,
        page_data: dict,
        pdf_doc: fitz.Document,
        heading_sizes: list[float],
    ) -> str:
        page_num = page_data["page_num"] - 1
        parts: list[str] = []
        for block in page_data["blocks"]:
            md = self._process_block(block, page_num, pdf_doc, heading_sizes)
            if md and md.strip():
                parts.append(md.strip())
        return "\n\n".join(parts)

    def _process_block(
        self,
        block: dict,
        page_num: int,
        pdf_doc: fitz.Document,
        heading_sizes: list[float],
    ) -> str:
        track = block["track"]
        btype = block["type"]
        coords = block["coords"]

        if track == "VECTOR_TEXT":
            return self._vector_text(btype, coords, page_num, pdf_doc, heading_sizes)

        if track == "RASTER_TABLE":
            return self._raster_table(page_num, coords, pdf_doc)

        if track == "RASTER_TEXT":
            return self._raster_text(btype, coords, page_num, pdf_doc)

        if track == "SMART_FIGURE":
            return self._smart_figure(block, page_num, pdf_doc)

        return ""

    # ------------------------------------------------------------------
    # VECTOR_TEXT
    # ------------------------------------------------------------------

    def _vector_text(
        self,
        btype: str,
        coords: list,
        page_num: int,
        pdf_doc: fitz.Document,
        heading_sizes: list[float],
    ) -> str:
        text, font_size = extract_text_block(pdf_doc, page_num, coords)
        text = filter_noise_lines(text, min_chars=3)
        if not text:
            return ""

        heading_level = 0
        if btype in ("title", "section-header"):
            heading_level = detect_heading_level(font_size, heading_sizes)
        return format_text_markdown(text, btype, heading_level)

    # ------------------------------------------------------------------
    # RASTER_TEXT
    # ------------------------------------------------------------------

    def _raster_text(
        self,
        btype: str,
        coords: list,
        page_num: int,
        pdf_doc: fitz.Document,
    ) -> str:
        if self.vlm is None:
            return ""
        img = render_block_image(pdf_doc, page_num, coords, dpi=VLM_RENDER_DPI)
        text = self.vlm.extract_markdown(img).strip()
        if not text:
            # Fallback для коротких блоков (4-6 слов), где extract_markdown даёт пустоту
            text = self.vlm.free_ocr(img).strip()
        text = filter_noise_lines(text, min_chars=3)
        if not text:
            return ""

        heading_level = 0
        if btype == "title":
            heading_level = 1
        elif btype == "section-header":
            heading_level = 2
        return format_text_markdown(text, btype, heading_level)

    # ------------------------------------------------------------------
    # RASTER_TABLE
    # ------------------------------------------------------------------

    def _raster_table(self, page_num: int, coords: list, pdf_doc: fitz.Document) -> str:
        if self.vlm is None:
            return ""
        img = render_block_image(pdf_doc, page_num, coords, dpi=VLM_RENDER_DPI)
        md = self.vlm.extract_markdown(img).strip()
        return _postprocess_table_markdown(md)

    # ------------------------------------------------------------------
    # SMART_FIGURE
    # ------------------------------------------------------------------

    def _smart_figure(self, block: dict, page_num: int, pdf_doc: fitz.Document) -> str:
        coords = block["coords"]
        md_image_name = block.get("md_image_name")
        content_path = block.get("content_path")

        pil_img: Optional[Image.Image] = None
        if content_path and os.path.exists(content_path):
            try:
                pil_img = Image.open(content_path).convert("RGB")
            except Exception:
                pil_img = None
        if pil_img is None:
            pil_img = render_block_image(pdf_doc, page_num, coords, dpi=VLM_RENDER_DPI)

        if self.vlm is None:
            return self._save_figure(pil_img, md_image_name, alt="image")

        raw = ""
        try:
            raw = self.vlm.extract_markdown(pil_img).strip()
            if not raw:
                # Fallback для маленьких блоков с 4-6 словами
                raw = self.vlm.free_ocr(pil_img).strip()
        except Exception:
            pass

        # Определяем таблицу по содержимому вывода — без отдельного classify.
        is_table = _looks_like_table(raw)

        if is_table:
            # Таблицы НЕ сохраняем как PNG: они тяжёлые и метрика их не требует.
            md = _postprocess_table_markdown(raw)
            if md:
                return md
            # Постпроцесс не справился — сохраняем как fallback
            return self._save_figure(pil_img, md_image_name, alt="таблица")

        # Для обычных изображений: ссылка + OCR-текст построчно.
        text = filter_noise_lines(raw, min_chars=3)
        # Убираем из текста строки-мусор вида ![...](...)  которые OCR подхватил из PDF.
        text = _strip_md_image_lines(text)

        # alt: первая чистая строка OCR, иначе русская подпись
        alt = _first_clean_line(text)
        if not alt:
            try:
                alt = self.vlm.short_caption(pil_img) or "image"
            except Exception:
                alt = "image"

        image_md = self._save_figure(pil_img, md_image_name, alt=alt)
        return f"{image_md}\n{text}" if text else image_md

    def _save_figure(
        self,
        pil_img: Image.Image,
        md_image_name: Optional[str],
        alt: str,
    ) -> str:
        if not md_image_name:
            return ""
        dest = os.path.join(self.images_dir, md_image_name)
        try:
            img = pil_img.convert("RGB")
            img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.NEAREST)
            img.save(dest, "PNG", optimize=True, compress_level=9)
        except Exception:
            pass

        alt = (alt or "image").strip() or "image"
        for ch in "![]()":
            alt = alt.replace(ch, "")
        alt = alt.strip() or "image"
        return f"![{alt}](images/{md_image_name})"

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def _cleanup_temp(self, plan: dict) -> None:
        temp_doc_dir = os.path.join(
            self.output_dir, "temp", f"document_{plan['doc_id']}"
        )
        shutil.rmtree(temp_doc_dir, ignore_errors=True)

    def _create_zip(self) -> str:
        zip_path = os.path.join(self.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for md_file in sorted(Path(self.output_dir).glob("document_*.md")):
                zf.write(md_file, md_file.name)
            img_dir = Path(self.images_dir)
            if img_dir.exists():
                for img_file in sorted(img_dir.glob("*.png")):
                    zf.write(img_file, f"images/{img_file.name}")
        return zip_path


# ---------------------------------------------------------------------------
# Склейка таблиц, разрезанных переносом страницы
# ---------------------------------------------------------------------------


def _merge_two_tables(t1: str, t2: str) -> str:
    """
    Склеивает две части таблицы:
    - pipe + pipe → добавляет строки t2, отбрасывая повторный заголовок
    - plain_text + pipe → plain text становится заголовком (VLM вернул
      заголовочную строку без pipe-форматирования, данные — нормальной таблицей)
    """
    lines1 = [l for l in t1.splitlines() if l.strip()]
    lines2 = [l for l in t2.splitlines() if l.strip()]
    if not lines1 or not lines2:
        return (t1 + "\n\n" + t2).strip()

    t1_has_pipe = any(l.strip().startswith("|") for l in lines1)
    t2_has_pipe = any(l.strip().startswith("|") for l in lines2)

    if not t1_has_pipe and t2_has_pipe:
        # t1 = plain-text слова (по одному на строку) — заголовки столбцов
        # t2 = pipe-таблица с данными
        pipe_lines2 = [l for l in lines2 if l.strip().startswith("|")]
        n_cols = len([c for c in pipe_lines2[0].strip().strip("|").split("|")])
        if abs(len(lines1) - n_cols) <= 1:
            headers = lines1[:n_cols] + [""] * max(0, n_cols - len(lines1))
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| " + " | ".join(["---"] * n_cols) + " |"
            # Все не-разделители из t2 — данные (включая "ложную" первую строку t2)
            data_rows = [l for l in lines2
                         if l.strip().startswith("|")
                         and not re.match(r"^\|\s*[-:]+", l.strip())]
            return "\n".join([header_row, sep_row] + data_rows)

    # Оба pipe: добавляем данные t2 к t1, пропуская повторный заголовок
    header1 = lines1[0].strip()
    data2 = []
    for i, ln in enumerate(lines2):
        if i == 0 and ln.strip() == header1:
            continue
        if re.match(r"^\|\s*[-:]+", ln.strip()):
            continue
        data2.append(ln)
    return "\n".join(lines1 + data2)


def _text_looks_like_table_headers(md: str, n_cols: int) -> bool:
    """True если md — список коротких строк без pipe, похожий на заголовки столбцов."""
    if "|" in md:
        return False
    lines = [l.strip() for l in md.splitlines() if l.strip()]
    if not lines or abs(len(lines) - n_cols) > 1:
        return False
    # Каждая строка короткая (не абзац)
    return all(len(l) <= 60 for l in lines)


def _merge_cross_page_tables(flat: list[tuple[str, str, bool]]) -> list[str]:
    """
    Склеивает части таблицы, разрезанной переносом страницы.
    Два подряд RASTER_TABLE → мержим.
    TEXT-блок + RASTER_TABLE с совпадающим числом столбцов → text становится заголовком.
    flat — список (track, md, is_last_on_page).
    """
    result: list[str] = []
    i = 0
    while i < len(flat):
        track, md, _ = flat[i]

        # Текстовый блок перед таблицей — может быть заголовком
        if (
            track in ("VECTOR_TEXT", "RASTER_TEXT")
            and i + 1 < len(flat)
            and flat[i + 1][0] == "RASTER_TABLE"
        ):
            _, next_md, next_last = flat[i + 1]
            pipe_lines = [l for l in next_md.splitlines() if l.strip().startswith("|")]
            if pipe_lines:
                n_cols = len([c for c in pipe_lines[0].strip().strip("|").split("|")])
                if _text_looks_like_table_headers(md, n_cols):
                    merged = _merge_two_tables(md, next_md)
                    flat[i + 1] = ("RASTER_TABLE", merged, next_last)
                    i += 1
                    continue

        # Два подряд RASTER_TABLE → мержим (продолжение таблицы через страницу)
        while (
            track == "RASTER_TABLE"
            and i + 1 < len(flat)
            and flat[i + 1][0] == "RASTER_TABLE"
        ):
            i += 1
            _, next_md, _ = flat[i]
            md = _merge_two_tables(md, next_md)

        result.append(md)
        i += 1
    return result


# ---------------------------------------------------------------------------
# VLM-таблицы: пост-обработка (единый формат с векторными)
# ---------------------------------------------------------------------------


_TD_RE = re.compile(r"<(td|th)([^>]*)>(.*?)</(td|th)>", re.S | re.I)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _parse_html_table(html: str) -> tuple[list[list[str]], int]:
    """
    Парсит HTML <table> в (rows, n_header_rows).
    n_header_rows — количество строк, где все ячейки были <th>.
    """
    rows: list[list[str]] = []
    n_header_rows = 0
    for tr in _TR_RE.finditer(html):
        cells = []
        all_th = True
        for td in _TD_RE.finditer(tr.group(1)):
            tag = td.group(1).lower()
            text = _TAG_STRIP_RE.sub("", td.group(3)).strip()
            cells.append(text)
            if tag != "th":
                all_th = False
        if cells:
            rows.append(cells)
            if all_th and len(rows) == n_header_rows + 1:
                n_header_rows += 1
    return rows, n_header_rows


def _postprocess_table_markdown(md: str) -> str:
    """
    Нормализует вывод DeepSeek-OCR в pipe-markdown.
    DeepSeek может вернуть HTML или pipe-markdown.
    Прогоняем через format_table_markdown (forward_fill + multilevel headers).
    """
    if not md:
        return ""

    # Путь 1: HTML-таблица
    if "<table" in md.lower():
        rows, n_hdr = _parse_html_table(md)
        if len(rows) >= 2:
            return format_table_markdown(rows, n_header_rows=n_hdr) or md.strip()

    # Путь 2: pipe-markdown
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) >= 2:
        rows = []
        for ln in lines:
            if re.match(r"^\|\s*:?-{2,}", ln):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
        if len(rows) >= 2:
            return format_table_markdown(rows) or md.strip()

    return md.strip()


# ---------------------------------------------------------------------------
# Вспомогательные функции для _smart_figure
# ---------------------------------------------------------------------------

_MD_IMAGE_LINE_RE = re.compile(r"^!?\[.*?\]\(.*?\)\s*$")


def _looks_like_table(raw: str) -> bool:
    """True если вывод VLM содержит таблицу (HTML или pipe-markdown)."""
    if not raw:
        return False
    if "<table" in raw.lower():
        return True
    lines = raw.splitlines()
    pipe_lines = [l for l in lines if l.strip().startswith("|")]
    sep_lines = [l for l in pipe_lines if re.match(r"^\|\s*:?-{2,}", l.strip())]
    return len(pipe_lines) >= 3 and len(sep_lines) >= 1


def _strip_md_image_lines(text: str) -> str:
    """Удаляет строки вида ![...](...)  — OCR иногда «читает» встроенные картинки из PDF."""
    if not text:
        return ""
    lines = [l for l in text.splitlines() if not _MD_IMAGE_LINE_RE.match(l.strip())]
    return "\n".join(lines).strip()


def _first_clean_line(text: str) -> str:
    """Первая строка текста без markdown-символов, пригодная для alt (≤80 символов)."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _MD_IMAGE_LINE_RE.match(line):
            continue
        if line.startswith(("#", "|", "!", "[")):
            continue
        clean = line[:80]
        for ch in "![]()":
            clean = clean.replace(ch, "")
        clean = clean.strip()
        if clean:
            return clean
    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF → Markdown (YOLO + PyMuPDF + DeepSeek-OCR)"
    )
    parser.add_argument("--pdf", type=str, help="Путь к одному PDF")
    parser.add_argument(
        "--all", action="store_true", help="Обработать все PDF из raw-dir"
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-vlm", action="store_true", help="Отключить DeepSeek-OCR")
    args = parser.parse_args()

    pipeline = Pipeline(
        output_dir=args.output_dir,
        use_vlm=not args.no_vlm,
    )

    if args.all:
        pipeline.process_all(args.raw_dir)
    elif args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Файл не найден: {args.pdf}")
            sys.exit(1)
        pipeline.process_pdf(args.pdf)
        pipeline._create_zip()
    else:
        print("Используйте --pdf <path> или --all")
        parser.print_help()
