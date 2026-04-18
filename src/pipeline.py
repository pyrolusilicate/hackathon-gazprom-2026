"""
Главный пайплайн: PDF → Markdown + images/ → submission.zip

Использование:
    python src/pipeline.py --all                        # все файлы из data/raw/
    python src/pipeline.py --pdf data/raw/document_001.pdf
    python src/pipeline.py --all --no-ocr              # без OCR (быстрее)
    python src/pipeline.py --all --no-vlm              # без VLM (быстрее, хуже OCR на сканах)
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import fitz
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from content_extractor import (
    collect_heading_sizes,
    detect_heading_level,
    estimate_text_density,
    extract_table,
    extract_text_block,
    filter_noise_lines,
    format_table_markdown,
    format_text_markdown,
    looks_like_table,
    render_block_for_ocr,
)
from device import setup_environment
from layout_router import LayoutRouter

setup_environment()

OUTPUT_DIR = "data/output"
IMAGE_SAVE_DPI = 200        # DPI для PNG из routing_plan'а (было 150)
IMAGE_MAX_SIDE = 2400       # Максимум по большей стороне PNG

# Плотность «чернила/пиксель», выше которой считаем блок текстово-содержательным
# (скан, растровая таблица), а не фотографией/графиком.
TEXT_DENSITY_THRESHOLD = 0.06


class Pipeline:
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        use_ocr: bool = True,
        use_vlm: bool = True,
    ):
        self.output_dir = output_dir
        self.router = LayoutRouter()
        self.ocr = None
        self.vlm = None

        if use_ocr:
            from ocr_engine import OCREngine
            self.ocr = OCREngine()

        if use_vlm:
            try:
                from vlm_engine import VLMEngine
                self.vlm = VLMEngine.get()
            except Exception as e:
                print(f"  [WARN] VLM недоступен: {e}. Фолбэк на EasyOCR.")
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

        heading_sizes = collect_heading_sizes(pdf_doc, plan)

        page_chunks: list[str] = []
        for page_data in plan["pages"]:
            chunk = self._process_page(page_data, plan, pdf_doc, heading_sizes)
            if chunk.strip():
                page_chunks.append(chunk.strip())

        pdf_doc.close()

        markdown = "\n\n".join(page_chunks)
        md_path = os.path.join(self.output_dir, f"{doc_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"  ✓ {doc_name}.md")
        return md_path

    def process_all(self, raw_dir: str = "data/raw") -> str:
        pdfs = sorted(Path(raw_dir).glob("document_*.pdf"))
        print(f"Найдено {len(pdfs)} PDF файлов")

        for i, pdf_path in enumerate(pdfs, 1):
            print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
            try:
                self.process_pdf(str(pdf_path))
            except Exception as e:
                print(f"  ✗ ОШИБКА: {e}")

        zip_path = self._create_zip()
        print(f"\nGOTOV: {zip_path}")
        return zip_path

    # ------------------------------------------------------------------
    # Страница
    # ------------------------------------------------------------------

    def _process_page(self, page_data, plan, pdf_doc, heading_sizes) -> str:
        page_num = page_data["page_num"] - 1
        pdf_path = plan["pdf_path"]
        fitz_page = pdf_doc[page_num]
        native_text = fitz_page.get_text("text").strip()
        is_rasterized = len(native_text) < 20

        parts: list[str] = []
        for block in page_data["blocks"]:
            md = self._process_block(
                block, page_num, pdf_path, pdf_doc, heading_sizes, is_rasterized
            )
            if md and md.strip():
                parts.append(md.strip())
        return "\n\n".join(parts)

    def _process_block(
        self, block, page_num, pdf_path, pdf_doc, heading_sizes, is_rasterized
    ) -> str:
        btype = block["type"]
        coords = block["coords"]
        track = block["track"]
        content_path = block.get("content_path")
        md_image_name = block.get("md_image_name")

        if track == "DOCLING_TEXT":
            return self._text_block(
                btype, coords, page_num, pdf_doc, heading_sizes,
                is_rasterized, content_path,
            )

        if track == "DOCLING_TABLE":
            return self._table_block(
                coords, page_num, pdf_path, pdf_doc,
                content_path, is_rasterized,
            )

        if track == "PADDLE_OCR":
            return self._image_block(
                btype, coords, page_num, pdf_doc,
                content_path, md_image_name, is_rasterized,
            )

        return ""

    # ------------------------------------------------------------------
    # Текстовый блок
    # ------------------------------------------------------------------

    def _text_block(
        self, btype, coords, page_num, pdf_doc, heading_sizes,
        is_rasterized, content_path,
    ) -> str:
        text, font_size = "", 12.0

        if not is_rasterized:
            text, font_size = extract_text_block(pdf_doc, page_num, coords)

        # OCR-фоллбэк для растрового текста
        if not text.strip():
            pil_img = render_block_for_ocr(pdf_doc, page_num, coords)
            # VLM приоритетнее — он лучше склеивает строки.
            if self.vlm is not None:
                text = self.vlm.extract_text(pil_img)
            elif self.ocr is not None:
                text = self.ocr.read_pil(pil_img)

        text = filter_noise_lines(text.strip(), min_chars=3)
        if not text:
            return ""

        heading_level = 0
        if btype in ("title", "section-header"):
            heading_level = detect_heading_level(font_size, heading_sizes)

        return format_text_markdown(text, btype, heading_level)

    # ------------------------------------------------------------------
    # Таблица
    # ------------------------------------------------------------------

    def _table_block(
        self, coords, page_num, pdf_path, pdf_doc,
        content_path, is_rasterized,
    ) -> str:
        if not is_rasterized:
            table_data = extract_table(pdf_path, page_num, coords)
            if table_data and self._table_ok(table_data):
                md = format_table_markdown(table_data)
                if md.strip():
                    return md

        # Фолбэк: рендерим блок в картинку с высоким DPI и отдаём в VLM.
        if self.vlm is not None:
            img = render_block_for_ocr(pdf_doc, page_num, coords, dpi=300)
            md = self.vlm.extract_table(img).strip()
            if md.startswith("|") or "|" in md:
                return md
            if md:
                return md

        # Последний шанс: OCR-картинка как plain-text.
        if self.ocr and content_path and os.path.exists(content_path):
            return self.ocr.read_image(content_path)

        return ""

    @staticmethod
    def _table_ok(data) -> bool:
        """Пустые и одноколоночные результаты pdfplumber лучше переделать VLM'ом."""
        if not data or len(data) < 2:
            return False
        width = max(len(r) for r in data)
        if width < 2:
            return False
        total = sum(len(r) for r in data)
        empties = sum(
            1 for row in data for cell in row
            if cell is None or not str(cell).strip()
        )
        return empties / total < 0.55

    # ------------------------------------------------------------------
    # Картинка / растровый блок
    # ------------------------------------------------------------------

    def _image_block(
        self, btype, coords, page_num, pdf_doc,
        content_path, md_image_name, is_rasterized,
    ) -> str:
        # 1) Явно растровые типы из задания → только текст, без PNG.
        ocr_only_types = {
            "scan_image", "rasterized_pdf", "handwritten_ru", "image_with_table",
        }

        if not content_path or not os.path.exists(content_path):
            return ""

        pil_img = Image.open(content_path)
        density = estimate_text_density(pil_img)

        # Если страница целиком растровая — весь блок отдаём как текст.
        if is_rasterized or btype in ocr_only_types:
            return self._extract_text_only(pil_img, btype)

        # 2) Фото/схема vs текст/таблица. VLM-классификация.
        kind = "picture"
        if self.vlm is not None:
            try:
                kind = self.vlm.classify(pil_img)
            except Exception:
                kind = "picture"
        else:
            # Без VLM: эвристика на плотности
            kind = "text" if density > TEXT_DENSITY_THRESHOLD else "picture"

        if kind == "table":
            # Растровая таблица — извлекаем как markdown.
            if self.vlm is not None:
                md = self.vlm.extract_table(pil_img).strip()
                if md:
                    return md
            return self._extract_text_only(pil_img, "scan_image")

        if kind in ("text", "handwritten"):
            return self._extract_text_only(pil_img, btype, handwritten=(kind == "handwritten"))

        # 3) Обычная картинка — сохраняем PNG и даём alt.
        return self._save_image_with_alt(pil_img, md_image_name, content_path)

    def _extract_text_only(self, pil_img: Image.Image, btype: str, handwritten: bool = False) -> str:
        if self.vlm is not None:
            try:
                if handwritten or btype == "handwritten_ru":
                    return self.vlm.extract_handwritten(pil_img).strip()
                return self.vlm.extract_text(pil_img).strip()
            except Exception:
                pass
        if self.ocr is not None:
            return self.ocr.read_pil(pil_img).strip()
        return ""

    def _save_image_with_alt(self, pil_img: Image.Image, md_image_name, content_path) -> str:
        if not md_image_name:
            return ""
        dest = os.path.join(self.images_dir, md_image_name)
        try:
            img = pil_img
            if max(img.width, img.height) > IMAGE_MAX_SIDE:
                img = img.copy()
                img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
            img.save(dest, "PNG", optimize=True)
        except Exception:
            shutil.copy2(content_path, dest)

        alt = "image"
        if self.vlm is not None:
            try:
                cap = self.vlm.short_caption(pil_img).strip()
                if cap:
                    alt = cap
            except Exception:
                pass
        elif self.ocr is not None:
            ocr_text = self.ocr.read_image(dest).strip()
            if len(ocr_text.replace(" ", "")) > 5:
                alt = ocr_text.split("\n")[0][:120].replace("]", "")

        return f"![{alt}](images/{md_image_name})"

    # ------------------------------------------------------------------
    # ZIP
    # ------------------------------------------------------------------

    def _create_zip(self) -> str:
        zip_path = os.path.join(self.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for md_file in Path(self.output_dir).glob("document_*.md"):
                zf.write(md_file, md_file.name)
            img_dir = Path(self.images_dir)
            if img_dir.exists():
                for img_file in sorted(img_dir.glob("*.png")):
                    zf.write(img_file, f"images/{img_file.name}")
        print(f"ZIP: {zip_path} ({Path(zip_path).stat().st_size // 1024} KB)")
        return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → Markdown Pipeline")
    parser.add_argument("--pdf", type=str, help="Путь к одному PDF")
    parser.add_argument("--all", action="store_true", help="Обработать все PDF из raw-dir")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-ocr", action="store_true", help="Отключить OCR")
    parser.add_argument("--no-vlm", action="store_true", help="Отключить VLM (Qwen2.5-VL)")
    parser.add_argument("--visualize", action="store_true", help="Сохранить визуализацию YOLO")
    args = parser.parse_args()

    pipeline = Pipeline(
        output_dir=args.output_dir,
        use_ocr=not args.no_ocr,
        use_vlm=not args.no_vlm,
    )

    if args.all:
        pipeline.process_all(args.raw_dir)
    elif args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Файл не найден: {args.pdf}")
            sys.exit(1)
        pipeline.process_pdf(args.pdf)
    else:
        print("Используйте --pdf <path> или --all")
        parser.print_help()
