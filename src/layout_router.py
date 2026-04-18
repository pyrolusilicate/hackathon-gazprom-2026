import argparse
import json
import os

import cv2
import fitz
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from device import get_torch_device, setup_environment

setup_environment()


class LayoutRouter:
    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        self.device = get_torch_device()
        self.model = self._load_model()
        self.ignore_classes = {
            "page-header",
            "page-footer",
            "footnote",
            "watermark",
            "abandon",
        }
        self.vlm_candidates = {"picture", "figure", "image"}
        # Приоритет при NMS: при перекрытии оставляем класс с бо́льшим приоритетом
        self._class_priority = {
            "title": 10,
            "section-header": 9,
            "table": 8,
            "figure": 7,
            "picture": 7,
            "table_caption": 6,
            "figure_caption": 6,
            "caption": 6,
            "list-item": 5,
            "text": 4,
            "plain text": 4,
        }

    def _load_model(self) -> YOLOv10:
        os.makedirs(self.weights_dir, exist_ok=True)
        weights_path = os.path.join(
            self.weights_dir, "doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        if not os.path.exists(weights_path):
            hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                local_dir=self.weights_dir,
            )
        return YOLOv10(weights_path)

    def get_document_id(self, filename: str) -> str:
        base_name = os.path.basename(filename)
        try:
            return str(int(base_name.replace("document_", "").replace(".pdf", "")))
        except ValueError:
            return base_name.replace(".pdf", "")

    def _sort_reading_order(self, boxes: list, page_width: float, page_height: float) -> list:
        """
        Порядок чтения с устойчивой детекцией колонок.

        Алгоритм:
          1. Парсим боксы и отбрасываем шапки/подвалы.
          2. Склеиваем пары (figure/table) + их caption в единый логический блок.
          3. Определяем, является ли страница многоколоночной, кластеризуя
             x-центры «узких» блоков (ширина < 55% страницы).
          4. «Широкие» блоки (заголовки, full-width images) становятся
             разделителями секций — между ними поток читается колоночно.
          5. Внутри секции блоки назначаются в ближайшую колонку по x-центру,
             каждая колонка читается сверху вниз.
        """
        if not boxes:
            return []

        parsed = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            parsed.append(
                {
                    "box": box,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "cx": (x1 + x2) / 2,
                    "cls": self.model.names[int(box.cls[0])].lower(),
                }
            )

        # 1. Отфильтровать шапки/подвалы (ближе 5% к краю + типовая метка)
        margin_top = page_height * 0.05
        margin_bottom = page_height * 0.95
        filtered = []
        for b in parsed:
            if b["cls"] in ("page-header", "page-footer"):
                continue
            if b["cls"] in ("text", "plain text", "title"):
                if b["y2"] < margin_top or b["y1"] > margin_bottom:
                    continue
            filtered.append(b)

        if not filtered:
            return []

        filtered.sort(key=lambda x: x["y1"])

        # 2. Склейка media + caption в логические блоки.
        caption_classes = {"figure_caption", "table_caption", "caption"}
        media_classes = {"figure", "picture", "image", "table"}
        logical = []
        used = set()
        for i, b in enumerate(filtered):
            if i in used:
                continue
            group = [b]
            used.add(i)
            base = b
            if base["cls"] in media_classes:
                for j in range(i + 1, len(filtered)):
                    if j in used:
                        continue
                    nxt = filtered[j]
                    y_gap = nxt["y1"] - base["y2"]
                    if y_gap > page_height * 0.05:
                        break
                    overlap_x = max(0, min(base["x2"], nxt["x2"]) - max(base["x1"], nxt["x1"]))
                    min_w = min(base["w"], nxt["w"]) or 1
                    if nxt["cls"] in caption_classes and overlap_x / min_w > 0.4:
                        group.append(nxt)
                        used.add(j)
                        base = nxt
                        break  # подпись одна — дальше не смотрим
            logical.append(
                {
                    "boxes": group,
                    "x1": min(b["x1"] for b in group),
                    "y1": min(b["y1"] for b in group),
                    "x2": max(b["x2"] for b in group),
                    "y2": max(b["y2"] for b in group),
                    "cx": sum((b["x1"] + b["x2"]) / 2 for b in group) / len(group),
                }
            )
            logical[-1]["w"] = logical[-1]["x2"] - logical[-1]["x1"]
            logical[-1]["h"] = logical[-1]["y2"] - logical[-1]["y1"]

        # 3. Кластеризация центров «узких» блоков → количество колонок.
        narrow_centers = sorted(lb["cx"] for lb in logical if lb["w"] < page_width * 0.55)
        col_centers = self._cluster_centers(narrow_centers, gap=page_width * 0.08)

        # Фильтруем слабые кластеры: минимум 2 блока, центры разнесены ≥ 15%
        col_centers = [c for c in col_centers if c["count"] >= 2]
        col_centers.sort(key=lambda c: c["center"])
        if len(col_centers) >= 2:
            dx = col_centers[-1]["center"] - col_centers[0]["center"]
            if dx < page_width * 0.15:
                col_centers = col_centers[:1]

        is_multi_col = len(col_centers) >= 2
        centers_x = [c["center"] for c in col_centers] if is_multi_col else None

        # 4. Секции разделены «широкими» блоками (≥ 55% ширины).
        sections: list[list[dict]] = []
        current: list[dict] = []
        for lb in logical:
            if lb["w"] >= page_width * 0.55:
                if current:
                    sections.append(current)
                    current = []
                sections.append([lb])
            else:
                current.append(lb)
        if current:
            sections.append(current)

        # 5. Сортировка внутри секций.
        final: list = []
        for sec in sections:
            if len(sec) == 1 and sec[0]["w"] >= page_width * 0.55:
                for phys in sec[0]["boxes"]:
                    final.append(phys["box"])
                continue

            if is_multi_col:
                cols: list[list[dict]] = [[] for _ in centers_x]
                for lb in sec:
                    dists = [abs(lb["cx"] - cx) for cx in centers_x]
                    cols[dists.index(min(dists))].append(lb)
                for col in cols:
                    col.sort(key=lambda x: x["y1"])
                    for lb in col:
                        for phys in lb["boxes"]:
                            final.append(phys["box"])
            else:
                sec.sort(key=lambda x: x["y1"])
                for lb in sec:
                    for phys in lb["boxes"]:
                        final.append(phys["box"])

        return final

    @staticmethod
    def _cluster_centers(centers: list[float], gap: float) -> list[dict]:
        """
        Жадная 1D-кластеризация: соседние точки, расстояние между которыми
        меньше `gap`, идут в один кластер. Возвращает [{center, count}, ...].
        """
        if not centers:
            return []
        clusters: list[list[float]] = [[centers[0]]]
        for c in centers[1:]:
            if c - clusters[-1][-1] <= gap:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [{"center": sum(cl) / len(cl), "count": len(cl)} for cl in clusters]

    def _apply_nms(self, boxes: list, iom_threshold: float = 0.7) -> list:
        """
        Intersection-over-Min-Area NMS: удаляет боксы, у которых
        >iom_threshold площади перекрывается с уже принятым боксом.
        Приоритет: класс-приоритет, затем confidence.
        """
        if not boxes:
            return boxes

        def priority(box):
            cls = self.model.names[int(box.cls[0])].lower()
            return self._class_priority.get(cls, 3), float(box.conf[0])

        sorted_boxes = sorted(boxes, key=priority, reverse=True)
        kept: list = []

        for box in sorted_boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area == 0:
                continue

            dominated = False
            for kb in kept:
                kx1, ky1, kx2, ky2 = kb.xyxy[0].tolist()
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                min_area = min(area, k_area)
                if min_area > 0 and inter / min_area > iom_threshold:
                    dominated = True
                    break

            if not dominated:
                kept.append(box)

        return kept

    def _crop_image(
        self, img: Image.Image, coords: list, padding: int = 0
    ) -> Image.Image:
        x1 = max(0, coords[0] - padding)
        y1 = max(0, coords[1] - padding)
        x2 = min(img.width, coords[2] + padding)
        y2 = min(img.height, coords[3] + padding)
        return img.crop((x1, y1, x2, y2))

    def build_routing_plan(
        self, pdf_path: str, output_dir: str = "data/output", visualize: bool = False
    ) -> dict:
        doc_id = self.get_document_id(pdf_path)

        temp_dir = os.path.join(output_dir, "temp", f"document_{doc_id}")
        os.makedirs(temp_dir, exist_ok=True)

        vis_dir = None
        if visualize:
            vis_dir = os.path.join("data", "visualization", f"document_{doc_id}")
            os.makedirs(vis_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        routing_plan = {"doc_id": doc_id, "pdf_path": os.path.abspath(pdf_path), "pages": []}
        global_image_counter = 1

        print(f"\nАнализ: {os.path.basename(pdf_path)} (ID: {doc_id})")

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=400)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            img_cv2 = cv2.cvtColor(
                img_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
            )
            img_rgb = (
                cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) if pix.n == 4 else img_array
            )
            pil_img = Image.fromarray(img_rgb)

            results = self.model.predict(
                img_cv2, imgsz=1024, conf=0.2, device=self.device, verbose=False
            )[0]

            if visualize and vis_dir:
                annotated_frame = results.plot(pil=True, line_width=2, font_size=12)
                cv2.imwrite(
                    os.path.join(vis_dir, f"page_{page_num + 1}.jpg"), annotated_frame
                )

            filtered_boxes = self._apply_nms(list(results.boxes))
            sorted_boxes = self._sort_reading_order(filtered_boxes, pil_img.width, pil_img.height)
            page_plan = {
                "page_num": page_num + 1,
                "width_px": pil_img.width,
                "height_px": pil_img.height,
                "width_pt": page.rect.width,
                "height_pt": page.rect.height,
                "blocks": [],
            }

            for box in sorted_boxes:
                coords = [int(c) for c in box.xyxy[0].tolist()]
                label = self.model.names[int(box.cls[0])].lower()

                if label in self.ignore_classes:
                    continue

                block = {
                    "type": label,
                    "coords": coords,
                    "track": "DOCLING_TEXT",  # Текст и заголовки
                    "content_path": None,
                    "md_image_name": None,
                }

                if label in self.vlm_candidates:
                    block["track"] = "PADDLE_OCR"  # VLM
                elif label == "table":
                    block["track"] = "DOCLING_TABLE"  # Таблицы

                # Кропы только для OCR и таблиц
                if block["track"] in ["PADDLE_OCR", "DOCLING_TABLE"]:
                    prefix = (
                        "table" if block["track"] == "DOCLING_TABLE" else "candidate"
                    )
                    fname = (
                        f"{prefix}_{global_image_counter}.png"
                        if prefix == "candidate"
                        else f"table_p{page_num + 1}_{coords[1]}.png"
                    )
                    temp_path = os.path.join(temp_dir, fname)

                    self._crop_image(
                        pil_img, coords, padding=10 if label == "table" else 0
                    ).save(temp_path)

                    block["content_path"] = temp_path
                    if block["track"] == "PADDLE_OCR":
                        block["md_image_name"] = (
                            f"doc_{doc_id}_image_{global_image_counter}.png"
                        )
                        global_image_counter += 1

                page_plan["blocks"].append(block)

            # Ищем растровые объекты, которые пропустила YOLO
            physical_images = page.get_image_info(xrefs=True)
            for img in physical_images:
                img_rect = fitz.Rect(img["bbox"])

                if img_rect.width < 50 or img_rect.height < 50:
                    continue

                is_caught_by_yolo = False
                for box in sorted_boxes:
                    coords = [int(c) for c in box.xyxy[0].tolist()]
                    yolo_rect = fitz.Rect(
                        coords[0] * (72 / 400),
                        coords[1] * (72 / 400),
                        coords[2] * (72 / 400),
                        coords[3] * (72 / 400),
                    )
                    if (
                        yolo_rect.intersect(img_rect).get_area()
                        > img_rect.get_area() * 0.5
                    ):
                        is_caught_by_yolo = True
                        break

                if not is_caught_by_yolo:
                    missed_coords = [
                        int(img_rect.x0 * (400 / 72)),
                        int(img_rect.y0 * (400 / 72)),
                        int(img_rect.x1 * (400 / 72)),
                        int(img_rect.y1 * (400 / 72)),
                    ]

                    missed_coords = [
                        max(0, missed_coords[0]),
                        max(0, missed_coords[1]),
                        min(pil_img.width, missed_coords[2]),
                        min(pil_img.height, missed_coords[3]),
                    ]

                    if (
                        missed_coords[2] <= missed_coords[0]
                        or missed_coords[3] <= missed_coords[1]
                    ):
                        continue

                    block = {
                        "type": "missed_raster",
                        "coords": missed_coords,
                        "track": "PADDLE_OCR",
                        "content_path": None,
                        "md_image_name": None,
                    }

                    temp_path = os.path.join(
                        temp_dir, f"candidate_{global_image_counter}.png"
                    )
                    self._crop_image(pil_img, missed_coords, padding=5).save(temp_path)

                    block["content_path"] = temp_path
                    block["md_image_name"] = (
                        f"doc_{doc_id}_image_{global_image_counter}.png"
                    )
                    global_image_counter += 1

                    page_plan["blocks"].append(block)

            routing_plan["pages"].append(page_plan)
        
            if visualize and vis_dir:
                # Рисуем поверх оригинальной картинки
                img_draw = img_cv2.copy()
                
                # Отдельный счетчик для валидных (не игнорируемых) блоков
                valid_block_counter = 1 
                
                for box in sorted_boxes:
                    label = self.model.names[int(box.cls[0])].lower()
                    
                    # Пропускаем abandon и все остальные классы из ignore_classes
                    if label in self.ignore_classes:
                        continue

                    coords = [int(c) for c in box.xyxy[0].tolist()]
                    
                    # Отрисовываем рамку (зеленая)
                    cv2.rectangle(img_draw, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                    
                    # Рисуем порядковый номер и класс блока (например: "1 (text)")
                    order_text = f"{valid_block_counter} ({label})"
                    cv2.putText(
                        img_draw, 
                        order_text, 
                        (coords[0] + 5, coords[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, # Немного уменьшили шрифт, чтобы влезло название класса
                        (0, 0, 255), # Красный цвет (BGR)
                        2 # Толщина линии
                    )
                    
                    valid_block_counter += 1
                
                cv2.imwrite(os.path.join(vis_dir, f"page_{page_num + 1}_order.jpg"), img_draw)

        doc.close()
        return routing_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutRouter")
    parser.add_argument("--pdf", type=str, required=True, help="Путь к PDF")
    parser.add_argument("--visualize", action="store_true", help="Визуализация")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[ERROR] Файл не найден по пути '{args.pdf}'")
        exit(1)

    router = LayoutRouter()
    plan = router.build_routing_plan(args.pdf, visualize=args.visualize)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
