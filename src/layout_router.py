import argparse
import json
import os

import cv2
import fitz
import numpy as np
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image


class LayoutRouter:
    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        self.device = self._get_device()
        self.model = self._load_model()
        self.ignore_classes = {
            "page-header",
            "page-footer",
            "footnote",
            "watermark",
            "abandon",
        }
        self.vlm_candidates = {"picture", "figure", "image"}

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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
        if not boxes:
            return []

        parsed_boxes = []
        for box in boxes:
            coords = box.xyxy[0].tolist()
            cls_name = self.model.names[int(box.cls[0])].lower()
            parsed_boxes.append(
                {
                    "box": box,
                    "x1": coords[0],
                    "y1": coords[1],
                    "x2": coords[2],
                    "y2": coords[3],
                    "w": coords[2] - coords[0],
                    "h": coords[3] - coords[1],
                    "cls": cls_name
                }
            )

        # 1. Фильтрация колонтитулов
        filtered_boxes = []
        margin_top = page_height * 0.05
        margin_bottom = page_height * 0.95
        
        for b in parsed_boxes:
            if b["cls"] in ["text", "title", "page-header", "page-footer"]:
                if b["y2"] < margin_top or b["y1"] > margin_bottom:
                    continue
            filtered_boxes.append(b)

        # Первичная сортировка сверху вниз
        filtered_boxes.sort(key=lambda x: x["y1"])

        # 2. ЛОГИЧЕСКАЯ ПРЕ-СКЛЕЙКА (Figure + Caption)
        # Связываем медиа-объекты с их подписями в единые неразрывные блоки
        logical_blocks = []
        used_indices = set()

        for i, b in enumerate(filtered_boxes):
            if i in used_indices:
                continue
                
            current_logical = [b]
            used_indices.add(i)
            
            base_box = b
            for j in range(i + 1, len(filtered_boxes)):
                if j in used_indices:
                    continue
                next_box = filtered_boxes[j]
                
                # Расстояние по вертикали (если пересекаются, gap отрицательный)
                y_gap = next_box["y1"] - base_box["y2"]
                
                # Доля пересечения по X
                overlap_x = max(0, min(base_box["x2"], next_box["x2"]) - max(base_box["x1"], next_box["x1"]))
                min_w = min(base_box["w"], next_box["w"])
                x_ratio = overlap_x / min_w if min_w > 0 else 0
                
                is_caption = next_box["cls"] in ["figure_caption", "table_caption", "caption"]
                is_parent_media = base_box["cls"] in ["figure", "picture", "image", "table", "table_merged", "table_borderless"]
                
                # Если блок находится близко под текущим (< 8% высоты страницы) и сильно выровнен по ширине
                if y_gap < page_height * 0.08 and x_ratio > 0.5:
                    # Склеиваем, если это картинка/таблица и текст под ней, либо если это явно подпись
                    if is_caption or is_parent_media:
                        current_logical.append(next_box)
                        used_indices.add(j)
                        base_box = next_box # Сдвигаем низ для цепочки
                    else:
                        break # Два обычных абзаца текста не склеиваем жестко
                elif y_gap >= page_height * 0.08:
                    pass # Ушли слишком далеко вниз
                    
            # Формируем габариты объединенного логического блока
            logical_blocks.append({
                "boxes": current_logical,
                "x1": min(cb["x1"] for cb in current_logical),
                "y1": min(cb["y1"] for cb in current_logical),
                "x2": max(cb["x2"] for cb in current_logical),
                "y2": max(cb["y2"] for cb in current_logical),
                "w": max(cb["x2"] for cb in current_logical) - min(cb["x1"] for cb in current_logical),
                "h": max(cb["y2"] for cb in current_logical) - min(cb["y1"] for cb in current_logical),
            })

        # Вспомогательные функции для работы с объединенными блоками
        def get_y_overlap_ratio(b1, b2):
            overlap = max(0, min(b1["y2"], b2["y2"]) - max(b1["y1"], b2["y1"]))
            min_h = min(b1["h"], b2["h"])
            return overlap / min_h if min_h > 0 else 0

        def get_x_overlap_ratio(b1, b2):
            overlap = max(0, min(b1["x2"], b2["x2"]) - max(b1["x1"], b2["x1"]))
            min_w = min(b1["w"], b2["w"])
            return overlap / min_w if min_w > 0 else 0

        # 3. Группируем логические блоки в горизонтальные полосы (Bands)
        bands = []
        current_band = []

        for lb in logical_blocks:
            is_full_width = lb["w"] > page_width * 0.55
            
            if is_full_width:
                if current_band:
                    bands.append(current_band)
                    current_band = []
                bands.append([lb])
            else:
                if not current_band:
                    current_band.append(lb)
                else:
                    if any(get_y_overlap_ratio(lb, cb) > 0.1 for cb in current_band):
                        current_band.append(lb)
                    else:
                        bands.append(current_band)
                        current_band = [lb]
                        
        if current_band:
            bands.append(current_band)

        # 4. Формируем колонки внутри каждой полосы и распаковываем
        final_sorted = []
        for band in bands:
            if len(band) <= 1:
                for lb in band:
                    for phys_box in lb["boxes"]:
                        final_sorted.append(phys_box["box"])
                continue
                
            columns = []
            for lb in band:
                placed = False
                for col in columns:
                    if any(get_x_overlap_ratio(lb, cb) > 0.1 for cb in col):
                        col.append(lb)
                        placed = True
                        break
                
                if not placed:
                    columns.append([lb])

            # Сортируем колонки слева направо
            columns.sort(key=lambda col: min(cb["x1"] for cb in col))
            
            # Читаем элементы сверху вниз внутри колонки
            for col in columns:
                col.sort(key=lambda x: x["y1"])
                for lb in col:
                    # Внутри логического блока физические боксы уже отсортированы сверху вниз
                    for phys_box in lb["boxes"]:
                        final_sorted.append(phys_box["box"])

        return final_sorted

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
        routing_plan = {"doc_id": doc_id, "pages": []}
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

            sorted_boxes = self._sort_reading_order(results.boxes, pil_img.width, pil_img.height)
            page_plan = {"page_num": page_num + 1, "blocks": []}

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
