from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_IMAGE_EDGE = 1280
JPEG_QUALITY = 86


def load_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    return ImageOps.exif_transpose(image).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def resize_for_analysis(image: Image.Image, max_edge: int = MAX_IMAGE_EDGE) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    largest = max(width, height)
    if largest <= max_edge:
        return image
    scale = max_edge / largest
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def image_to_base64_jpeg(image: Image.Image, max_edge: int = 960) -> str:
    compressed = resize_for_analysis(image, max_edge=max_edge)
    buffer = io.BytesIO()
    compressed.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def draw_overlay(image: Image.Image, observations: dict) -> Image.Image:
    canvas = pil_to_cv(image).copy()
    overlay = observations.get("overlay", {})

    for box in overlay.get("boxes", []):
        x1, y1, x2, y2 = [int(v) for v in box.get("coords", [0, 0, 0, 0])]
        color = tuple(int(v) for v in box.get("color", [79, 70, 229]))
        label = box.get("label", "")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(canvas, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for point in overlay.get("landmarks", []):
        x, y = int(point[0]), int(point[1])
        cv2.circle(canvas, (x, y), 1, (34, 197, 94), -1)

    heatmap = np.zeros_like(canvas, dtype=np.uint8)
    for region in overlay.get("heat_regions", []):
        cx, cy, radius, intensity = region
        color = (0, int(120 * intensity), int(255 * intensity))
        cv2.circle(heatmap, (int(cx), int(cy)), int(radius), color, -1)

    if np.any(heatmap):
        canvas = cv2.addWeighted(canvas, 0.82, heatmap, 0.32, 0)

    return cv_to_pil(canvas)

