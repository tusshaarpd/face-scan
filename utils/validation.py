from __future__ import annotations

from PIL import Image

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG"}
DISCLAIMER = (
    "This system is an AI wellness assistant and not a medical diagnostic tool. "
    "Results are estimations based on visual indicators only."
)


def validate_image_file(uploaded_file) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "No image was provided."
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if suffix not in {"jpg", "jpeg", "png"}:
        return False, "Unsupported image format. Please upload JPG, JPEG, or PNG."
    return True, ""


def validate_dimensions(image: Image.Image) -> tuple[bool, str]:
    width, height = image.size
    if width < 240 or height < 240:
        return False, "Image is too small for reliable landmark analysis. Use at least 240x240 pixels."
    if width * height > 16_000_000:
        return False, "Image is very large. It will be compressed before analysis."
    return True, ""

