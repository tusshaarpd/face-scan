from __future__ import annotations

import importlib.util


OPTIONAL_PACKAGES = {
    "deepface": "DeepFace",
    "fer": "FER",
    "face_recognition": "face_recognition",
    "dlib": "dlib",
}


def optional_analyzer_status() -> dict[str, bool]:
    return {
        label: importlib.util.find_spec(module_name) is not None
        for module_name, label in OPTIONAL_PACKAGES.items()
    }

