from __future__ import annotations


def describe_face_agent() -> dict:
    return {
        "name": "Face Detection Agent",
        "role": "Detects face presence, image quality, and facial landmarks.",
        "tools": ["OpenCV", "MediaPipe"],
    }

