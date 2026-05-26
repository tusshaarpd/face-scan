from __future__ import annotations


def describe_eye_agent() -> dict:
    return {
        "name": "Eye Fatigue Agent",
        "role": "Reviews eye openness, asymmetry, and under-eye fatigue proxies.",
        "tools": ["MediaPipe landmarks", "OpenCV region metrics"],
    }

