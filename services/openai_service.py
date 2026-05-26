from __future__ import annotations

import json
import os
from typing import Any

from PIL import Image

from utils.image_utils import image_to_base64_jpeg

DEFAULT_MODEL = "gpt-4o-2024-08-06"

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "stress_score": {"type": "number", "minimum": 0, "maximum": 100},
        "fatigue_score": {"type": "number", "minimum": 0, "maximum": 100},
        "eye_strain": {"type": "number", "minimum": 0, "maximum": 100},
        "recovery_score": {"type": "number", "minimum": 0, "maximum": 100},
        "wellness_score": {"type": "number", "minimum": 0, "maximum": 100},
        "recovery_need": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "wellness_summary": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "contributing_factors": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
        "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
    },
    "required": [
        "stress_score",
        "fatigue_score",
        "eye_strain",
        "recovery_score",
        "wellness_score",
        "recovery_need",
        "wellness_summary",
        "recommendations",
        "confidence",
        "contributing_factors",
        "limitations",
    ],
}


def get_api_key() -> str | None:
    try:
        import streamlit as st

        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_model() -> str:
    try:
        import streamlit as st

        if "OPENAI_MODEL" in st.secrets:
            return str(st.secrets["OPENAI_MODEL"])
    except Exception:
        pass
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _extract_text(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _sanitize_report(data: dict[str, Any]) -> dict[str, Any]:
    for key in ["stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score"]:
        data[key] = round(max(0, min(100, float(data.get(key, 0)))), 1)
    data["confidence"] = round(max(0, min(1, float(data.get("confidence", 0)))), 2)
    data["recovery_need"] = data.get("recovery_need") if data.get("recovery_need") in {"Low", "Medium", "High"} else "Medium"
    for key in ["recommendations", "contributing_factors", "limitations"]:
        value = data.get(key, [])
        data[key] = value if isinstance(value, list) else [str(value)]
    return data


def analyze_with_openai(image: Image.Image, observations: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    api_key = api_key or get_api_key()
    if not api_key:
        return {"status": "unavailable", "message": "OpenAI API key is not configured. Using local CV fallback."}
    if observations.get("face_count") != 1:
        return {"status": "skipped", "message": "OpenAI analysis skipped until exactly one face is detected."}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=35)
        image_b64 = image_to_base64_jpeg(image)
        prompt = {
            "task": "Analyze one facial image for non-medical wellness indicators only.",
            "must_not": [
                "Do not diagnose disease.",
                "Do not detect or claim mental illness.",
                "Do not claim clinical accuracy.",
                "Do not replace professional medical advice.",
            ],
            "required_disclaimer": (
                "This system is an AI wellness assistant and not a medical diagnostic tool. "
                "Results are estimations based on visual indicators only."
            ),
            "cv_observations": observations,
            "output": "Return JSON matching the schema. Keep recommendations supportive, practical, and non-medical.",
        }

        response = client.responses.create(
            model=get_model(),
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a wellness assistant interpreting facial visual cues. "
                                "You provide indicative wellness observations only, never diagnoses. "
                                "Return JSON only."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(prompt, default=str)},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "facial_wellness_report",
                    "strict": True,
                    "schema": REPORT_SCHEMA,
                }
            },
        )
        data = json.loads(_extract_text(response))
        return {"status": "ok", "data": _sanitize_report(data)}
    except Exception as exc:
        return {"status": "error", "message": f"AI analysis unavailable: {exc}. Using local CV fallback."}
