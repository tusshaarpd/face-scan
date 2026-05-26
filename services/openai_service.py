from __future__ import annotations

import json
import os
from typing import Any

from PIL import Image

from utils.image_utils import image_to_base64_jpeg

DEFAULT_MODEL = "gpt-4o-mini"

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

        # Sidebar selector takes priority, then secrets, then env var
        if "openai_model" in st.session_state:
            return str(st.session_state["openai_model"])
        if "OPENAI_MODEL" in st.secrets:
            return str(st.secrets["OPENAI_MODEL"])
    except Exception:
        pass
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _sanitize_report(data: dict[str, Any]) -> dict[str, Any]:
    for key in ["stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score"]:
        data[key] = round(max(0, min(100, float(data.get(key, 0)))), 1)
    data["confidence"] = round(max(0, min(1, float(data.get("confidence", 0)))), 2)
    data["recovery_need"] = (
        data.get("recovery_need") if data.get("recovery_need") in {"Low", "Medium", "High"} else "Medium"
    )
    for key in ["recommendations", "contributing_factors", "limitations"]:
        value = data.get(key, [])
        data[key] = value if isinstance(value, list) else [str(value)]
    return data


def analyze_with_openai(
    image: Image.Image, observations: dict[str, Any], api_key: str | None = None
) -> dict[str, Any]:
    api_key = api_key or get_api_key()
    if not api_key:
        return {
            "status": "unavailable",
            "message": "No OpenAI API key — enter your key in the sidebar to enable AI analysis.",
        }
    if observations.get("face_count") != 1:
        return {
            "status": "skipped",
            "message": "OpenAI analysis skipped: exactly one face must be detected first.",
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=45)
        image_b64 = image_to_base64_jpeg(image)
        model = get_model()

        system_prompt = (
            "You are a wellness assistant that interprets facial visual cues from a live photo. "
            "You provide indicative wellness observations only — never medical diagnoses. "
            "You will be given computer vision observations alongside the image. "
            "Use both sources to fill the JSON schema accurately. "
            "Keep all recommendations supportive, practical, and non-medical."
        )

        user_text = json.dumps(
            {
                "task": "Analyse this facial image for non-medical wellness indicators.",
                "cv_observations": observations,
                "must_not": [
                    "Do not diagnose disease or mental illness.",
                    "Do not claim clinical accuracy.",
                    "Do not replace professional medical advice.",
                ],
                "output_instruction": "Return valid JSON exactly matching the provided schema.",
            },
            default=str,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "facial_wellness_report",
                    "strict": True,
                    "schema": REPORT_SCHEMA,
                },
            },
            max_tokens=1024,
        )

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        return {"status": "ok", "data": _sanitize_report(data)}

    except Exception as exc:
        return {
            "status": "error",
            "message": f"AI analysis error: {exc}",
        }
