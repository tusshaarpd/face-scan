from __future__ import annotations

import json
import os
from typing import Any

from PIL import Image

from utils.image_utils import image_to_base64_jpeg

DEFAULT_MODEL = "gpt-4o-mini"

# Expected keys and their valid ranges — used for manual validation
# when json_object mode is used instead of strict json_schema.
_SCORE_KEYS = ["stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score"]

SCHEMA_PROMPT = """\
Return ONLY a JSON object with exactly these keys (no extra keys, no markdown):
{
  "stress_score": <number 0-100>,
  "fatigue_score": <number 0-100>,
  "eye_strain": <number 0-100>,
  "recovery_score": <number 0-100>,
  "wellness_score": <number 0-100>,
  "recovery_need": <"Low" | "Medium" | "High">,
  "wellness_summary": <string>,
  "recommendations": [<string>, ...],  (1-6 items)
  "confidence": <number 0.0-1.0>,
  "contributing_factors": [<string>, ...],  (1-8 items)
  "limitations": [<string>, ...]  (1-6 items)
}"""


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

        if "openai_model" in st.session_state:
            return str(st.session_state["openai_model"])
        if "OPENAI_MODEL" in st.secrets:
            return str(st.secrets["OPENAI_MODEL"])
    except Exception:
        pass
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _sanitize_report(data: dict[str, Any]) -> dict[str, Any]:
    for key in _SCORE_KEYS:
        data[key] = round(max(0, min(100, float(data.get(key, 0)))), 1)
    data["confidence"] = round(max(0, min(1, float(data.get("confidence", 0)))), 2)
    data["recovery_need"] = (
        data.get("recovery_need") if data.get("recovery_need") in {"Low", "Medium", "High"} else "Medium"
    )
    for key in ["recommendations", "contributing_factors", "limitations"]:
        value = data.get(key, [])
        data[key] = value if isinstance(value, list) else [str(value)]
    for key in _SCORE_KEYS:
        if key not in data:
            data[key] = 0.0
    for key in ["wellness_summary", "recovery_need"]:
        if key not in data:
            data[key] = ""
    return data


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON from the response, stripping any markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)


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

        client = OpenAI(api_key=api_key, timeout=60)
        model = get_model()

        # Compress image to 512px max — keeps payload small, sufficient for vision
        image_b64 = image_to_base64_jpeg(image, max_edge=512)

        system_prompt = (
            "You are a wellness assistant that interprets facial visual cues from a live photo. "
            "You provide indicative wellness observations only — never medical diagnoses. "
            "You will receive computer vision measurements alongside the photo. "
            "Use both sources together. Keep recommendations supportive and non-medical. "
            + SCHEMA_PROMPT
        )

        cv_summary = {
            k: observations.get(k)
            for k in [
                "face_count", "eye_openness", "eye_asymmetry",
                "under_eye_darkness", "forehead_tension", "jaw_tension",
                "blur_quality", "lighting_quality", "landmark_quality",
                "stress_score_local", "fatigue_score_local",
                "eye_strain_local", "wellness_score_local", "issues",
            ]
            if observations.get(k) is not None
        }

        user_text = (
            "Analyse this facial photo for non-medical wellness indicators.\n"
            f"CV observations: {json.dumps(cv_summary, default=str)}\n"
            "Do NOT diagnose disease, mental illness, or any medical condition.\n"
            "Return ONLY the JSON object described in the system prompt."
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
            response_format={"type": "json_object"},
            max_tokens=1024,
        )

        raw = response.choices[0].message.content or "{}"
        data = _parse_json_response(raw)
        return {"status": "ok", "data": _sanitize_report(data)}

    except Exception as exc:
        err = str(exc)
        if "401" in err or "invalid_api_key" in err or "Incorrect API key" in err:
            msg = "Invalid API key. Check your key at platform.openai.com → API Keys."
        elif "403" in err or "permission" in err.lower():
            msg = "API key does not have permission for this model. Try gpt-4o-mini."
        elif "429" in err or "quota" in err.lower() or "rate" in err.lower():
            msg = "Rate limit or quota exceeded. Check your usage at platform.openai.com → Usage."
        elif "Connection" in err or "connect" in err.lower():
            msg = (
                "Connection to OpenAI failed. "
                "Check your internet, disable any VPN/firewall blocking api.openai.com, "
                "or check if antivirus is intercepting HTTPS traffic."
            )
        else:
            msg = f"OpenAI error: {exc}"
        return {"status": "error", "message": msg}
