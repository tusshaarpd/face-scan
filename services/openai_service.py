from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from utils.image_utils import image_to_base64_jpeg

DEFAULT_MODEL = "gpt-4o-mini"

_SCORE_KEYS = ["stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score"]

# Load .env from project root once at import time
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except Exception:
        pass

_load_env()

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
  "recommendations": [<string>, ...],
  "confidence": <number 0.0-1.0>,
  "contributing_factors": [<string>, ...],
  "limitations": [<string>, ...]
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


def _make_client(api_key: str):
    """Build an OpenAI client with an explicit httpx transport.

    Forces HTTP/1.1 and uses certifi's CA bundle — avoids SSL handshake
    failures caused by the SDK's default httpx configuration on some
    Windows setups.
    """
    import certifi
    import httpx
    from openai import OpenAI

    http_client = httpx.Client(
        verify=certifi.where(),
        http2=False,
        timeout=60,
    )
    return OpenAI(api_key=api_key, http_client=http_client)


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
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)


def analyze_with_openai(
    image: Image.Image, observations: dict[str, Any], api_key: str | None = None
) -> dict[str, Any]:
    api_key = api_key or get_api_key()
    if not api_key or api_key.strip() == "sk-your-key-here":
        return {
            "status": "unavailable",
            "message": "No OpenAI API key — paste your key in the .env file or the sidebar.",
        }
    if observations.get("face_count") != 1:
        return {
            "status": "skipped",
            "message": "OpenAI analysis skipped: exactly one face must be detected first.",
        }

    try:
        client = _make_client(api_key)
        model = get_model()
        image_b64 = image_to_base64_jpeg(image, max_edge=512)

        system_prompt = (
            "You are a wellness assistant that interprets facial visual cues from a live photo. "
            "You provide indicative wellness observations only — never medical diagnoses. "
            "Use the CV measurements and image together. "
            "Keep all recommendations supportive, practical, and non-medical.\n\n"
            + SCHEMA_PROMPT
        )

        cv_summary = {
            k: observations.get(k)
            for k in [
                "eye_openness", "eye_asymmetry", "under_eye_darkness",
                "forehead_tension", "jaw_tension", "blur_quality",
                "lighting_quality", "stress_score_local", "fatigue_score_local",
                "eye_strain_local", "wellness_score_local", "issues",
            ]
            if observations.get(k) is not None
        }

        user_text = (
            "Analyse this facial photo for non-medical wellness indicators.\n"
            f"CV observations: {json.dumps(cv_summary, default=str)}\n"
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
        if "401" in err or "invalid_api_key" in err:
            msg = "Invalid API key — check your key at platform.openai.com/api-keys."
        elif "403" in err or "permission" in err.lower():
            msg = "Key has no permission for this model. Try switching to gpt-4o-mini in the sidebar."
        elif "429" in err or "quota" in err.lower() or "rate" in err.lower():
            msg = "Rate limit or quota exceeded — check platform.openai.com/usage."
        elif "Connection" in err or "connect" in err.lower() or "SSL" in err:
            msg = f"Network error reaching OpenAI: {err}"
        else:
            msg = f"OpenAI error: {err}"
        return {"status": "error", "message": msg}
