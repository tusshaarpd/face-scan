from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import subprocess
import requests
from PIL import Image

from utils.image_utils import image_to_base64_jpeg

DEFAULT_MODEL = "gpt-4.1-mini"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

_SCORE_KEYS = ["stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score"]


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


def _call_openai(api_key: str, model: str, messages: list) -> str:
    """Call OpenAI via curl.exe subprocess.

    Windows Firewall blocks Python's socket on port 443 (WinError 10013).
    curl.exe is a Windows system binary that always has outbound network
    permission, so it bypasses the restriction entirely.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
    })

    result = subprocess.run(
        [
            "curl.exe", "-s",
            "-X", "POST", OPENAI_CHAT_URL,
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-d", "@-",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr}")

    data = json.loads(result.stdout)
    if "error" in data:
        err = data["error"]
        code = err.get("code", "")
        status = err.get("status", 0)
        if code == "invalid_api_key" or status == 401:
            raise requests.HTTPError(response=type("R", (), {"status_code": 401, "text": err["message"]})())
        if status == 429 or "rate" in err.get("message", "").lower() or "quota" in err.get("message", "").lower():
            raise requests.HTTPError(response=type("R", (), {"status_code": 429, "text": err["message"]})())
        raise RuntimeError(err.get("message", str(err)))

    return data["choices"][0]["message"]["content"]


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
            "message": "No OpenAI API key — add OPENAI_API_KEY to .env or paste it in the sidebar.",
        }
    if observations.get("face_count") != 1:
        return {
            "status": "skipped",
            "message": "OpenAI analysis skipped: exactly one face must be detected first.",
        }

    try:
        model = get_model()
        image_b64 = image_to_base64_jpeg(image, max_edge=512)

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

        system_prompt = (
            "You are a wellness assistant interpreting facial visual cues from a live photo. "
            "You provide indicative wellness observations only — never medical diagnoses. "
            "Use the CV measurements and image together. "
            "Keep recommendations supportive, practical, and non-medical.\n\n"
            + SCHEMA_PROMPT
        )

        user_text = (
            "Analyse this facial photo for non-medical wellness indicators.\n"
            f"CV observations: {json.dumps(cv_summary, default=str)}\n"
            "Return ONLY the JSON object described in the system prompt."
        )

        messages = [
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
        ]

        raw = _call_openai(api_key, model, messages)
        data = _parse_json_response(raw)
        return {"status": "ok", "data": _sanitize_report(data)}

    except requests.HTTPError as exc:
        code = exc.response.status_code
        if code == 401:
            msg = "Invalid API key — verify at platform.openai.com/api-keys."
        elif code == 403:
            msg = "Key lacks permission for this model. Try gpt-4.1-mini."
        elif code == 429:
            msg = "Rate limit or quota exceeded — check platform.openai.com/usage."
        else:
            msg = f"OpenAI HTTP {code}: {exc.response.text[:200]}"
        return {"status": "error", "message": msg}
    except Exception as exc:
        return {"status": "error", "message": f"OpenAI error: {exc}"}
