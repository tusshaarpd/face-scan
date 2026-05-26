from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

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
    """Call OpenAI via a PowerShell .ps1 script (Windows WinHTTP stack).

    Python sockets (WinError 10013) and curl (exit 7) are blocked inside
    Streamlit's process context. PowerShell's Invoke-RestMethod uses the
    Windows HTTP stack and is always permitted.
    Script is written to a .ps1 file so multi-line execution is reliable.
    """
    payload_str = json.dumps({
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
    })

    fd_json, json_path = tempfile.mkstemp(suffix=".json")
    fd_ps,   ps1_path  = tempfile.mkstemp(suffix=".ps1")
    try:
        # Write JSON payload
        with os.fdopen(fd_json, "w", encoding="utf-8") as f:
            f.write(payload_str)

        # Escape backslashes for PowerShell string literals
        safe_json = json_path.replace("\\", "\\\\")
        safe_key  = api_key.replace("'", "''")   # escape single quotes

        ps_script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
            f"$body = [System.IO.File]::ReadAllText('{safe_json}')\n"
            "$headers = New-Object 'System.Collections.Generic.Dictionary[[String],[String]]'\n"
            f"$headers.Add('Authorization', 'Bearer {safe_key}')\n"
            "$headers.Add('Content-Type', 'application/json')\n"
            "try {\n"
            f"    $r = Invoke-RestMethod -Uri '{OPENAI_CHAT_URL}' "
            "-Method POST -Headers $headers -Body $body -ContentType 'application/json'\n"
            "    $r.choices[0].message.content\n"
            "} catch {\n"
            "    $code = $_.Exception.Response.StatusCode.value__\n"
            "    $msg  = $_.Exception.Message\n"
            "    Write-Output (\"OPENAI_ERROR:\" + $code + \":\" + $msg)\n"
            "    exit 1\n"
            "}\n"
        )

        with os.fdopen(fd_ps, "w", encoding="utf-8") as f:
            f.write(ps_script)

        result = subprocess.run(
            [
                "powershell.exe",
                "-NonInteractive", "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", ps1_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )

        stdout = result.stdout.strip()

        # Errors written to stdout as OPENAI_ERROR:code:msg
        if stdout.startswith("OPENAI_ERROR:"):
            parts = stdout.split(":", 2)
            code = parts[1] if len(parts) > 1 else ""
            msg  = parts[2] if len(parts) > 2 else stdout
            if code == "401" or "Unauthorized" in msg:
                raise PermissionError("Invalid API key — verify at platform.openai.com/api-keys.")
            if code == "429" or "quota" in msg.lower() or "rate" in msg.lower():
                raise RuntimeError("Rate limit or quota exceeded — check platform.openai.com/usage.")
            raise RuntimeError(f"OpenAI {code}: {msg[:200]}")

        if result.returncode != 0:
            err = (stdout or result.stderr.strip())[:300]
            raise RuntimeError(f"PowerShell script failed: {err}")

        if not stdout:
            raise RuntimeError("Empty response from OpenAI.")

        return stdout

    finally:
        for p in (json_path, ps1_path):
            try:
                os.unlink(p)
            except Exception:
                pass


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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a wellness assistant interpreting facial visual cues. "
                    "Provide indicative wellness observations only — never medical diagnoses. "
                    "Use the CV measurements and the photo together. "
                    "Keep recommendations supportive and non-medical.\n\n"
                    + SCHEMA_PROMPT
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyse this facial photo for non-medical wellness indicators. "
                            f"CV observations: {json.dumps(cv_summary, default=str)}. "
                            "Return ONLY the JSON object described in the system prompt."
                        ),
                    },
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

    except PermissionError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"OpenAI error: {exc}"}
