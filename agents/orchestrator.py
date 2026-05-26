from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.eye_agent import describe_eye_agent
from agents.face_agent import describe_face_agent
from agents.report_agent import DISCLAIMER, attach_disclaimer, describe_report_agent
from agents.stress_agent import describe_stress_agent
from agents.wellness_agent import describe_wellness_agent
from services.openai_service import analyze_with_openai
from services.stress_service import build_local_report


@dataclass
class AgentRun:
    report: dict[str, Any]
    trace: list[dict[str, str]]
    used_openai: bool
    used_crewai: bool


_SCORE_KEYS = ["stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score"]


def _agent_trace() -> list[dict[str, str]]:
    return [
        describe_face_agent(),
        describe_eye_agent(),
        describe_stress_agent(),
        describe_wellness_agent(),
        describe_report_agent(),
    ]


def _try_build_crewai_agents() -> bool:
    try:
        from crewai import Agent  # noqa: F401
    except Exception:
        return False
    return True


def _blend_reports(local: dict[str, Any], ai: dict[str, Any], ai_weight: float = 0.6) -> dict[str, Any]:
    """Blend local ML and OpenAI Vision scores. AI weighted at 60%, local CV at 40%."""
    ml_weight = 1.0 - ai_weight
    blended = dict(ai)
    for key in _SCORE_KEYS:
        ai_val = float(ai.get(key, 0))
        ml_val = float(local.get(key, 0))
        blended[key] = round(ai_val * ai_weight + ml_val * ml_weight, 1)
    blended["confidence"] = round(
        min(0.95, float(ai.get("confidence", 0)) * ai_weight + float(local.get("confidence", 0)) * ml_weight), 2
    )
    return blended


def run_wellness_crew(image, cv_observations: dict[str, Any], api_key: str | None = None) -> AgentRun:
    """Run both local ML and OpenAI Vision analyses, then blend results.

    Local ML (MediaPipe + OpenCV) always runs for instant facial landmark
    analysis. OpenAI Vision runs concurrently when an API key is available.
    When both succeed, scores are blended (OpenAI 60% + ML 40%) so the result
    is informed by both the geometric CV signals and GPT-4o's visual reasoning.
    Local ML is the sole fallback when OpenAI is unavailable.
    """
    trace = _agent_trace()
    used_crewai = _try_build_crewai_agents()

    # Local ML always runs first — fast and key-free
    local_report = build_local_report(cv_observations)

    ai_result = analyze_with_openai(image, cv_observations, api_key=api_key)
    if ai_result.get("status") == "ok":
        ai_report = attach_disclaimer(ai_result["data"])
        blended = _blend_reports(local_report, ai_report)
        blended["analysis_mode"] = "Dual Analysis (ML + OpenAI Vision)"
        # Store both individual score sets for the comparison dashboard section
        blended["ml_scores"] = {k: round(float(local_report.get(k, 0)), 1) for k in _SCORE_KEYS}
        blended["ai_scores"] = {k: round(float(ai_report.get(k, 0)), 1) for k in _SCORE_KEYS}
        return AgentRun(report=blended, trace=trace, used_openai=True, used_crewai=used_crewai)

    local_report["ai_status"] = ai_result.get("message", "AI analysis unavailable.")
    local_report["analysis_mode"] = "Local ML Only (OpenAI unavailable)"
    local_report["disclaimer"] = DISCLAIMER
    return AgentRun(report=local_report, trace=trace, used_openai=False, used_crewai=used_crewai)

