from __future__ import annotations


def describe_stress_agent() -> dict:
    return {
        "name": "Stress Interpretation Agent",
        "role": "Converts visual observations into non-medical stress and fatigue estimates.",
        "tools": ["OpenAI GPT-4o", "local scoring fallback"],
    }

