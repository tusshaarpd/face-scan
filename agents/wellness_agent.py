from __future__ import annotations


def describe_wellness_agent() -> dict:
    return {
        "name": "Wellness Coach Agent",
        "role": "Creates supportive recovery suggestions without medical claims.",
        "tools": ["OpenAI GPT-4o", "wellness recommendation rules"],
    }

