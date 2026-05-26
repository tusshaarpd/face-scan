from __future__ import annotations


DISCLAIMER = (
    "This system is an AI wellness assistant and not a medical diagnostic tool. "
    "Results are estimations based on visual indicators only."
)


def describe_report_agent() -> dict:
    return {
        "name": "Report Generator Agent",
        "role": "Builds the final explainable wellness report with the required disclaimer.",
        "tools": ["Structured JSON report", "Streamlit dashboard"],
    }


def attach_disclaimer(report: dict) -> dict:
    report = dict(report)
    report["disclaimer"] = DISCLAIMER
    return report

