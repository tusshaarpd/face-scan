from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.orchestrator import run_wellness_crew
from agents.report_agent import DISCLAIMER
from services.db_service import init_db, recent_scans, save_scan
from services.openai_service import get_api_key
from services.optional_analyzers import optional_analyzer_status
from services.stress_service import analyze_face
from utils.image_utils import draw_overlay, load_image, resize_for_analysis
from utils.plotting_utils import facial_zone_heatmap, gauge, radar_chart, trend_chart, wellness_gauge
from utils.validation import validate_dimensions, validate_image_file

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AI Facial Stress & Wellness Analyzer",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def bootstrap_database() -> bool:
    init_db()
    return True


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0F172A; }
        [data-testid="stSidebar"] { background: #111827; }
        .hero {
            padding: 1.2rem 0 0.2rem 0;
        }
        .hero h1 {
            font-size: clamp(2rem, 5vw, 4.2rem);
            line-height: 1.02;
            margin-bottom: 0.4rem;
        }
        .muted { color: #CBD5E1; }
        .notice {
            border: 1px solid #334155;
            background: #111827;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: #E2E8F0;
        }
        .metric-card {
            border: 1px solid #334155;
            background: #111827;
            border-radius: 8px;
            padding: 1rem;
            min-height: 110px;
        }
        .metric-card .label { color: #94A3B8; font-size: 0.82rem; }
        .metric-card .value {
            color: #F8FAFC;
            font-size: 1.18rem;
            line-height: 1.35;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .agent-pill {
            display: inline-block;
            border: 1px solid #334155;
            border-radius: 999px;
            padding: 0.32rem 0.65rem;
            margin: 0.18rem;
            color: #DBEAFE;
            background: #172554;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def disclaimer_box() -> None:
    st.markdown(f"<div class='notice'><strong>Disclaimer:</strong> {DISCLAIMER}</div>", unsafe_allow_html=True)


def sidebar() -> str:
    st.sidebar.title("Wellness Analyzer")
    page = st.sidebar.radio("Navigate", ["Home", "Scan", "Trends"], label_visibility="collapsed")
    st.sidebar.divider()

    st.sidebar.markdown("#### OpenAI API Key")
    st.sidebar.caption(
        "Paste your key to enable Dual ML + AI analysis. "
        "Get a key at platform.openai.com → API Keys. "
        "Stored in this session only — never saved to disk."
    )
    saved_key = st.session_state.get("openai_api_key", "")
    key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value=saved_key,
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed",
    )
    if key_input:
        st.session_state["openai_api_key"] = key_input
    elif not key_input and saved_key:
        st.session_state.pop("openai_api_key", None)

    effective_key = st.session_state.get("openai_api_key") or get_api_key()
    if effective_key:
        st.sidebar.success("OpenAI key active — Dual ML + AI mode enabled")
    else:
        st.sidebar.warning("No OpenAI key — Local ML analysis only")

    st.sidebar.markdown("#### AI Model")
    model_choice = st.sidebar.selectbox(
        "AI Model",
        options=["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano"],
        index=0,
        label_visibility="collapsed",
        help="gpt-4.1-mini: fast and low-cost. gpt-4.1: richest analysis. gpt-4.1-nano: cheapest.",
    )
    st.session_state["openai_model"] = model_choice

    with st.sidebar.expander("Optional analyzers"):
        for label, available in optional_analyzer_status().items():
            st.write(f"{'Available' if available else 'Not installed'}: {label}")
    st.sidebar.caption("Images are processed in memory. Numeric trend data only is stored.")
    return page


def home_page() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1>AI Facial Stress & Wellness Analyzer</h1>
            <p class="muted">Indicative facial wellness observations, recovery guidance, and private trend tracking.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    disclaimer_box()
    st.write("")
    cols = st.columns(4)
    summaries = [
        ("Live Capture", "Webcam or image upload"),
        ("ML Analysis", "MediaPipe + OpenCV signals"),
        ("AI Vision", "GPT-4o dual analysis (add key)"),
        ("Trends", "SQLite numeric history"),
    ]
    for col, (label, value) in zip(cols, summaries):
        col.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>", unsafe_allow_html=True)

    st.subheader("How It Works")
    st.write(
        "**Step 1 — Live capture:** Use the webcam tab on the Scan page to take a photo, or upload an image. "
        "**Step 2 — ML analysis:** MediaPipe face mesh and OpenCV extract eye openness, under-eye darkness, "
        "forehead tension, jaw tension, and image quality — instantly, with no API key. "
        "**Step 3 — AI Vision (optional):** When an OpenAI API key is entered in the sidebar, GPT-4o "
        "also analyses the live photo. Final scores are blended: OpenAI 60% + local ML 40%."
    )
    st.info("Add your OpenAI key in the sidebar to unlock Dual Analysis mode, then go to Scan. Consent is required before any image is analyzed.")


def source_selector():
    st.subheader("Before Scan")
    disclaimer_box()
    consent = st.checkbox("I consent to temporary in-memory facial image processing for wellness estimation.")
    if not consent:
        st.warning("Consent is required before scanning.")
        return None

    tab_camera, tab_upload = st.tabs(["Webcam Capture", "Image Upload"])
    with tab_camera:
        st.caption("Use even lighting, center your face, and keep the camera near eye level.")
        st.info(
            "If the camera preview does not open, allow camera permission for this browser tab. "
            "If capture works but no face is detected, retake with your full face centered and looking forward."
        )
        if st.button("Start 3-second readiness countdown", use_container_width=True):
            placeholder = st.empty()
            for remaining in [3, 2, 1]:
                placeholder.info(f"Get ready... {remaining}")
                time.sleep(1)
            placeholder.success("Capture when ready.")
        camera_file = st.camera_input("Capture a clear face image", label_visibility="collapsed")
        if camera_file:
            st.success("Webcam image captured. Running face detection now.")
            return camera_file
    with tab_upload:
        uploaded_file = st.file_uploader("Upload JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            valid, message = validate_image_file(uploaded_file)
            if not valid:
                st.error(message)
                return None
            return uploaded_file
    return None


def run_analysis(uploaded_file) -> None:
    try:
        image = resize_for_analysis(load_image(uploaded_file))
    except Exception as exc:
        st.error(f"Unsupported or unreadable image: {exc}")
        return

    valid_dims, dim_message = validate_dimensions(image)
    if not valid_dims:
        st.error(dim_message)
        return
    if dim_message:
        st.info(dim_message)

    api_key = st.session_state.get("openai_api_key") or get_api_key()

    spinner_msg = (
        "Running ML analysis (MediaPipe + OpenCV) and OpenAI Vision in parallel..."
        if api_key
        else "Running local ML analysis (MediaPipe + OpenCV)..."
    )
    with st.spinner(spinner_msg):
        observations = analyze_face(image)
        agent_run = run_wellness_crew(image, observations, api_key=api_key)

    report = agent_run.report
    st.session_state["latest_report"] = report
    st.session_state["latest_observations"] = observations
    st.session_state["latest_overlay"] = draw_overlay(image, observations)
    st.session_state["latest_agent_trace"] = agent_run.trace
    st.session_state["latest_modes"] = {
        "openai": agent_run.used_openai,
        "crewai": agent_run.used_crewai,
    }

    if observations.get("face_count") == 1:
        save_scan(report)
        st.success("Analysis complete. Numeric trend metrics were saved; facial image was not stored.")
    else:
        st.warning("Scan validation did not pass, so no trend metric was saved.")
        for issue in observations.get("issues", []):
            st.write(f"- {issue}")


def metric_card(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='muted'>{helper}</div></div>",
        unsafe_allow_html=True,
    )


def dashboard(report: dict, observations: dict) -> None:
    st.subheader("Analysis Dashboard")
    modes = st.session_state.get("latest_modes", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.plotly_chart(gauge("Stress", report.get("stress_score", 0), "#F97316"), use_container_width=True)
    c2.plotly_chart(gauge("Fatigue", report.get("fatigue_score", 0), "#EF4444"), use_container_width=True)
    c3.plotly_chart(gauge("Eye Strain", report.get("eye_strain", 0), "#38BDF8"), use_container_width=True)
    c4.plotly_chart(gauge("Recovery", report.get("recovery_score", 0), "#A855F7"), use_container_width=True)
    c5.plotly_chart(wellness_gauge("Wellness", report.get("wellness_score", 0)), use_container_width=True)

    st.write("")
    a, b, c = st.columns(3)
    a.metric("Recovery Need", report.get("recovery_need", "Unknown"))
    b.metric("Confidence", f"{float(report.get('confidence', 0)):.2f}")
    c.metric("Analysis Mode", report.get("analysis_mode", "Unknown"))

    if not modes.get("openai"):
        ai_status = report.get("ai_status", "OpenAI Vision was not used for this scan.")
        if "error" in ai_status.lower() or "401" in ai_status or "403" in ai_status:
            st.error(f"AI analysis failed — {ai_status}")
            st.caption(
                "Common fixes: (1) Check the key starts with `sk-` and was copied in full. "
                "(2) Switch to **gpt-4o-mini** in the sidebar — it works on free-tier keys. "
                "(3) Verify the key at platform.openai.com → API Keys."
            )
        else:
            st.info(ai_status)
    if modes.get("crewai"):
        st.caption("CrewAI dependency detected; agent role pipeline is available.")
    else:
        st.caption("CrewAI dependency was not importable in this environment; role pipeline fallback was used.")

    if modes.get("openai") and "ml_scores" in report and "ai_scores" in report:
        st.divider()
        st.subheader("ML vs AI Score Breakdown")
        st.caption("Blended scores above use OpenAI Vision (60%) + Local ML (40%) weighting.")
        score_labels = [
            ("stress_score", "Stress"),
            ("fatigue_score", "Fatigue"),
            ("eye_strain", "Eye Strain"),
            ("recovery_score", "Recovery Need"),
            ("wellness_score", "Wellness"),
        ]
        ml_col, ai_col = st.columns(2)
        with ml_col:
            st.markdown("**Local ML — MediaPipe + OpenCV**")
            ml = report["ml_scores"]
            for key, label in score_labels:
                st.metric(label, f"{ml.get(key, 0):.0f} / 100")
        with ai_col:
            st.markdown("**OpenAI Vision — GPT-4o**")
            ai = report["ai_scores"]
            for key, label in score_labels:
                delta = round(float(ai.get(key, 0)) - float(ml.get(key, 0)), 1)
                st.metric(label, f"{ai.get(key, 0):.0f} / 100", delta=f"{delta:+.0f} vs ML")

    left, right = st.columns([1.05, 1])
    with left:
        st.image(st.session_state["latest_overlay"], caption="Facial landmarks, regions, and stress heat overlay", use_column_width=True)
    with right:
        st.plotly_chart(radar_chart(report), use_container_width=True)
        st.plotly_chart(facial_zone_heatmap(observations), use_container_width=True)

    st.subheader("Wellness Summary")
    st.write(report.get("wellness_summary", "No summary available."))

    rec_col, why_col = st.columns(2)
    with rec_col:
        st.markdown("#### Recommendations")
        for item in report.get("recommendations", []):
            st.write(f"- {item}")
    with why_col:
        st.markdown("#### Why These Scores")
        for item in report.get("contributing_factors", []):
            st.write(f"- {item}")

    with st.expander("Limitations and report disclaimer", expanded=True):
        st.write(report.get("disclaimer", DISCLAIMER))
        for item in report.get("limitations", []):
            st.write(f"- {item}")

    with st.expander("CrewAI Agent Trace"):
        trace = st.session_state.get("latest_agent_trace", [])
        st.markdown("".join([f"<span class='agent-pill'>{agent['name']}</span>" for agent in trace]), unsafe_allow_html=True)
        for agent in trace:
            st.write(f"**{agent['name']}**: {agent['role']}")


def scan_page() -> None:
    st.title("Scan")
    uploaded_file = source_selector()
    if uploaded_file is not None:
        run_analysis(uploaded_file)
    if "latest_report" in st.session_state and "latest_observations" in st.session_state:
        dashboard(st.session_state["latest_report"], st.session_state["latest_observations"])


def trends_page() -> None:
    st.title("Trends")
    st.caption("Only numeric metrics and short summaries are stored locally in SQLite.")
    rows = recent_scans(limit=60)
    st.plotly_chart(trend_chart(rows), use_container_width=True)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df[["created_at", "stress_score", "fatigue_score", "eye_strain", "recovery_score", "wellness_score", "confidence", "recovery_need"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No scans have been saved yet.")


def main() -> None:
    inject_css()
    bootstrap_database()
    page = sidebar()
    if page == "Home":
        home_page()
    elif page == "Scan":
        scan_page()
    else:
        trends_page()


if __name__ == "__main__":
    main()
