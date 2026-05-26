from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def gauge(title: str, value: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": "/100", "font": {"color": "#FFFFFF"}},
            title={"text": title, "font": {"color": "#FFFFFF", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#CBD5E1"},
                "bar": {"color": color},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 35], "color": "#064E3B"},
                    {"range": [35, 70], "color": "#713F12"},
                    {"range": [70, 100], "color": "#7F1D1D"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=18, r=18, t=48, b=12), paper_bgcolor="#0F172A")
    return fig


def wellness_gauge(title: str, value: float) -> go.Figure:
    fig = gauge(title, value, "#22C55E")
    fig.data[0].gauge.steps = [
        {"range": [0, 35], "color": "#7F1D1D"},
        {"range": [35, 70], "color": "#713F12"},
        {"range": [70, 100], "color": "#064E3B"},
    ]
    return fig


def radar_chart(report: dict) -> go.Figure:
    categories = ["Stress", "Fatigue", "Eye Strain", "Recovery Need", "Wellness"]
    values = [
        report.get("stress_score", 0),
        report.get("fatigue_score", 0),
        report.get("eye_strain", 0),
        report.get("recovery_score", 55),
        report.get("wellness_score", 50),
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Current scan",
            line_color="#38BDF8",
        )
    )
    fig.update_layout(
        height=360,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font_color="#FFFFFF",
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"),
            angularaxis=dict(gridcolor="#334155"),
        ),
        margin=dict(l=24, r=24, t=36, b=24),
    )
    return fig


def trend_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            title="No trends yet",
            height=320,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font_color="#FFFFFF",
        )
        return fig
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    fig = px.line(
        df,
        x="created_at",
        y=["stress_score", "fatigue_score", "recovery_score", "wellness_score"],
        markers=True,
        labels={"value": "Score", "created_at": "Scan time", "variable": "Metric"},
    )
    fig.update_layout(
        height=360,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font_color="#FFFFFF",
        xaxis_gridcolor="#334155",
        yaxis_gridcolor="#334155",
        yaxis_range=[0, 100],
        margin=dict(l=24, r=24, t=36, b=24),
    )
    return fig


def facial_zone_heatmap(observations: dict) -> go.Figure:
    zones = observations.get("zone_scores", {})
    labels = list(zones.keys()) or ["eyes", "under_eyes", "forehead", "mouth_jaw"]
    values = [zones.get(label, 0) for label in labels]
    fig = go.Figure(data=go.Heatmap(z=[values], x=labels, y=["Facial zones"], colorscale="Turbo", zmin=0, zmax=100))
    fig.update_layout(
        height=180,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font_color="#FFFFFF",
        margin=dict(l=24, r=24, t=20, b=24),
    )
    return fig
