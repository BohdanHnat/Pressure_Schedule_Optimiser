"""
ui/charts.py — Dual-axis comparison chart.

  Left Y  — Demand forecast (m³/h): grey dashed (SARIMAX) — upper portion
  Right Y — Pressure (bar): blue=planned, green=GA — lower portion

  Legend order: Suggested, Planned, Forecast
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

_BG    = "#0d1117"
_GRID  = "#21262d"
_BLUE  = "#58a6ff"
_GREEN = "#3fb950"
_GREY  = "#8b949e"
_WHITE = "#e6edf3"

_HOURS = [f"{h:02d}:00" for h in range(24)]


def render_comparison_chart(results: dict) -> None:
    Q_forecast  = results["Q_forecast"]
    ga_schedule = results["ga_schedule"]
    planned_p   = results["planned_pressure"]

    fig = go.Figure()

    # TRACE ORDER matches legend order: Suggested first, Planned second, Forecast last

    # 1) GA-optimised pressure — green solid, RIGHT axis
    fig.add_trace(go.Scatter(
        x=_HOURS, y=ga_schedule,
        mode="lines+markers",
        line=dict(color=_GREEN, width=3),
        marker=dict(color=_GREEN, size=8, line=dict(color="#0d1117", width=1)),
        name="Suggested Pressure (GA)",
        hovertemplate="<b>%{x}</b><br>Optimised: %{y:.2f} bar<extra></extra>",
        yaxis="y2",
    ))

    # 2) Planned pressure (operator baseline) — blue solid, RIGHT axis
    fig.add_trace(go.Scatter(
        x=_HOURS, y=planned_p,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(color=_BLUE, size=7, line=dict(color="#0d1117", width=1)),
        name="Planned Pressure (Operator)",
        hovertemplate="<b>%{x}</b><br>Planned: %{y:.2f} bar<extra></extra>",
        yaxis="y2",
    ))

    # 3) Demand forecast (SARIMAX) — grey dashed, LEFT axis
    fig.add_trace(go.Scatter(
        x=_HOURS, y=Q_forecast,
        mode="lines+markers",
        line=dict(color=_GREY, width=2, dash="dash"),
        marker=dict(color=_GREY, size=5),
        name="Demand Forecast (SARIMAX)",
        hovertemplate="<b>%{x}</b><br>Demand: %{y:.0f} m\u00b3/h<extra></extra>",
        yaxis="y1",
    ))

    q_min = float(np.min(Q_forecast))
    q_max = float(np.max(Q_forecast))
    p_min = float(min(planned_p.min(), ga_schedule.min()))
    p_max = float(max(planned_p.max(), ga_schedule.max()))

    # Extend both Y axes so demand line sits ABOVE pressure lines visually.
    # Demand axis: range starts well below min so line appears in upper half.
    # Pressure axis: range extends above max so lines appear in lower half.
    demand_range = [0, 100]
    pressure_range = [p_min - 0.3, p_max + (p_max - p_min) * 2.5]

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_WHITE, family="monospace", size=13),
        height=520,
        margin=dict(l=70, r=80, t=50, b=55),
        hovermode="x unified",
        legend=dict(
            orientation="h", x=0.0, y=1.16,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=15),
        ),
        xaxis=dict(
            gridcolor=_GRID, showgrid=True,
            tickfont=dict(size=12), title="Hour",
            title_font=dict(size=14),
            tickangle=-45,
        ),
        yaxis=dict(
            title="Demand (m\u00b3/h)",
            gridcolor=_GRID, showgrid=True,
            range=demand_range,
            tickfont=dict(size=12),
            title_font=dict(color=_GREY, size=14),
            tickfont_color=_GREY,
        ),
        yaxis2=dict(
            title="Pressure (bar)",
            overlaying="y", side="right",
            range=pressure_range,
            showgrid=False,
            tickfont=dict(size=12),
            title_font=dict(color=_GREEN, size=14),
            tickfont_color=_GREEN,
        ),
    )

    st.plotly_chart(fig, width='stretch', config={
        "modeBarButtonsToRemove": [
            "toImage", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "zoom2d",
            "resetScale2d",
        ],
        "displayModeBar": True,
    })
