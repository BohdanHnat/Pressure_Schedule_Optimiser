"""
app.py — Municipal PS Operator Dashboard
=========================================
Entry point. Run with:

    streamlit run app.py

Then open http://localhost:8501 in a browser.

Prerequisite:
  Run prepare_window.py once (offline) to generate:
    data_files/sarimax_historical_window.csv

Three dashboard views:
  View 1 — Data Input: demand + planned pressure side-by-side
  View 2 — Optimisation Progress: step list + progress bar
  View 3 — Results: comparison chart + pressure map + metrics

NOTE on Streamlit's execution model:
  Streamlit re-executes this entire script from line 1 on every user interaction
  (button click, file upload, slider move). Ordinary Python variables do not
  survive between interactions — persistent state must be stored in st.session_state.
  This is not a design choice; it is a fundamental constraint of the framework.
  Comments marked "NOTE: Streamlit" throughout this file identify patterns that
  exist because of this constraint and cannot be simplified further.
"""

import os, sys
import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="Municipal PS — Pressure Optimiser",
    page_icon="\U0001f4a7",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.data_input import render_demand_section, render_pressure_section
from core.pipeline import Pipeline
from ui.charts import render_comparison_chart
from ui.pressure_map import render_pressure_map


# ── CSS ───────────────────────────────────────────────────────────────────────

def _load_css():
    """Load dashboard CSS from file."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "static", "styles", "dashboard.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()


# ── Fixed station header (shown on all views) ─────────────────────────────────

def _render_fixed_header():
    """Render the fixed station title banner at the top of the page."""
    st.markdown(
        "<div class='fixed-station-header'>"
        "<div class='station-header'>"
        "\U0001f4a7 Municipal Water Pumping Station — Pressure Schedule Optimiser"
        "</div>"
        "<div class='station-subtitle'>"
        "SARIMAX demand forecasting + Genetic Algorithm "
        "pressure schedule optimisation"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Session state initialisation ──────────────────────────────────────────────

# NOTE: Streamlit — session_state is the only persistence mechanism across reruns.
# This block initialises keys with defaults on the very first run of the session.
_SS_DEFAULTS = {
    "view":             "input",   # "input" | "progress" | "results"
    "last_day_demand":  None,      # pd.DataFrame (24 rows)
    "planned_pressure": None,      # pd.DataFrame (24 rows)
    "results":          None,      # PipelineResult from pipeline
    "opt_complete":     False,     # True after optimisation finishes
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
#   NAVIGATION TOP-BAR
# ═════════════════════════════════════════════════════════════════════════════

def _render_topbar(current_view: str, csv_bytes: bytes = None,
                   input_ready: bool = False):
    """
    Render the stepper navigation bar.

    Step states and what they mean:
      "current"      — this is the active view (button disabled)
      "filled"       — step is complete; clicking navigates without resetting
      "filled_locked"— step is complete but navigation is blocked (computing)
      "ready"        — input is ready; clicking starts a new optimisation run
      "empty"        — step is not yet reachable (button disabled)

    NOTE: Streamlit — the CSS class wrappers (<div class='tb-filled'> etc.) apply
    colour coding to the buttons via the dashboard.css file. Streamlit does not
    expose a native API for styling individual buttons differently, so this HTML
    injection is the standard Streamlit workaround for per-button styling.
    """
    has_results = st.session_state.results is not None
    opt_done    = st.session_state.opt_complete

    if current_view == "input":
        s1 = "current"
        input_changed = st.session_state.get("_input_changed", False)
        if opt_done and has_results and not input_changed:
            s2 = "filled"    # same data, already optimised
        elif input_ready:
            s2 = "ready"     # new data ready — clicking triggers a new run
        else:
            s2 = "empty"
        s3 = "filled" if (opt_done and has_results and not input_changed) else "empty"
    elif current_view == "progress":
        s1 = "filled" if opt_done else "filled_locked"
        s2 = "current"
        s3 = "filled" if opt_done and has_results else "empty"
    else:  # results
        s1, s2, s3 = "filled", "filled", "current"

    steps = [
        (1, "Data Input",   s1, "input"),
        (2, "Optimisation", s2, "progress"),
        (3, "Results",      s3, "results"),
    ]

    # 7-column layout: spacer | step1 | arrow | step2 | arrow | step3 | spacer
    c = st.columns([3.2, 1.1, 0.15, 1.2, 0.15, 0.9, 3.2])
    btn_cols = [c[1], c[3], c[5]]
    arr_cols = [c[2], c[4]]

    for ac in arr_cols:
        with ac:
            st.markdown("<div class='stepper-arrow-inline'>▶</div>",
                        unsafe_allow_html=True)

    css_map = {
        "filled":       "tb-filled",
        "filled_locked":"tb-filled-locked",
        "current":      "tb-current",
        "empty":        "tb-empty",
        "ready":        "tb-filled",
    }

    for idx, (num, label, state, target) in enumerate(steps):
        with btn_cols[idx]:
            clickable = state in ("filled", "ready")
            check     = "✓ " if state in ("filled", "filled_locked") else ""
            btn_text  = f"{check}{num}. {label}"
            st.markdown(f"<div class='{css_map[state]}'>", unsafe_allow_html=True)
            if clickable:
                if st.button(btn_text, key=f"tb_{num}", use_container_width=True):
                    if state == "ready":
                        # Save pending input data, reset opt state for fresh run
                        d = st.session_state.get("_pending_demand")
                        p = st.session_state.get("_pending_pressure")
                        st.session_state.last_day_demand  = d
                        st.session_state.planned_pressure = p
                        st.session_state["_input_changed"] = False
                        st.session_state.opt_complete      = False
                        st.session_state.results           = None
                        st.session_state.pop("_progress_armed", None)
                    st.session_state.view = target
                    st.rerun()
            else:
                st.button(btn_text, key=f"tb_{num}",
                          use_container_width=True, disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Download button — results page only; CSS positions it flush-right
    if csv_bytes is not None:
        st.markdown("<div class='dl-btn-absolute-wrap'>", unsafe_allow_html=True)
        st.download_button(
            "⭳ Download Pressure Schedule (CSV)",
            data=csv_bytes,
            file_name="suggested_outlet_pressure_schedule.csv",
            mime="text/csv",
            key="dl_csv_btn",
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#   VIEW 1 — DATA INPUT
# ═════════════════════════════════════════════════════════════════════════════

def render_input_view():
    """
    Data input view: demand section (left) and pressure section (right).
    The Step 2 topbar button becomes clickable when both inputs are ready.
    """
    # Use previous-run input_ready for the topbar to avoid a one-frame flicker
    # NOTE: Streamlit — the topbar renders before widget values are read this run,
    # so we store the ready-state in session_state and use it one run later.
    prev_ready = st.session_state.get("_input_ready", False)
    _render_topbar("input", input_ready=prev_ready)

    col_demand, col_spacer, col_pressure = st.columns([5, 0.3, 5])
    with col_demand:
        demand_df, demand_ok = render_demand_section()
    with col_pressure:
        pressure_df, pressure_ok = render_pressure_section()

    both_ready = demand_ok and pressure_ok

    if both_ready:
        st.session_state["_pending_demand"]   = demand_df
        st.session_state["_pending_pressure"] = pressure_df

    # Detect whether the current input differs from the data used for the last run.
    # df.equals() is a direct pandas comparison — no hashing needed.
    input_changed = False
    if both_ready and st.session_state.opt_complete:
        d = st.session_state.last_day_demand
        p = st.session_state.planned_pressure
        input_changed = not (demand_df.equals(d) and pressure_df.equals(p))

    prev_changed = st.session_state.get("_input_changed", False)
    st.session_state["_input_changed"] = input_changed
    st.session_state["_input_ready"]   = both_ready

    # NOTE: Streamlit — trigger a rerun when ready/changed state flips so the
    # topbar updates its button states immediately, not one interaction later.
    if both_ready != prev_ready or input_changed != prev_changed:
        st.rerun()

    # Status notice at the bottom
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    col_btn_l, col_btn_c, col_btn_r = st.columns([1.5, 5, 1.5])
    with col_btn_c:
        if both_ready:
            st.markdown(
                "<div class='input-ready'>"
                "Both sections have data — optimisation is ready to begin."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='input-warning'>"
                "Complete both sections to proceed: provide all 24 hourly values "
                "in each table, either by uploading a CSV or entering manually."
                "</div>",
                unsafe_allow_html=True,
            )

# ═════════════════════════════════════════════════════════════════════════════
#   VIEW 2 — OPTIMISATION PROGRESS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStep:
    """Defines one entry in the progress display: icon, title, and description."""
    icon: str
    title: str
    description: str  # HTML string with <ul class='step-bullets'> bullet list


_STEPS = [
    PipelineStep(
        icon="\U0001f4e5",
        title="Reading Operator Input",
        description=(
            "<ul class='step-bullets'>"
            "<li>Processing yesterday's hourly water consumption readings "
            "(24 values, m³/h)</li>"
            "<li>Processing today's planned pressure schedule "
            "(24 hourly setpoints, bar)</li>"
            "</ul>"
        ),
    ),
    PipelineStep(
        icon="\U0001f4ca",
        title="Updating Demand Forecast Database",
        description=(
            "<ul class='step-bullets'>"
            "<li>Appending last 24-hour consumption data to the historical "
            "demand records</li>"
            "<li>Removing oldest day entry to preserve a fixed 6-month "
            "historical dataset</li>"
            "</ul>"
        ),
    ),
    PipelineStep(
        icon="\U0001f4c8",
        title="Retraining Demand Forecast Model",
        description=(
            "<ul class='step-bullets'>"
            "<li>Updating the forecasting model with last day's consumption data</li>"
            "<li>Applying weekly demand cycle patterns via Fourier variables "
            "(7-day cycle)</li>"
            "<li>Applying Model Parameters "
            "(SARIMAX; p:1, d:0, q:1; P:1, D:1, Q:1, m:24)</li>"
            "<li>Generating predicted hourly water consumption for today</li>"
            "</ul>"
        ),
    ),
    PipelineStep(
        icon="\U0001f4a7",
        title="Running Hydraulic Model Simulation",
        description=(
            "<ul class='step-bullets'>"
            "<li>Launching the hydraulic model of the pumping station and "
            "distribution network (EPANET)</li>"
            "<li>Loading today's forecasted demand data and simulating water "
            "consumption across consumers</li>"
            "<li>Calculating actual pressures at consumer endpoints accounting "
            "for network pressure losses (Hazen-Williams head loss formula)</li>"
            "</ul>"
        ),
    ),
    PipelineStep(
        icon="\U0001f50d",
        title="Energy Consumption Baseline Determination",
        description=(
            "<ul class='step-bullets'>"
            "<li>Converting operator's planned pressure setpoints to relative "
            "pump speeds and loading into the hydraulic model</li>"
            "<li>Simulating today's pumping station operation under forecasted demand</li>"
            "<li>Calculating total pump energy consumption under the planned pressure "
            "schedule and predicted demand to establish a baseline for comparison</li>"
            "</ul>"
        ),
    ),
    PipelineStep(
        icon="\U0001f9ec",
        title="Optimised Energy Consumption Determination",
        description=(
            "<ul class='step-bullets'>"
            "<li>Launching Artificial Intelligence Pressure Optimisation Model "
            "(Genetic Algorithm)</li>"
            "<li>Generating 80 alternative pressure schedules across 100 improvement "
            "rounds (100 generations &times; 80 candidates per generation)</li>"
            "<li>Scoring each candidate schedule on three criteria:"
            "<ul style='list-style-type: none; padding-left: 1.5em; margin-top: 0.4em;'>"
            "<li style='padding: 0.15em 0;'>&#8212; Pressure safety: all consumer nodes "
            "within regulation limits 2.5&ndash;6.0 bar</li>"
            "<li style='padding: 0.15em 0;'>&#8212; Energy consumption: minimising total "
            "pump power use</li>"
            "<li style='padding: 0.15em 0;'>&#8212; Pump efficiency: operating near Best "
            "Efficiency Point (BEP)</li>"
            "</ul></li>"
            "<li>Selecting the best-performing schedule across all three criteria</li>"
            "<li>Calculating total pump energy consumption under the optimised pressure "
            "schedule via hydraulic simulation</li>"
            "</ul>"
        ),
    ),
    PipelineStep(
        icon="\U0001f4cb",
        title="Preparing Results",
        description=(
            "<ul class='step-bullets'>"
            "<li>Calculating energy savings against the baseline energy consumption "
            "(kWh and %)</li>"
            "<li>Generating comparison chart and hydraulic network pressure "
            "distribution map</li>"
            "</ul>"
        ),
    ),
]


def _render_step_card(idx: int, step: PipelineStep, status: str, detail: str = ""):
    """
    Render a single step card in the progress list.
    status: "pending" | "running" | "done"
    """
    num = idx + 1
    indicators = {
        "done":    '<div class="step-indicator step-indicator-done">✓</div>',
        "running": '<div class="step-indicator step-indicator-running"></div>',
        "pending": '<div class="step-indicator step-indicator-pending">○</div>',
    }
    detail_html = f'<div class="step-detail">{detail}</div>' if detail else ""
    st.markdown(
        f'<div class="progress-step progress-step-{status}">'
        f'<div class="progress-step-icon">{step.icon}</div>'
        f'<div class="progress-step-body">'
        f'<div class="progress-step-title">{num}. {step.title}</div>'
        f'<div class="progress-step-desc">{step.description}</div>'
        f'{detail_html}'
        f'</div>'
        f'{indicators[status]}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_progress_completed():
    """Show the progress page in a fully completed, frozen (read-only) state."""
    _render_topbar("progress")

    st.markdown(
        "<div class='progress-header'>OPTIMISATION COMPLETE</div>",
        unsafe_allow_html=True,
    )

    for i, step in enumerate(_STEPS):
        _render_step_card(i, step, "done", "")

    st.progress(1.0)
    st.markdown(
        "<div style='text-align:center; font-family:monospace; "
        "color:#3fb950; font-size:14px;'>100%</div>",
        unsafe_allow_html=True,
    )


def render_progress_view(topbar_already=False):
    """
    Optimisation progress view: step list with live status updates and progress bar.
    Uses a clean page (DOM flush guard in main()) before the blocking pipeline call.
    """
    # If optimisation is already complete (e.g. navigated back from results),
    # show the frozen completed state — do not re-run the pipeline.
    if st.session_state.opt_complete and st.session_state.results is not None:
        _render_progress_completed()
        return

    # Hide Streamlit footer on the progress page
    st.markdown(
        """<style>
        footer, [data-testid="stBottom"] { display: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )

    if not topbar_already:
        _render_topbar("progress")

    st.markdown(
        "<div class='progress-header'>COMPUTING · PLEASE WAIT</div>",
        unsafe_allow_html=True,
    )

    # Validate required input and files before running
    if st.session_state.last_day_demand is None or st.session_state.planned_pressure is None:
        st.error("Input data missing. Please return to the input screen.")
        if st.button("Back to Input"):
            st.session_state.view = "input"
            st.rerun()
        return

    missing = [os.path.basename(f) for f in [C.NETWORK_FILE, C.HISTORICAL_WINDOW_FILE]
               if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        st.info("Run prepare_window.py first to generate required data files.")
        return

    # NOTE: Streamlit — st.empty() creates a single-slot container that can be
    # updated in place. Each step card occupies one container so _refresh_steps()
    # can rewrite individual cards without re-rendering the whole page.
    # This is the only way Streamlit supports live UI updates from a blocking function.
    step_containers = [st.empty() for _ in range(len(_STEPS))]
    progress_bar    = st.progress(0.0)
    pct_text        = st.empty()
    status_text     = st.empty()

    step_states  = ["pending"] * len(_STEPS)
    step_details = [""]        * len(_STEPS)

    def _refresh_steps():
        """Redraw all step cards with their current states."""
        for i, step in enumerate(_STEPS):
            with step_containers[i]:
                _render_step_card(i, step, step_states[i], step_details[i])

    def step_callback(idx, status_val):
        """Called by Pipeline.run() after each stage to update the step list."""
        step_states[idx] = status_val
        _refresh_steps()

    _refresh_steps()

    try:
        pipeline = Pipeline(
            demand_df=st.session_state.last_day_demand,
            pressure_df=st.session_state.planned_pressure,
        )
        results = pipeline.run(
            progress_bar=progress_bar,
            status_text=status_text,
            step_callback=step_callback,
        )

        # Populate per-step detail annotations
        meta = results.meta
        step_details[0] = (
            f"{meta.get('demand_rows', 24)} demand + "
            f"{meta.get('pressure_rows', 24)} pressure rows ingested"
        )
        step_details[1] = f"Training dataset: {meta.get('window_rows', 'N/A')} entries"
        step_details[2] = (
            f"SARIMAX {meta.get('sarimax_order', '')} × "
            f"{meta.get('sarimax_seasonal', '')} · "
            f"AIC = {meta.get('sarimax_aic', 'N/A')} · "
            f"Mean forecast = {meta.get('forecast_mean', 'N/A')} m³/h"
        )
        step_details[3] = (
            f"{meta.get('n_junctions', '?')} junctions · "
            f"{meta.get('n_pumps', '?')} pumps · "
            f"{meta.get('n_pipes', '?')} pipes"
        )
        step_details[4] = f"Baseline energy: {meta.get('E_planned', '?')} kWh"
        step_details[5] = f"{C.POP_SIZE} candidates × {C.N_GEN} generations"
        step_details[6] = (
            f"Baseline: {meta.get('E_planned', '?')} kWh · "
            f"Optimised: {meta.get('E_ga', '?')} kWh · "
            f"Saving: {meta.get('saving_pct', '?')}%"
        )
        _refresh_steps()

        st.session_state.results      = results
        st.session_state.opt_complete = True
        progress_bar.progress(1.0)
        pct_text.markdown(
            "<div style='text-align:center; font-family:monospace; "
            "color:#3fb950; font-size:14px;'>100%</div>",
            unsafe_allow_html=True,
        )
        status_text.markdown(
            "<div class='progress-header'>OPTIMISATION COMPLETE</div>",
            unsafe_allow_html=True,
        )

        import time
        time.sleep(0.5)
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        st.error(f"Optimisation error: {e}")
        st.exception(e)
        if st.button("Back to Input"):
            st.session_state.view = "input"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#   VIEW 3 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def render_results_view():
    """
    Results view: topbar, comparison chart + side metrics, hourly pressure table,
    network pressure map, and footer.
    """
    results = st.session_state.results
    if results is None:
        st.warning("No results available.")
        if st.button("Back to Input"):
            st.session_state.view = "input"
            st.rerun()
        return

    saving_kwh = results.saving_kwh
    saving_pct = results.saving_pct
    saving_uah = results.saving_uah
    E_planned  = results.E_planned_kWh
    E_ga       = results.E_ga_kWh

    # Prepare CSV download data
    import io
    buf = io.BytesIO()
    pd.DataFrame({
        "Time":         [f"{h:02d}:00" for h in range(C.N_STEPS)],
        "P_target_bar": np.round(results.ga_schedule, 1),
    }).to_csv(buf, index=False)
    csv_bytes = buf.getvalue()

    _render_topbar("results", csv_bytes=csv_bytes)

    # Comparison chart (90%) + side metrics (10%)
    col_chart, col_metrics = st.columns([9.5, 0.7])
    with col_chart:
        render_comparison_chart(results)
    with col_metrics:
        st.markdown(
            f"""
            <div class="side-metrics">
                <div class="side-metric-card">
                    <div class="side-metric-label">Energy Saving</div>
                    <div class="side-metric-value positive">{saving_pct:.1f}%</div>
                </div>
                <div class="side-metric-card">
                    <div class="side-metric-label">Baseline (Planned)</div>
                    <div class="side-metric-value">{E_planned:.1f} kWh</div>
                </div>
                <div class="side-metric-card">
                    <div class="side-metric-label">Optimised (GA)</div>
                    <div class="side-metric-value">{E_ga:.1f} kWh</div>
                </div>
                <div class="side-metric-card">
                    <div class="side-metric-label">Daily Saving</div>
                    <div class="side-metric-value positive">{saving_kwh:.1f} kWh</div>
                </div>
                <div class="side-metric-card">
                    <div class="side-metric-label">Daily Saving</div>
                    <div class="side-metric-value positive">{saving_uah:,.0f} UAH</div>
                </div>
                <div class="side-metric-card">
                    <div class="side-metric-label">Tariff</div>
                    <div class="side-metric-value">{C.TARIFF_UAH_PER_KWH} UAH/kWh</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_hourly_pressure_table(results)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='map-section-header'>Pressure Distribution Map</div>",
        unsafe_allow_html=True,
    )
    if results.wn is not None and results.node_pressures_ga:
        render_pressure_map(results)
    else:
        st.info("Pressure map data unavailable — check EPANET model file.")

    st.markdown(
        "<div class='page-footer'>"
        "Designed and Developed by Bohdan Hnat Studio, 2026"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_hourly_pressure_table(results):
    """
    Display a horizontal 2-row × 24-column comparison table: planned vs optimised pressure.
    The optimised row is highlighted green via pandas Styler.

    NOTE: Streamlit — st.dataframe() accepts a pandas Styler object, which allows
    per-row CSS colour rules. This is simpler than building a raw HTML string but still
    requires pandas styling rather than plain Python, because Streamlit's native
    st.dataframe() has no row-colour API of its own.
    """
    hours   = [f"{h:02d}:00" for h in range(24)]
    planned = np.round(results.planned_pressure, 2)
    ga      = np.round(results.ga_schedule, 2)

    # Transpose so hours become columns and the two schedules become rows
    df = pd.DataFrame(
        {"Planned (bar)": planned, "Optimised (bar)": ga},
        index=hours,
    ).T

    def _highlight_ga(row):
        """Apply green colour to the Optimised row only."""
        if row.name == "Optimised (bar)":
            return ["color: #3fb950; background-color: rgba(63,185,80,0.08); "
                    "font-weight: bold"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_ga, axis=1).format("{:.2f}")

    # Set a narrow fixed width on every hour column so all 24 fit without scrolling.
    # The row-label column (index) gets a slightly wider slot for "Optimised (bar)".
    col_cfg = {h: st.column_config.NumberColumn(width=68) for h in hours}
    st.dataframe(styled, use_container_width=True, column_config=col_cfg)


# ═════════════════════════════════════════════════════════════════════════════
#   MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Route to the correct view based on session state."""
    view = st.session_state.view

    # Fixed header shown on all views
    _render_fixed_header()

    # DOM flush guard for the progress view.
    # NOTE: Streamlit — when navigating to "progress" for a new run, one empty rerun
    # is needed to flush the previous page's widgets from the browser DOM before the
    # blocking Pipeline.run() call begins. Without this flush, the old DOM (input
    # page buttons) remains visible while the pipeline computes.
    # The topbar is rendered above the guard so the browser has a minimal valid
    # script output to replace the old DOM during the flush rerun.
    if view == "progress" and not st.session_state.opt_complete:
        _render_topbar("progress")
        if not st.session_state.get("_progress_armed", False):
            st.session_state["_progress_armed"] = True
            st.rerun()
            return  # unreachable; satisfies linters

    # Clear the arming flag when leaving the progress view
    if view != "progress":
        st.session_state.pop("_progress_armed", None)

    if   view == "input":    render_input_view()
    elif view == "progress": render_progress_view(topbar_already=True)
    elif view == "results":  render_results_view()
    else:
        st.session_state.view = "input"
        st.rerun()


if __name__ == "__main__":
    main()
