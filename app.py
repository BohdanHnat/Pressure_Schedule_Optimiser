"""
app.py — НС «Вітрука» Operator Dashboard
=========================================
Точка входу. Запуск:

    streamlit run app.py

Потім відкрити http://localhost:8501 у браузері.

Передумова розгортання:
  Запустити prepare_window.py один раз (офлайн) для створення:
    data_files/sarimax_historical_window.csv

Три відображення (views) дашборду:
  View 1 — Введення даних: попит + плановий тиск (side-by-side)
  View 2 — Прогрес оптимізації: список етапів + прогрес-бар
  View 3 — Результати: графік порівняння + карта тиску + метрики
"""

import os, sys
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# ── Конфігурація сторінки — повинна бути першим викликом Streamlit ────────────
st.set_page_config(
    page_title="Municipal PS \u2014 Pressure Optimiser",
    page_icon="\U0001f4a7",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Імпорти (після налаштування шляхів) ──────────────────────────────────────
from utils.data_input import render_demand_section, render_pressure_section
from core.pipeline import run_full_pipeline
from ui.charts import render_comparison_chart
from ui.pressure_map import render_pressure_map


# ── CSS ──────────────────────────────────────────────────────────────────────
def _load_css():
    """Завантажити CSS стилі з файлу."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "static", "styles", "dashboard.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()


# ── Фіксований заголовок станції (показується на обох сторінках) ─────────────
def _render_fixed_header():
    """Відобразити фіксований заголовок станції зверху сторінки."""
    st.markdown(
        "<div class='fixed-station-header'>"
        "<div class='station-header'>"
        "\U0001f4a7 Municipal Water Pumping Station \u2014 Pressure Schedule Optimiser"
        "</div>"
        "<div class='station-subtitle'>"
        "SARIMAX demand forecasting + Genetic Algorithm "
        "pressure schedule optimisation"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Ініціалізація стану сесії ────────────────────────────────────────────────
_SS_DEFAULTS = {
    "view":              "input",    # "input" | "progress" | "results"
    "last_day_demand":   None,       # pd.DataFrame (24 рядки)
    "planned_pressure":  None,       # pd.DataFrame (24 рядки)
    "results":           None,       # dict з pipeline
    "opt_complete":      False,      # True after optimisation finishes
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
#   SHARED TOP-BAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════

def _render_topbar(current_view: str, csv_bytes: bytes = None, input_ready: bool = False):
    """
    Render stepper navigation bar.
    Pages 1-2: centred 3 step-buttons with HTML arrows.
    Page 3: 3 centred step-buttons + download button flush-right, all on one line.
    """
    has_results = st.session_state.results is not None
    opt_done = st.session_state.opt_complete

    if current_view == "input":
        s1 = "current"
        # Check if input data changed since last optimisation
        input_changed = st.session_state.get("_input_changed", False)
        if opt_done and has_results and not input_changed:
            # Same data, already optimised — step 2 navigates without reset
            s2 = "filled"
        elif input_ready:
            # New/changed input ready — step 2 triggers new optimisation
            s2 = "ready"
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

    # Columns: spacer | step1 | arrow | step2 | arrow | step3 | spacer
    # CSS zeros the gap on this specific horizontal block via :has([class*="tb-"])
    c = st.columns([3.2, 1.1, 0.15, 1.2, 0.15, 0.9, 3.2])
    btn_cols = [c[1], c[3], c[5]]
    arr_cols = [c[2], c[4]]

    for ac in arr_cols:
        with ac:
            st.markdown("<div class='stepper-arrow-inline'>\u25B6</div>",
                        unsafe_allow_html=True)

    for idx, (num, label, state, target) in enumerate(steps):
        with btn_cols[idx]:
            clickable = state in ("filled", "ready")
            check = "\u2713 " if state in ("filled", "filled_locked") else ""
            btn_text = f"{check}{num}. {label}"
            css_map = {"filled": "tb-filled", "filled_locked": "tb-filled-locked",
                       "current": "tb-current", "empty": "tb-empty",
                       "ready": "tb-filled"}
            css_cls = css_map[state]
            st.markdown(f"<div class='{css_cls}'>", unsafe_allow_html=True)
            if clickable:
                if st.button(btn_text, key=f"tb_{num}", use_container_width=True):
                    if state == "ready":
                        # Proceed to optimisation — save input data + fingerprint
                        import hashlib
                        def _fp(df):
                            return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
                        d = st.session_state.get("_pending_demand")
                        p = st.session_state.get("_pending_pressure")
                        st.session_state.last_day_demand = d
                        st.session_state.planned_pressure = p
                        st.session_state["_opt_data_fingerprint"] = _fp(d) + _fp(p)
                        st.session_state["_input_changed"] = False
                        st.session_state.opt_complete = False
                        st.session_state.results = None
                        st.session_state.pop("_progress_armed", None)
                    st.session_state.view = target
                    st.session_state["_scroll_top"] = True
                    st.rerun()
            else:
                st.button(btn_text, key=f"tb_{num}", use_container_width=True, disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Download button — results page only, rendered separately, CSS positions it
    if csv_bytes is not None:
        st.markdown("<div class='dl-btn-absolute-wrap'>", unsafe_allow_html=True)
        st.download_button(
            "\u2B73 Download Pressure Schedule (CSV)",
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
    Відображення введення даних: дві секції поруч (попит + тиск).
    Step 2 button in topbar becomes clickable when both inputs are ready.
    """
    # Use previous-run input_ready state for topbar (avoids flicker on data edit)
    prev_ready = st.session_state.get("_input_ready", False)
    _render_topbar("input", input_ready=prev_ready)

    # Дві колонки: попит (ліва) та тиск (права)
    col_demand, col_spacer, col_pressure = st.columns([5, 0.3, 5])

    with col_demand:
        demand_df, demand_ok = render_demand_section()

    with col_pressure:
        pressure_df, pressure_ok = render_pressure_section()

    both_ready = demand_ok and pressure_ok

    # Store input state for topbar on next rerun + for proceed logic
    if both_ready:
        st.session_state["_pending_demand"] = demand_df
        st.session_state["_pending_pressure"] = pressure_df

    # Detect if input data changed since last optimisation
    input_changed = False
    if both_ready and st.session_state.opt_complete:
        import hashlib
        def _fingerprint(df):
            return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
        curr_fp = _fingerprint(demand_df) + _fingerprint(pressure_df)
        prev_fp = st.session_state.get("_opt_data_fingerprint", "")
        input_changed = curr_fp != prev_fp

    prev_changed = st.session_state.get("_input_changed", False)
    st.session_state["_input_changed"] = input_changed
    st.session_state["_input_ready"] = both_ready

    # If ready-state or changed-state flipped, rerun so topbar updates
    if both_ready != prev_ready or input_changed != prev_changed:
        st.rerun()

    # Status notice at the bottom
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    col_btn_l, col_btn_c, col_btn_r = st.columns([1.5, 5, 1.5])
    with col_btn_c:
        if both_ready:
            st.markdown(
                "<div class='input-ready'>"
                "Both sections have data \u2014 optimisation is ready to begin."
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

# Визначення етапів для прогрес-листа
_STEPS = [
    {
        "icon": "\U0001f4e5",
        "title": "Reading Operator Input",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Processing yesterday's hourly water consumption readings (24 values, m³/h)</li>"
            "<li>Processing today's planned pressure schedule (24 hourly setpoints, bar)</li>"
            "</ul>"
        ),
    },
    {
        "icon": "\U0001f4ca",
        "title": "Updating Demand Forecast Database",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Appending last 24-hour consumption data to the historical demand records</li>"
            "<li>Removing oldest day entry to preserve a fixed 6-month historical dataset</li>"
            "</ul>"
        ),
    },
    {
        "icon": "\U0001f4c8",
        "title": "Retraining Demand Forecast Model",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Updating the forecasting model with last day's consumption data</li>"
            "<li>Applying weekly demand cycle patterns via Fourier variables (7-day cycle)</li>"
            "<li>Applying Model Parameters (SARIMAX; p:1, d:0, q:1; P:1, D:1, Q:1, m:24)</li>"
            "<li>Generating predicted hourly water consumption for today</li>"
            "</ul>"
        ),
    },
    {
        "icon": "\U0001f4a7",
        "title": "Running Hydraulic Model Simulation",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Launching the hydraulic model of the pumping station and distribution network (EPANET)</li>"
            "<li>Loading today's forecasted demand data and simulating water consumption across consumers</li>"
            "<li>Calculating actual pressures at consumer endpoints accounting for network pressure losses (Hazen-Williams head loss formula)</li>"
            "</ul>"
        ),
    },
    {
        "icon": "\U0001f50d",
        "title": "Energy Consumption Baseline Determination",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Converting operator's planned pressure setpoints to relative pump speeds and loading into the hydraulic model</li>"
            "<li>Simulating today's pumping station operation under forecasted demand</li>"
            "<li>Calculating total pump energy consumption under the planned pressure schedule and predicted demand to establish a baseline for comparison</li>"
            "</ul>"
        ),
    },
    {
        "icon": "\U0001f9ec",
        "title": "Optimised Energy Consumption Determination",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Launching Artificial Intelligence Pressure Optimisation Model (Genetic Algorithm)</li>"
            "<li>Generating 80 alternative pressure schedules across 100 improvement rounds (100 generations &times; 80 candidates per generation)</li>"
            "<li>Scoring each candidate schedule on three criteria:"
            "<ul style='list-style-type: none; padding-left: 1.5em; margin-top: 0.4em;'>"
            "<li style='padding: 0.15em 0;'>&#8212; Pressure safety: all consumer nodes within regulation limits 2.5&ndash;6.0 bar</li>"
            "<li style='padding: 0.15em 0;'>&#8212; Energy consumption: minimising total pump power use</li>"
            "<li style='padding: 0.15em 0;'>&#8212; Pump efficiency: operating near Best Efficiency Point (BEP)</li>"
            "</ul></li>"
            "<li>Selecting the best-performing schedule across all three criteria</li>"
            "<li>Calculating total pump energy consumption under the optimised pressure schedule via hydraulic simulation</li>"
            "</ul>"
        ),
    },
    {
        "icon": "\U0001f4cb",
        "title": "Preparing Results",
        "desc": (
            "<ul class='step-bullets'>"
            "<li>Calculating energy savings against the baseline energy consumption (kWh and %)</li>"
            "<li>Generating comparison chart and hydraulic network pressure distribution map</li>"
            "</ul>"
        ),
    },
]

def _render_step_card(idx: int, step: dict, status: str, detail: str = ""):
    """
    Відобразити одну картку етапу у прогрес-листі.
    status: "pending" | "running" | "done"
    """
    num = idx + 1
    status_color = {"pending": "#484f58", "running": "#d29922", "done": "#3fb950"}[status]
    step_cls = f"progress-step progress-step-{status}"

    # Dynamic status indicator
    if status == "done":
        indicator = (
            '<div class="step-indicator step-indicator-done">'
            '\u2713</div>'
        )
    elif status == "running":
        indicator = '<div class="step-indicator step-indicator-running"></div>'
    else:
        indicator = (
            '<div class="step-indicator step-indicator-pending">'
            '\u25CB</div>'
        )

    detail_html = ""
    if detail:
        detail_html = f'<div class="step-detail">{detail}</div>'

    html = (
        f'<div class="{step_cls}">'
        f'<div class="progress-step-icon">{step["icon"]}</div>'
        f'<div class="progress-step-body">'
        f'<div class="progress-step-title">{num}. {step["title"]}</div>'
        f'<div class="progress-step-desc">{step["desc"]}</div>'
        f'{detail_html}'
        f'</div>'
        f'{indicator}'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def _render_progress_completed():
    """Show the progress page in a fully-completed, frozen (read-only) state.
    No step-detail annotations — clean completed view."""
    _render_topbar("progress")

    st.markdown(
        "<div class='progress-header'>OPTIMISATION COMPLETE</div>",
        unsafe_allow_html=True,
    )

    # Render all steps as done WITHOUT detail annotations
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
    Відображення прогресу оптимізації: список етапів з прогрес-баром.
    Використовує чисту сторінку завдяки clean-page guard у main().
    """
    # If optimisation already completed (navigated back from results),
    # show frozen completed state — no re-run.
    if st.session_state.opt_complete and st.session_state.results is not None:
        _render_progress_completed()
        return

    # Hide Streamlit footer on progress page (scrolling enabled)
    st.markdown(
        """<style>
        footer, [data-testid="stBottom"] {
            display: none !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # Topbar — skip if already rendered by main() guard
    if not topbar_already:
        _render_topbar("progress")

    st.markdown(
        "<div class='progress-header'>COMPUTING \u00b7 PLEASE WAIT</div>",
        unsafe_allow_html=True,
    )

    # Перевірити вхідні дані
    if st.session_state.last_day_demand is None or st.session_state.planned_pressure is None:
        st.error("Input data missing. Please return to the input screen.")
        if st.button("Back to Input"):
            st.session_state.view = "input"
            st.rerun()
        return

    # Перевірити файли
    missing = []
    for f in [C.NETWORK_FILE, C.HISTORICAL_WINDOW_FILE]:
        if not os.path.exists(f):
            missing.append(os.path.basename(f))
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        st.info("Run prepare_window.py first to generate required data files.")
        return

    # Контейнери для кожного етапу — вертикальний список, одна колонка
    step_containers = []
    for i in range(len(_STEPS)):
        step_containers.append(st.empty())

    # Прогрес-бар і статус — під кроками
    progress_bar = st.progress(0.0)
    pct_text = st.empty()
    status_text = st.empty()

    # Стан етапів
    step_states = ["pending"] * len(_STEPS)
    step_details = [""] * len(_STEPS)

    def _refresh_steps():
        """Перемалювати всі картки етапів."""
        for i, step in enumerate(_STEPS):
            with step_containers[i]:
                _render_step_card(i, step, step_states[i], step_details[i])

    def step_callback(idx, status_val):
        """Callback з pipeline для оновлення стану етапу."""
        step_states[idx] = status_val
        _refresh_steps()

    # Початковий рендер — всі pending
    _refresh_steps()

    try:
        results = run_full_pipeline(
            last_day_demand=st.session_state.last_day_demand,
            planned_pressure_df=st.session_state.planned_pressure,
            progress_bar=progress_bar,
            status_text=status_text,
            step_callback=step_callback,
        )

        # Додати деталі до етапів
        meta = results.get("meta", {})
        step_details[0] = f"{meta.get('demand_rows', 24)} demand + {meta.get('pressure_rows', 24)} pressure rows ingested"
        step_details[1] = f"Training dataset: {meta.get('window_rows', 'N/A')} entries"
        step_details[2] = (
            f"SARIMAX {meta.get('sarimax_order', '')} \u00d7 {meta.get('sarimax_seasonal', '')} \u00b7 "
            f"AIC = {meta.get('sarimax_aic', 'N/A')} \u00b7 "
            f"Mean forecast = {meta.get('forecast_mean', 'N/A')} m\u00b3/h"
        )
        step_details[3] = (
            f"{meta.get('n_junctions', '?')} junctions \u00b7 "
            f"{meta.get('n_pumps', '?')} pumps \u00b7 "
            f"{meta.get('n_pipes', '?')} pipes"
        )
        step_details[4] = f"Baseline energy: {meta.get('E_planned', '?')} kWh"
        step_details[5] = (
            f"{C.POP_SIZE} candidates \u00d7 {C.N_GEN} generations"
        )
        step_details[6] = (
            f"Baseline: {meta.get('E_planned', '?')} kWh \u00b7 "
            f"Optimised: {meta.get('E_ga', '?')} kWh \u00b7 "
            f"Saving: {meta.get('saving_pct', '?')}%"
        )
        _refresh_steps()

        st.session_state.results = results
        st.session_state.opt_complete = True
        progress_bar.progress(1.0)
        pct_text.markdown(
            "<div style='text-align:center; font-family:monospace; "
            "color:#3fb950; font-size:14px;'>100%</div>",
            unsafe_allow_html=True,
        )

        # Show completion header and rerun to update topbar (buttons become clickable)
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
    Відображення результатів: top-bar навігації, графік + метрики,
    таблиця тиску, карта тиску мережі, футер.
    """
    results = st.session_state.results
    if results is None:
        st.warning("No results available.")
        if st.button("Back to Input"):
            st.session_state.view = "input"
            st.rerun()
        return

    # ── Metrics data ─────────────────────────────────────────────────────────
    saving_kwh = results["saving_kwh"]
    saving_pct = results["saving_pct"]
    saving_uah = results["saving_uah"]
    E_planned  = results["E_planned_kWh"]
    E_ga       = results["E_ga_kWh"]

    # ── Prepare CSV download data ────────────────────────────────────────────
    import io
    buf = io.BytesIO()
    pd.DataFrame({
        "Time": [f"{h:02d}:00" for h in range(C.N_STEPS)],
        "P_target_bar": np.round(results["ga_schedule"], 1),
    }).to_csv(buf, index=False)
    csv_bytes = buf.getvalue()

    # ── Top navigation bar with download button ──────────────────────────────
    _render_topbar("results", csv_bytes=csv_bytes)

    # ── Chart (90%) + side metrics (10%) ─────────────────────────────────────
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

    # ── Горизонтальна таблиця порівняння тиску по годинах ────────────────────
    _render_hourly_pressure_table(results)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # ── Карта тиску мережі ───────────────────────────────────────────────────
    st.markdown(
        "<div class='map-section-header'>Pressure Distribution Map</div>",
        unsafe_allow_html=True,
    )
    if results.get("wn") is not None and results.get("node_pressures_ga"):
        render_pressure_map(results)
    else:
        st.info("Pressure map data unavailable \u2014 check EPANET model file.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='page-footer'>"
        "Designed and Developed by Bohdan Hnat Studio, 2026"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_hourly_pressure_table(results: dict):
    """
    Відобразити горизонтальну таблицю: 2 рядки × 24 стовпці
    (плановий тиск vs оптимізований тиск по годинах).
    Оптимізований рядок підсвічений зеленим. Без прокрутки.
    """
    planned = results["planned_pressure"]
    ga = results["ga_schedule"]

    # Build header row
    header_cells = "".join(
        f"<th style='padding:5px 6px; font-size:14px; color:#8b949e; "
        f"border-bottom:1px solid #30363d; white-space:nowrap;'>"
        f"{h:02d}:00</th>" for h in range(24)
    )

    # Planned row (default styling)
    planned_cells = "".join(
        f"<td style='padding:5px 6px; font-size:15px; color:#c9d1d9; "
        f"border-bottom:1px solid #21262d; text-align:center;'>"
        f"{planned[i]:.2f}</td>" for i in range(24)
    )

    # Optimised row (green highlighted)
    ga_cells = "".join(
        f"<td style='padding:5px 6px; font-size:15px; color:#3fb950; "
        f"font-weight:700; background:rgba(63,185,80,0.08); "
        f"text-align:center;'>"
        f"{ga[i]:.2f}</td>" for i in range(24)
    )

    html = (
        "<div style='overflow-x:auto; overflow-y:hidden;'>"
        "<table style='width:100%; border-collapse:collapse; "
        "font-family:monospace; background:#0d1117;'>"
        f"<tr><th style='padding:5px 8px; font-size:14px; color:#8b949e; "
        f"text-align:left; border-bottom:1px solid #30363d; "
        f"white-space:nowrap; min-width:120px;'>Hour</th>{header_cells}</tr>"
        f"<tr><td style='padding:5px 8px; font-size:15px; color:#c9d1d9; "
        f"font-weight:600; border-bottom:1px solid #21262d; "
        f"white-space:nowrap;'>Planned (bar)</td>{planned_cells}</tr>"
        f"<tr><td style='padding:5px 8px; font-size:15px; color:#3fb950; "
        f"font-weight:700; background:rgba(63,185,80,0.08); "
        f"white-space:nowrap;'>Optimised (bar)</td>{ga_cells}</tr>"
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#   MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Головний маршрутизатор відображень."""
    view = st.session_state.view

    # ── Фіксований заголовок — показувати на всіх views ────────────────────
    _render_fixed_header()

    # ── Clean-page guard для view "progress" ─────────────────────────────────
    # When transitioning to progress for a NEW run, do two empty reruns so
    # Streamlit fully flushes page-1 widgets from the browser DOM before
    # the blocking pipeline call begins.  The header is rendered above so
    # the browser has a valid (minimal) script output to replace the old DOM.
    if view == "progress" and not st.session_state.opt_complete:
        # Render topbar BEFORE guard so buttons are visible during flush
        _render_topbar("progress")
        armed = st.session_state.get("_progress_armed", 0)
        if armed < 2:
            st.session_state["_progress_armed"] = armed + 1
            st.rerun()
            return  # unreachable; for linters

    # Reset arming flag when leaving progress view
    if view != "progress":
        st.session_state.pop("_progress_armed", None)

    # Scroll to top on page change
    if st.session_state.pop("_scroll_top", False):
        import streamlit.components.v1 as components
        components.html(
            "<script>window.parent.document.querySelector("
            "'[data-testid=\"stMainBlockContainer\"]')"
            ".scrollTo(0, 0);</script>",
            height=0,
        )

    if   view == "input":    render_input_view()
    elif view == "progress": render_progress_view(topbar_already=True)
    elif view == "results":  render_results_view()
    else:
        st.session_state.view = "input"
        st.rerun()


if __name__ == "__main__":
    main()
