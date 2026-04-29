"""
utils/data_input.py
===================
Streamlit віджети для введення оператором вхідних даних:

  Секція A — Попит за попередні 24 години (м³/год).
  Секція B — Плановий тиск оператора на сьогодні (бар).

Обидва режими введення:
  a) Завантаження CSV — перший стовпець мітки часу, другий — значення.
  b) Ручне введення — 4-стовпчаста таблиця (12 рядків × 2 пари).

Висота таблиці налаштована так, щоб усі 24 години були видимі відразу
(без внутрішнього прокручування таблиці).
"""

import pandas as pd
import numpy as np
import streamlit as st

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


HOURS_LABELS = [f"{h:02d}:00" for h in range(24)]


# ── Допоміжна функція ─────────────────────────────────────────────────────────

def _build_consumption_df(values: np.ndarray,
                          timestamps: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    Побудувати DataFrame попиту з мітками часу.

    Якщо timestamps надано (CSV upload) — використовує їх без змін.
    Якщо timestamps не надано (ручне введення) — використовує фіксований
    якір 2026-01-29 00:00–23:00 (день після кінця вихідного вікна).

    Timestamps з CSV передаються прямо в rolling_window без перегенерації,
    що дозволяє позиційному захисту від дублікатів в update_rolling_window()
    коректно видаляти попередній запис того самого дня при кожному повторному
    запуску з тим самим файлом.
    """
    if timestamps is None:
        timestamps = pd.date_range(start=pd.Timestamp("2026-01-29"), periods=24, freq="h")
    return pd.DataFrame({"timestamp": timestamps, "hourly_demand_m3h": values.astype(float)})


# ── Секція A: Введення попиту ─────────────────────────────────────────────────

def render_demand_section() -> tuple:
    """
    Рендерити секцію введення попиту (ліва колонка).
    Повертає (df | None, has_input: bool).
    """
    st.markdown(
        "<div class='section-heading'>Last Day Demand</div>"
        "<div class='section-caption'>Hourly water consumption (m\u00b3/h) \u2014 previous day.</div>",
        unsafe_allow_html=True,
    )

    # ── Рядок завантаження: кнопка + підказка формату в одну лінію ───────────
    # label_visibility="collapsed" прибирає текст мітки; CSS перейменовує
    # кнопку "Browse files" → "Upload CSV" та знімає рамку/фон
    col_hint, col_up = st.columns([2, 1], gap="small")
    with col_hint:
        st.markdown(
            "<div class='upload-hint'>"
            "Column\u00a01: Hourly Timestamp<br>"
            "Column\u00a02: Demand Values (m\u00b3/h)"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_up:
        uploaded = st.file_uploader(
            "demand_upload",
            type="csv",
            key="demand_csv",
            label_visibility="collapsed",
        )

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            if len(raw.columns) < 2:
                st.error("CSV must have at least 2 columns.")
                return None, False
            if len(raw) != 24:
                st.error(f"CSV contains {len(raw)} rows; exactly 24 required.")
                return None, False
            # Перший стовпець — мітки часу, другий — значення попиту
            try:
                ts = pd.to_datetime(raw.iloc[:, 0])
                timestamps = pd.DatetimeIndex(ts).tz_localize(None)
            except Exception:
                timestamps = None  # fallback до фіксованого якоря
            vals = pd.to_numeric(raw.iloc[:, 1], errors="coerce").fillna(0.0).to_numpy()
            df = _build_consumption_df(vals, timestamps)
            # Save filename + method for back-navigation display
            st.session_state["_demand_filename"] = uploaded.name
            st.session_state["_demand_filesize"] = uploaded.size
            st.session_state["_demand_method"] = "csv"
            st.markdown(
                "<div class='csv-loaded-notice'>"
                "CSV file was successfully uploaded"
                "</div>",
                unsafe_allow_html=True,
            )
            return df, True
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, False

    # If file_uploader is empty but data exists from previous run (back-nav)
    if st.session_state.get("last_day_demand") is not None:
        method = st.session_state.get("_demand_method", "csv")
        if method == "csv":
            fname = st.session_state.get("_demand_filename", "uploaded file")
            fsize = st.session_state.get("_demand_filesize", 0)
            fsize_str = f"{fsize/1024:.1f}KB" if fsize >= 1024 else f"{fsize}B"
            st.markdown(
                f"<div style='font-family:monospace; font-size:13px; color:#8b949e; "
                f"padding:4px 0;'>"
                f"\U0001f4c4 {fname} &nbsp; <span style='font-size:11px'>{fsize_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='csv-loaded-notice'>"
                "CSV file was successfully uploaded"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='csv-loaded-notice'>"
                "The values were successfully entered manually"
                "</div>",
                unsafe_allow_html=True,
            )
        return st.session_state["last_day_demand"], True

    # ── Ручне введення — 4-стовпчаста таблиця ────────────────────────────────
    st.markdown(
        "<div class='alt-entry-label'>Alternative manual entry \u2014 fill in all 24 values below</div>",
        unsafe_allow_html=True,
    )

    # Ліва пара: 00:00–11:00 · Права пара: 12:00–23:00
    # Провідний пробіл у назвах правої пари запобігає конфлікту дублікатів
    init_df = pd.DataFrame({
        "Hour": HOURS_LABELS[:12],
        "Demand (m\u00b3/h)": [None] * 12,
        " Hour": HOURS_LABELS[12:],
        " Demand (m\u00b3/h)": [None] * 12,
    })

    edited = st.data_editor(
        init_df,
        width="stretch",
        hide_index=True,
        height=460,
        key="demand_editor",
        column_config={
            "Hour": st.column_config.TextColumn(disabled=True, width=68),
            "Demand (m\u00b3/h)": st.column_config.NumberColumn(
                min_value=0.0, format="%.1f", width=110
            ),
            " Hour": st.column_config.TextColumn(disabled=True, width=68),
            " Demand (m\u00b3/h)": st.column_config.NumberColumn(
                min_value=0.0, format="%.1f", width=110
            ),
        },
    )

    # Відновити 24 значення в правильному порядку (00:00–23:00)
    # pd.to_numeric уникає FutureWarning про downcast при fillna
    first_12  = edited["Demand (m\u00b3/h)"].values
    second_12 = edited[" Demand (m\u00b3/h)"].values
    all_vals  = pd.to_numeric(pd.Series(np.concatenate([first_12, second_12])), errors="coerce")
    all_filled = all_vals.notna().all()  # ALL 24 values required

    if all_filled:
        st.session_state["_demand_method"] = "manual"
        return _build_consumption_df(all_vals.to_numpy().astype(float)), True
    return None, False


# ── Секція B: Введення планового тиску ───────────────────────────────────────

def render_pressure_section() -> tuple:
    """
    Рендерити секцію введення планового тиску (права колонка).
    Повертає (df | None, has_input: bool).
    """
    st.markdown(
        "<div class='section-heading'>Planned Pressure Schedule</div>"
        "<div class='section-caption'>Enter today's planned pressure setpoints (bar) for baseline comparison.</div>",
        unsafe_allow_html=True,
    )

    # ── Рядок завантаження: кнопка + підказка формату в одну лінію ───────────
    col_hint, col_up = st.columns([2, 1], gap="small")
    with col_hint:
        st.markdown(
            "<div class='upload-hint'>"
            "Column\u00a01: Hourly Timestamp<br>"
            "Column\u00a02: Pressure Values (bar)"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_up:
        uploaded = st.file_uploader(
            "pressure_upload",
            type="csv",
            key="pressure_csv",
            label_visibility="collapsed",
        )

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            if len(raw.columns) < 2:
                st.error("CSV must have at least 2 columns.")
                return None, False
            if len(raw) != 24:
                st.error(f"CSV contains {len(raw)} rows; exactly 24 required.")
                return None, False
            # Другий стовпець — значення тиску (незалежно від назви)
            vals = np.clip(
                pd.to_numeric(raw.iloc[:, 1], errors="coerce").fillna(
                    (C.P_OUTLET_MIN + C.P_OUTLET_MAX) / 2
                ).to_numpy().astype(float),
                C.P_OUTLET_MIN, C.P_OUTLET_MAX,
            )
            df = pd.DataFrame({"Time": HOURS_LABELS, "P_planned_bar": vals})
            st.session_state["_pressure_filename"] = uploaded.name
            st.session_state["_pressure_filesize"] = uploaded.size
            st.session_state["_pressure_method"] = "csv"
            st.markdown(
                "<div class='csv-loaded-notice'>"
                "CSV file was successfully uploaded"
                "</div>",
                unsafe_allow_html=True,
            )
            return df, True
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, False

    # If file_uploader is empty but data exists from previous run (back-nav)
    if st.session_state.get("planned_pressure") is not None:
        method = st.session_state.get("_pressure_method", "csv")
        if method == "csv":
            fname = st.session_state.get("_pressure_filename", "uploaded file")
            fsize = st.session_state.get("_pressure_filesize", 0)
            fsize_str = f"{fsize/1024:.1f}KB" if fsize >= 1024 else f"{fsize}B"
            st.markdown(
                f"<div style='font-family:monospace; font-size:13px; color:#8b949e; "
                f"padding:4px 0;'>"
                f"\U0001f4c4 {fname} &nbsp; <span style='font-size:11px'>{fsize_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='csv-loaded-notice'>"
                "CSV file was successfully uploaded"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='csv-loaded-notice'>"
                "The values were successfully entered manually"
                "</div>",
                unsafe_allow_html=True,
            )
        return st.session_state["planned_pressure"], True

    # ── Ручне введення — 4-стовпчаста таблиця ────────────────────────────────
    st.markdown(
        "<div class='alt-entry-label'>Alternative manual entry \u2014 fill in all 24 values below</div>",
        unsafe_allow_html=True,
    )

    # Провідний пробіл у назвах правої пари запобігає конфлікту дублікатів
    init_df = pd.DataFrame({
        "Hour": HOURS_LABELS[:12],
        "Pressure (bar)": [None] * 12,
        " Hour": HOURS_LABELS[12:],
        " Pressure (bar)": [None] * 12,
    })

    edited = st.data_editor(
        init_df,
        width="stretch",
        hide_index=True,
        height=460,
        key="pressure_editor",
        column_config={
            "Hour": st.column_config.TextColumn(disabled=True, width=68),
            "Pressure (bar)": st.column_config.NumberColumn(
                min_value=0.0, format="%.2f",
                width=110,
            ),
            " Hour": st.column_config.TextColumn(disabled=True, width=68),
            " Pressure (bar)": st.column_config.NumberColumn(
                min_value=0.0, format="%.2f",
                width=110,
            ),
        },
    )

    # Відновити 24 значення в правильному порядку (00:00–23:00)
    first_12  = edited["Pressure (bar)"].values
    second_12 = edited[" Pressure (bar)"].values
    all_vals  = pd.to_numeric(pd.Series(np.concatenate([first_12, second_12])), errors="coerce")
    all_filled = all_vals.notna().all()  # ALL 24 values required

    if all_filled:
        st.session_state["_pressure_method"] = "manual"
        clipped = np.clip(all_vals.to_numpy().astype(float),
                          C.P_OUTLET_MIN, C.P_OUTLET_MAX)
        return pd.DataFrame({"Time": HOURS_LABELS, "P_planned_bar": clipped}), True
    return None, False