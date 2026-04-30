"""
utils/data_input.py
===================
Streamlit widgets for operator data entry:

  Section A — Last 24-hour demand (m³/h).
  Section B — Planned pressure schedule for today (bar).

Two input modes:
  a) CSV upload — first column timestamps, second column values.
  b) Manual 24-row table — 4-column layout (12 rows × 2 pairs).

Table height is configured so all 24 hours are visible without internal scrolling.

OOP design note:
  The original file had two nearly-identical ~150-line functions. The shared logic
  is now in the DataSection base class. DemandSection and PressureSection override
  only the three methods that differ: _build_df, _column_config, _build_df_from_manual.
  This removes ~90% code duplication while keeping all Streamlit functionality intact.
"""

import pandas as pd
import numpy as np
import streamlit as st

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


HOURS_LABELS = [f"{h:02d}:00" for h in range(24)]


def _build_consumption_df(values: np.ndarray,
                          timestamps: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    Build a demand DataFrame with timestamps.

    If timestamps are provided (CSV upload) — uses them unchanged.
    If timestamps are not provided (manual entry) — uses a fixed anchor
    2026-01-29 00:00–23:00 (day after the end of the initial historical window).

    Timestamps from CSV are passed directly to rolling_window without regeneration,
    allowing the positional duplicate guard in update_rolling_window() to correctly
    remove the previous entry for the same day on each re-run with the same file.
    """
    if timestamps is None:
        timestamps = pd.date_range(start=pd.Timestamp("2026-01-29"), periods=24, freq="h")
    return pd.DataFrame({"timestamp": timestamps, "hourly_demand_m3h": values.astype(float)})


class DataSection:
    """
    Encapsulates shared input logic for demand and pressure sections.

    Both sections share the same structure: CSV upload or manual 24-row table.
    The only differences are column names, session state keys, and validation bounds.
    Subclasses override _build_df, _column_config, and _build_df_from_manual.
    """

    def __init__(self, title, caption, col1_name, col2_name,
                 fill_value, value_bounds, session_key, method_key,
                 filename_key, filesize_key, upload_hint, upload_key):
        self.title = title
        self.caption = caption
        self.col1_name = col1_name       # e.g. "Demand (m³/h)"
        self.col2_name = col2_name       # leading space avoids duplicate column name in data_editor
        self.fill_value = fill_value     # default for NaN entries in CSV
        self.value_bounds = value_bounds # (min, max) for clipping, or None
        self.session_key = session_key   # e.g. "last_day_demand"
        self.method_key = method_key     # e.g. "_demand_method"
        self.filename_key = filename_key
        self.filesize_key = filesize_key
        self.upload_hint = upload_hint   # HTML string for the format hint
        self.upload_key = upload_key     # Streamlit widget key for file_uploader

    def render(self) -> tuple:
        """
        Render the section heading, file uploader, and manual entry table.
        Returns (DataFrame | None, has_input: bool).
        """
        st.markdown(
            f"<div class='section-heading'>{self.title}</div>"
            f"<div class='section-caption'>{self.caption}</div>",
            unsafe_allow_html=True,
        )

        col_hint, col_up = st.columns([2, 1], gap="small")
        with col_hint:
            st.markdown(
                f"<div class='upload-hint'>{self.upload_hint}</div>",
                unsafe_allow_html=True,
            )
        with col_up:
            # NOTE: Streamlit — label_visibility="collapsed" hides the label text.
            # file_uploader returns None when no file is selected, or a file-like object.
            # These are Streamlit-specific API behaviours with no plain-Python equivalent.
            uploaded = st.file_uploader(
                f"{self.upload_key}_label",
                type="csv",
                key=self.upload_key,
                label_visibility="collapsed",
            )

        if uploaded:
            return self._parse_csv(uploaded)

        # If the uploader is empty but data exists from a previous run (back-navigation),
        # display the stored file info and return the saved DataFrame.
        # NOTE: Streamlit — st.session_state persists values across reruns. After navigating
        # away and back, the file_uploader resets to None, but session_state retains the data.
        if st.session_state.get(self.session_key) is not None:
            self._show_back_nav_notice()
            return st.session_state[self.session_key], True

        return self._render_manual_table()

    def _parse_csv(self, uploaded) -> tuple:
        """Parse uploaded CSV. Returns (DataFrame, True) or (None, False)."""
        try:
            raw = pd.read_csv(uploaded)
            if len(raw.columns) < 2:
                st.error("CSV must have at least 2 columns.")
                return None, False
            if len(raw) != 24:
                st.error(f"CSV contains {len(raw)} rows; exactly 24 required.")
                return None, False
            vals = (pd.to_numeric(raw.iloc[:, 1], errors="coerce")
                    .fillna(self.fill_value).to_numpy().astype(float))
            if self.value_bounds:
                vals = np.clip(vals, *self.value_bounds)
            df = self._build_df(vals, raw)
            st.session_state[self.filename_key] = uploaded.name
            st.session_state[self.filesize_key] = uploaded.size
            st.session_state[self.method_key] = "csv"
            st.markdown(
                "<div class='csv-loaded-notice'>CSV file was successfully uploaded</div>",
                unsafe_allow_html=True,
            )
            return df, True
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, False

    def _show_back_nav_notice(self):
        """Display the upload status notice when returning from another page."""
        method = st.session_state.get(self.method_key, "csv")
        if method == "csv":
            fname = st.session_state.get(self.filename_key, "uploaded file")
            fsize = st.session_state.get(self.filesize_key, 0)
            fsize_str = f"{fsize/1024:.1f}KB" if fsize >= 1024 else f"{fsize}B"
            st.markdown(
                f"<div style='font-family:monospace; font-size:13px; color:#8b949e; "
                f"padding:4px 0;'>"
                f"\U0001f4c4 {fname} &nbsp; <span style='font-size:11px'>{fsize_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='csv-loaded-notice'>CSV file was successfully uploaded</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='csv-loaded-notice'>The values were successfully entered manually</div>",
                unsafe_allow_html=True,
            )

    def _render_manual_table(self) -> tuple:
        """
        Render 4-column manual entry table (12 rows × 2 pairs = 24 hours).
        Returns (DataFrame, True) if all 24 values are filled, else (None, False).
        """
        st.markdown(
            "<div class='alt-entry-label'>Alternative manual entry — "
            "fill in all 24 values below</div>",
            unsafe_allow_html=True,
        )

        # Leading space in right-pair column names prevents duplicate-name conflicts
        # inside Streamlit's data_editor widget.
        init_df = pd.DataFrame({
            "Hour": HOURS_LABELS[:12],
            self.col1_name: [None] * 12,
            " Hour": HOURS_LABELS[12:],
            self.col2_name: [None] * 12,
        })

        # NOTE: Streamlit — st.data_editor with column_config controls column editability,
        # numeric range, and display format. These are Streamlit-specific parameters
        # with no plain-Python equivalent.
        edited = st.data_editor(
            init_df,
            width="stretch",
            hide_index=True,
            height=460,
            key=f"{self.upload_key}_editor",
            column_config=self._column_config(),
        )

        first_12 = edited[self.col1_name].values
        second_12 = edited[self.col2_name].values
        all_vals = pd.to_numeric(
            pd.Series(np.concatenate([first_12, second_12])), errors="coerce"
        )
        all_filled = all_vals.notna().all()

        if all_filled:
            st.session_state[self.method_key] = "manual"
            return self._build_df_from_manual(all_vals.to_numpy().astype(float)), True
        return None, False

    def _build_df(self, vals, raw) -> pd.DataFrame:
        """Build output DataFrame from CSV values. Overridden per section."""
        raise NotImplementedError

    def _column_config(self) -> dict:
        """Return column_config dict for st.data_editor. Overridden per section."""
        raise NotImplementedError

    def _build_df_from_manual(self, vals) -> pd.DataFrame:
        """Build output DataFrame from manually entered values. Overridden per section."""
        raise NotImplementedError


class DemandSection(DataSection):
    """Data entry section for yesterday's hourly water demand (m³/h)."""

    def __init__(self):
        super().__init__(
            title="Last Day Demand",
            caption="Hourly water consumption (m³/h) — previous day.",
            col1_name="Demand (m³/h)",
            col2_name=" Demand (m³/h)",
            fill_value=0.0,
            value_bounds=None,
            session_key="last_day_demand",
            method_key="_demand_method",
            filename_key="_demand_filename",
            filesize_key="_demand_filesize",
            upload_hint=(
                "Column 1: Hourly Timestamp<br>"
                "Column 2: Demand Values (m³/h)"
            ),
            upload_key="demand_csv",
        )

    def _build_df(self, vals, raw) -> pd.DataFrame:
        try:
            ts = pd.to_datetime(raw.iloc[:, 0])
            timestamps = pd.DatetimeIndex(ts).tz_localize(None)
        except Exception:
            timestamps = None  # fallback to fixed anchor date
        return _build_consumption_df(vals, timestamps)

    def _column_config(self) -> dict:
        return {
            "Hour": st.column_config.TextColumn(disabled=True, width=68),
            "Demand (m³/h)": st.column_config.NumberColumn(
                min_value=0.0, format="%.1f", width=110
            ),
            " Hour": st.column_config.TextColumn(disabled=True, width=68),
            " Demand (m³/h)": st.column_config.NumberColumn(
                min_value=0.0, format="%.1f", width=110
            ),
        }

    def _build_df_from_manual(self, vals) -> pd.DataFrame:
        return _build_consumption_df(vals)


class PressureSection(DataSection):
    """Data entry section for today's planned pressure schedule (bar)."""

    def __init__(self):
        super().__init__(
            title="Planned Pressure Schedule",
            caption="Enter today's planned pressure setpoints (bar) for baseline comparison.",
            col1_name="Pressure (bar)",
            col2_name=" Pressure (bar)",
            fill_value=(C.P_OUTLET_MIN + C.P_OUTLET_MAX) / 2,
            value_bounds=(C.P_OUTLET_MIN, C.P_OUTLET_MAX),
            session_key="planned_pressure",
            method_key="_pressure_method",
            filename_key="_pressure_filename",
            filesize_key="_pressure_filesize",
            upload_hint=(
                "Column 1: Hourly Timestamp<br>"
                "Column 2: Pressure Values (bar)"
            ),
            upload_key="pressure_csv",
        )

    def _build_df(self, vals, raw) -> pd.DataFrame:
        return pd.DataFrame({"Time": HOURS_LABELS, "P_planned_bar": vals})

    def _column_config(self) -> dict:
        return {
            "Hour": st.column_config.TextColumn(disabled=True, width=68),
            "Pressure (bar)": st.column_config.NumberColumn(
                min_value=0.0, format="%.2f", width=110
            ),
            " Hour": st.column_config.TextColumn(disabled=True, width=68),
            " Pressure (bar)": st.column_config.NumberColumn(
                min_value=0.0, format="%.2f", width=110
            ),
        }

    def _build_df_from_manual(self, vals) -> pd.DataFrame:
        clipped = np.clip(vals, C.P_OUTLET_MIN, C.P_OUTLET_MAX)
        return pd.DataFrame({"Time": HOURS_LABELS, "P_planned_bar": clipped})


def render_demand_section() -> tuple:
    """Render the demand input section. Returns (DataFrame | None, has_input: bool)."""
    return DemandSection().render()


def render_pressure_section() -> tuple:
    """Render the pressure input section. Returns (DataFrame | None, has_input: bool)."""
    return PressureSection().render()
