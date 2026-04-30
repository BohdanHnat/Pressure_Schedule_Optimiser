"""
core/pipeline.py
================
Orchestrates the full daily optimisation pipeline (7 stages):

  Stage 1 — Reading operator input (demand + planned pressure)
  Stage 2 — Updating rolling window (appends new day, writes to session filesystem)
  Stage 3 — SARIMAX full retrain on updated window + 24h demand forecast
  Stage 4 — EPANET load + hydraulic simulation
  Stage 5 — Energy consumption baseline determination
  Stage 6 — GA pressure optimisation
  Stage 7 — Final simulation + result preparation

Baseline comparison logic:
  E_planned_kWh — EPANET simulation of operator's planned pressure schedule
                  with SARIMAX demand forecast.
  Reported saving = E_planned_kWh − E_ga_kWh.

OOP design note:
  PipelineResult replaces the original 15-key return dictionary. Attribute access
  (result.saving_pct) is clearer than dictionary access (result["saving_pct"]) and
  makes the full result structure explicit and self-documenting.

  Pipeline replaces the original single 120-line run_full_pipeline() function.
  Each stage is a clearly named method, making the sequence easy to follow.
"""
import os, sys
import numpy as np
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

from utils.rolling_window import (
    update_rolling_window, warm_start_retrain, generate_forecast
)
from core.optimisation import (
    load_network, simulate_schedule, epanet_energy,
    run_ga
)


@dataclass
class PipelineResult:
    """
    Typed container for all pipeline outputs.

    Using a dataclass (instead of a plain dict) gives attribute access,
    IDE autocomplete, and an explicit record of every result field.
    """
    Q_forecast: np.ndarray = None
    ga_schedule: np.ndarray = None
    planned_pressure: np.ndarray = None
    E_planned_kWh: float = 0.0
    E_ga_kWh: float = 0.0
    saving_kwh: float = 0.0
    saving_pct: float = 0.0
    saving_uah: float = 0.0
    annual_uah: float = 0.0
    node_pressures_ga: dict = field(default_factory=dict)
    node_pressures_planned: dict = field(default_factory=dict)
    wn: object = None
    pen_pressure: float = 0.0
    pen_bep: float = 0.0
    feasible: bool = False
    meta: dict = field(default_factory=dict)


def _tick(bar, status, frac, msg):
    """Update the progress bar and status text."""
    if bar:
        bar.progress(min(frac, 1.0))
    if status:
        status.markdown(
            f"<div style='text-align:center; font-family:monospace; "
            f"font-size:16px; padding-bottom:20px;'>"
            f"<strong>{msg}</strong></div>",
            unsafe_allow_html=True,
        )


def _extract_node_pressures(results, wn) -> dict:
    """Extract per-hour node pressures from an EPANET simulation result."""
    pressures = results.node["pressure"]
    data = {}
    for t in range(C.N_STEPS):
        hour_data = {}
        for jname in wn.junction_name_list:
            if jname in pressures.columns:
                # Convert from metres water column to bar
                p_bar = float(pressures[jname].iloc[t]) / 10.197
                hour_data[jname] = round(p_bar, 2)
        data[t] = hour_data
    return data


def _network_metadata(wn) -> dict:
    """Collect EPANET network metadata for the progress display."""
    return {
        "n_junctions": len(wn.junction_name_list),
        "n_pumps":     len(wn.pump_name_list),
        "n_pipes":     len(wn.pipe_name_list),
    }


class Pipeline:
    """
    Runs the full daily optimisation pipeline as a sequence of named stages.

    Using a class (instead of one long function) makes each stage a clearly
    named method and stores intermediate values as instance attributes,
    avoiding a tangle of local variables passed between code blocks.
    """

    def __init__(self, demand_df, pressure_df):
        self.demand_df = demand_df
        self.pressure_df = pressure_df
        self.result = PipelineResult()
        self._inp = C.NETWORK_FILE

    def run(self, progress_bar=None, status_text=None,
            step_callback=None) -> PipelineResult:
        """
        Execute all pipeline stages in order and return the populated PipelineResult.

        Parameters
        ----------
        progress_bar  : Streamlit progress bar widget, or None.
        status_text   : Streamlit empty() text placeholder, or None.
        step_callback : callable(step_index, status_string) — for live UI updates.

        NOTE: Streamlit — step_callback is called from within this blocking function
        to update the progress list in app.py via st.empty() containers. This is the
        only way to push live status updates from a long-running computation to the
        Streamlit UI. There is no simpler alternative within Streamlit's execution model.
        """
        self._stage1_read_input(progress_bar, status_text, step_callback)
        self._stage2_update_window(progress_bar, status_text, step_callback)
        self._stage3_retrain_forecast(progress_bar, status_text, step_callback)
        self._stage4_hydraulic_sim(progress_bar, status_text, step_callback)
        self._stage5_baseline_energy(progress_bar, status_text, step_callback)
        self._stage6_ga_optimisation(progress_bar, status_text, step_callback)
        self._stage7_prepare_results(progress_bar, status_text, step_callback)
        return self.result

    def _stage1_read_input(self, bar, status, cb):
        _tick(bar, status, 0.02, "Reading Operator Input...")
        if cb: cb(0, "running")
        self.result.meta["demand_rows"] = len(self.demand_df)
        self.result.meta["pressure_rows"] = len(self.pressure_df)
        self.result.planned_pressure = (
            self.pressure_df["P_planned_bar"].values.astype(float)
        )
        if cb: cb(0, "done")

    def _stage2_update_window(self, bar, status, cb):
        _tick(bar, status, 0.06, "Updating Demand Forecast Database...")
        if cb: cb(1, "running")
        self._window = update_rolling_window(self.demand_df)
        self.result.meta["window_rows"] = len(self._window)
        if cb: cb(1, "done")

    def _stage3_retrain_forecast(self, bar, status, cb):
        _tick(bar, status, 0.12, "Retraining Demand Forecast Model...")
        if cb: cb(2, "running")
        model_results = warm_start_retrain(self._window)
        self.result.meta["sarimax_order"]    = f"{C.SARIMAX_ORDER}"
        self.result.meta["sarimax_seasonal"] = f"{C.SARIMAX_SEASONAL_ORDER}"
        self.result.meta["sarimax_aic"]      = f"{model_results.aic:.1f}"
        _tick(bar, status, 0.20, "Generating 24h demand forecast...")
        self.result.Q_forecast = generate_forecast(model_results, self._window)
        self.result.meta["forecast_mean"] = f"{np.mean(self.result.Q_forecast):.0f}"
        if cb: cb(2, "done")

    def _stage4_hydraulic_sim(self, bar, status, cb):
        _tick(bar, status, 0.26, "Running Hydraulic Model Simulation...")
        if cb: cb(3, "running")
        self._wn, self._total_base_demand = load_network(self._inp)
        self.result.meta.update(_network_metadata(self._wn))
        if cb: cb(3, "done")

    def _stage5_baseline_energy(self, bar, status, cb):
        _tick(bar, status, 0.32, "Energy Consumption Baseline Determination...")
        if cb: cb(4, "running")
        self._res_planned, self._wn_planned = simulate_schedule(
            self._inp, self.result.planned_pressure,
            self.result.Q_forecast, self._total_base_demand,
        )
        self.result.E_planned_kWh = epanet_energy(self._res_planned)
        self.result.meta["E_planned"] = f"{self.result.E_planned_kWh:.1f}"
        if cb: cb(4, "done")

    def _stage6_ga_optimisation(self, bar, status, cb):
        _tick(bar, status, 0.38, "Optimised Energy Consumption Determination...")
        if cb: cb(5, "running")
        ga_schedule, _fitness, components = run_ga(
            inp_path=self._inp,
            Q_forecast=self.result.Q_forecast,
            total_base_demand=self._total_base_demand,
            E_baseline_kWh=self.result.E_planned_kWh,
            progress_bar=bar,
            status_text=status,
            stage_start=0.38,
            stage_end=0.86,
        )
        self.result.ga_schedule   = ga_schedule
        self.result.E_ga_kWh      = components[0]
        self.result.pen_pressure  = components[1]
        self.result.pen_bep       = components[2]
        if cb: cb(5, "done")

    def _stage7_prepare_results(self, bar, status, cb):
        _tick(bar, status, 0.90, "Preparing Results...")
        if cb: cb(6, "running")

        res_ga, wn_ga = simulate_schedule(
            self._inp, self.result.ga_schedule,
            self.result.Q_forecast, self._total_base_demand,
        )
        self.result.wn = wn_ga
        self.result.node_pressures_ga = _extract_node_pressures(res_ga, wn_ga)
        self.result.node_pressures_planned = _extract_node_pressures(
            self._res_planned, self._wn_planned,
        )

        saving_kwh = max(self.result.E_planned_kWh - self.result.E_ga_kWh, 0.0)
        self.result.saving_kwh = saving_kwh
        self.result.saving_pct = (
            saving_kwh / self.result.E_planned_kWh * 100
            if self.result.E_planned_kWh > 0 else 0.0
        )
        self.result.saving_uah = saving_kwh * C.TARIFF_UAH_PER_KWH
        self.result.annual_uah = self.result.saving_uah * C.ANNUAL_DAYS
        self.result.feasible   = self.result.pen_pressure == 0.0

        self.result.meta["E_ga"]        = f"{self.result.E_ga_kWh:.1f}"
        self.result.meta["saving_pct"]  = f"{self.result.saving_pct:.1f}"

        _tick(bar, status, 1.0, "Optimisation complete.")
        if cb: cb(6, "done")
