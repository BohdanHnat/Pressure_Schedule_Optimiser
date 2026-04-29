"""
core/pipeline.py
================
Оркеструє повний щоденний конвеєр оптимізації (6 етапів):

  Етап 1 — Прийом вхідних даних (попит + плановий тиск)
  Етап 2 — Оновлення rolling window (appends new day, writes to session filesystem)
  Етап 3 — SARIMAX повне перенавчання на оновленому вікні + прогноз попиту 24 год
  Етап 4 — Завантаження EPANET + симуляція планових уставок оператора
  Етап 5 — GA оптимізація тиску
  Етап 6 — Фінальна симуляція + формування результатів

Базова логіка:
  Плановий розклад тиску оператора (Крок 2) — єдина база порівняння
  для фітнес-функції GA та відображення економії на дашборді.

  E_planned_kWh — EPANET симуляція планових уставок оператора з
                  SARIMAX прогнозом попиту.

  Звітна економія = E_planned_kWh − E_ga_kWh
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

from utils.rolling_window import (
    update_rolling_window, warm_start_retrain, generate_forecast
)
from core.optimisation import (
    load_network, simulate_schedule, epanet_energy,
    run_ga
)


# ── Допоміжні функції для прогрес-бару ───────────────────────────────────────

def _tick(bar, status, frac, msg):
    """Оновити прогрес-бар та текст статусу."""
    if bar:
        bar.progress(min(frac, 1.0))
    if status:
        status.markdown(
            f"<div style='text-align:center; font-family:monospace; "
            f"font-size:16px; padding-bottom:20px;'>"
            f"<strong>{msg}</strong></div>",
            unsafe_allow_html=True,
        )


def _extract_node_pressures(results, wn):
    """Витягти тиск у вузлах мережі для кожної години."""
    pressures = results.node["pressure"]
    data = {}
    for t in range(C.N_STEPS):
        hour_data = {}
        for jname in wn.junction_name_list:
            if jname in pressures.columns:
                # Перетворення з метрів водяного стовпа в бари
                p_bar = float(pressures[jname].iloc[t]) / 10.197
                hour_data[jname] = round(p_bar, 2)
        data[t] = hour_data
    return data


# ── Збір метаданих мережі для прогрес-бару ───────────────────────────────────

def _network_metadata(wn) -> dict:
    """Зібрати метадані мережі EPANET для відображення у прогрес-барі."""
    return {
        "n_junctions": len(wn.junction_name_list),
        "n_pumps": len(wn.pump_name_list),
        "n_pipes": len(wn.pipe_name_list),
    }


def run_full_pipeline(last_day_demand, planned_pressure_df,
                      progress_bar=None, status_text=None,
                      step_callback=None):
    """
    Виконати всі етапи конвеєра оптимізації та повернути словник результатів.

    Параметри
    ----------
    last_day_demand      : pd.DataFrame  стовпці: timestamp, hourly_demand_m3h (24 рядки)
    planned_pressure_df  : pd.DataFrame  стовпці: Time, P_planned_bar (24 рядки)
    step_callback        : callable(step_index, status) — для оновлення прогрес-листа

    Повертає словник з ключами:
      Q_forecast, ga_schedule, planned_pressure
      E_planned_kWh, E_ga_kWh
      saving_kwh, saving_pct, saving_uah, annual_uah
      node_pressures_ga, node_pressures_planned
      wn, pen_pressure, pen_bep, feasible
      meta (метадані мережі та SARIMAX)
    """
    inp       = C.NETWORK_FILE
    planned_p = planned_pressure_df["P_planned_bar"].values.astype(float)

    # Метадані для збору інформації про прогрес
    meta = {}

    # ── Step 1: Reading Operator Input ──────────────────────────────────────
    _tick(progress_bar, status_text, 0.02,
          "Reading Operator Input...")
    if step_callback:
        step_callback(0, "running")

    meta["demand_rows"] = len(last_day_demand)
    meta["pressure_rows"] = len(planned_pressure_df)

    if step_callback:
        step_callback(0, "done")

    # ── Step 2: Updating Demand Forecast Database ────────────────────────────
    _tick(progress_bar, status_text, 0.06,
          "Updating Demand Forecast Database...")
    if step_callback:
        step_callback(1, "running")

    window = update_rolling_window(last_day_demand)
    meta["window_rows"] = len(window)

    if step_callback:
        step_callback(1, "done")

    # ── Step 3: Retraining Demand Forecast Model ─────────────────────────────
    _tick(progress_bar, status_text, 0.12,
          "Retraining Demand Forecast Model...")
    if step_callback:
        step_callback(2, "running")

    model_results = warm_start_retrain(window)

    meta["sarimax_order"] = f"{C.SARIMAX_ORDER}"
    meta["sarimax_seasonal"] = f"{C.SARIMAX_SEASONAL_ORDER}"
    meta["sarimax_aic"] = f"{model_results.aic:.1f}"

    _tick(progress_bar, status_text, 0.20,
          "Generating 24h demand forecast...")
    Q_forecast = generate_forecast(model_results, window)
    meta["forecast_mean"] = f"{np.mean(Q_forecast):.0f}"

    if step_callback:
        step_callback(2, "done")

    # ── Step 4: Running Hydraulic Model Simulation ───────────────────────────
    _tick(progress_bar, status_text, 0.26,
          "Running Hydraulic Model Simulation...")
    if step_callback:
        step_callback(3, "running")

    wn, total_base_demand = load_network(inp)
    net_meta = _network_metadata(wn)
    meta.update(net_meta)

    if step_callback:
        step_callback(3, "done")

    # ── Step 5: Energy Consumption Baseline Determination ────────────────────
    _tick(progress_bar, status_text, 0.32,
          "Energy Consumption Baseline Determination...")
    if step_callback:
        step_callback(4, "running")

    res_planned, wn_planned = simulate_schedule(
        inp, planned_p, Q_forecast, total_base_demand)
    E_planned_kWh = epanet_energy(res_planned)

    if step_callback:
        step_callback(4, "done")

    # ── Step 6: Optimised Energy Consumption Determination ───────────────────
    _tick(progress_bar, status_text, 0.38,
          "Optimised Energy Consumption Determination...")
    if step_callback:
        step_callback(5, "running")

    ga_schedule, fitness, components = run_ga(
        inp_path=inp,
        Q_forecast=Q_forecast,
        total_base_demand=total_base_demand,
        E_baseline_kWh=E_planned_kWh,
        progress_bar=progress_bar,
        status_text=status_text,
        stage_start=0.38,
        stage_end=0.86,
    )
    E_ga_kWh     = components[0]
    pen_pressure = components[1]
    pen_bep      = components[2]

    if step_callback:
        step_callback(5, "done")

    # ── Step 7: Preparing Results ────────────────────────────────────────────
    _tick(progress_bar, status_text, 0.90,
          "Preparing Results...")
    if step_callback:
        step_callback(6, "running")

    res_ga, wn_ga = simulate_schedule(inp, ga_schedule, Q_forecast, total_base_demand)

    node_pressures_ga      = _extract_node_pressures(res_ga,     wn_ga)
    node_pressures_planned = _extract_node_pressures(res_planned, wn_planned)

    saving_kwh = max(E_planned_kWh - E_ga_kWh, 0.0)
    saving_pct = (saving_kwh / E_planned_kWh * 100) if E_planned_kWh > 0 else 0.0
    saving_uah = saving_kwh * C.TARIFF_UAH_PER_KWH
    annual_uah = saving_uah * C.ANNUAL_DAYS

    meta["E_planned"] = f"{E_planned_kWh:.1f}"
    meta["E_ga"] = f"{E_ga_kWh:.1f}"
    meta["saving_pct"] = f"{saving_pct:.1f}"

    _tick(progress_bar, status_text, 1.0, "Optimisation complete.")

    if step_callback:
        step_callback(6, "done")

    return {
        "Q_forecast":              Q_forecast,
        "ga_schedule":             ga_schedule,
        "planned_pressure":        planned_p,
        "E_planned_kWh":           E_planned_kWh,
        "E_ga_kWh":                E_ga_kWh,
        "saving_kwh":              saving_kwh,
        "saving_pct":              saving_pct,
        "saving_uah":              saving_uah,
        "annual_uah":              annual_uah,
        "node_pressures_ga":       node_pressures_ga,
        "node_pressures_planned":  node_pressures_planned,
        "wn":                      wn_ga,
        "pen_pressure":            pen_pressure,
        "pen_bep":                 pen_bep,
        "feasible":                pen_pressure == 0.0,
        "meta":                    meta,
    }