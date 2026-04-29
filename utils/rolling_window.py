"""
utils/rolling_window.py
=======================
Два основних завдання:
  1. update_rolling_window(last_day_demand) — додає 24 нові рядки, видаляє
     найстаріші 24, підтримує вікно рівно ROLLING_WINDOW_SIZE рядків, зберігає CSV.
  2. warm_start_retrain(window_df) — навчає SARIMAX на rolling window, повертає fitted results.

Обґрунтування (задокументовано у проєкті):
  - Статичний SARIMAX стає функціонально застарілим за кілька днів (AR компоненти дрейфують).
  - Після 26 тижнів rolling-операцій кожен слот дня тижня у вікні містить
    свіжі оперативні дані; жоден рядок архіву з початкового навчання не залишається.
  - Навчання з maxiter=50 завершується за 30–90 с на фіксованому вікні 4 320 рядків,
    що робить щоденне перенавчання операційно придатним перед зміною о 06:00.

ВАЖЛИВО (демо-режим):
  generate_forecast() прогнозує наступні 24 години від КІНЦЯ навчального набору,
  а НЕ від поточної дати. Це усуває розрив у часі між останнім записом навчальних
  даних та поточною датою.
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

# ── Допоміжна функція для Фур'є ─────────────────────────────────────────────

def _fourier_exog(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Обчислити безперервні Фур'є екзогенні змінні для DatetimeIndex.

    Тижневий цикл: період = 168 годин.
      hour_in_week = день_тижня × 24 + година   (0 … 167)
      sin = sin(2π × hour_in_week / 168)
      cos = cos(2π × hour_in_week / 168)

    Використання dayofweek (Понеділок=0 … Неділя=6) дає стабільне,
    детерміноване відображення; жодна опорна епоха не потрібна.
    Ідентична формула зі Script 05f.
    """
    h_in_week = timestamps.dayofweek * 24 + timestamps.hour
    angle = 2 * np.pi * h_in_week / C.FOURIER_PERIOD_H
    return pd.DataFrame(
        {"fourier_sin": np.sin(angle), "fourier_cos": np.cos(angle)},
        index=timestamps,
    )


# ── Крок 1 — оновлення rolling window ───────────────────────────────────────

def update_rolling_window(last_day_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Додати 24 нові рядки попиту та видалити найстаріші 24 рядки з вікна.

    Параметри
    ----------
    last_day_demand : pd.DataFrame
        Повинен містити стовпці: timestamp, hourly_demand_m3h.
        Стовпці Фур'є (пере-)обчислюються тут, тому виклику не потрібно їх надавати.

    Повертає
    -------
    pd.DataFrame — оновлене вікно (рівно ROLLING_WINDOW_SIZE рядків).

    Захист від дублікатів при повторних запусках:
        Якщо оператор завантажує той самий 24-годинний CSV повторно, timestamps
        у вікні вже містять цей день (доданий при попередньому запуску). Замість
        видалення за збігом timestamps (що могло б видалити правильний запис),
        функція відрізає хвіст вікна до останніх 24 рядків перед злиттям:
        це гарантує, що в об'єднаному вікні залишається лише один примірник
        будь-якого 24-годинного блоку незалежно від кількості запусків.
    """

    # Завантажити існуюче вікно
    window = pd.read_csv(C.HISTORICAL_WINDOW_FILE, parse_dates=["timestamp"])
    window = window.set_index("timestamp")
    window.index = pd.DatetimeIndex(window.index).tz_localize(None)

    # Conditional duplicate guard: only remove the last 24 rows if they
    # match the timestamps of the incoming data (i.e. the same day was
    # already appended in a previous run this session). If the last 24 rows
    # are a different day, leave the window untouched — the oldest 24 rows
    # are removed by the iloc[-ROLLING_WINDOW_SIZE:] trim after concat.
    incoming_ts = pd.DatetimeIndex(last_day_demand["timestamp"]).tz_localize(None)
    tail_ts = pd.DatetimeIndex(window.index[-24:]).tz_localize(None)
    if len(tail_ts) == 24 and (tail_ts == incoming_ts).all():
        window = window.iloc[:-24]

    # Перерахувати Фур'є для нових рядків для консистентності
    ts_index = pd.DatetimeIndex(last_day_demand["timestamp"])
    fourier   = _fourier_exog(ts_index)
    new_rows  = pd.DataFrame(
        {"hourly_demand_m3h": last_day_demand["hourly_demand_m3h"].values},
        index=ts_index,
    )
    new_rows[C.EXOG_COLS] = fourier.values

    # Додати нові рядки та обрізати до розміру вікна
    window = pd.concat([window, new_rows])
    window = window.iloc[-C.ROLLING_WINDOW_SIZE:]

    window.index.name = "timestamp"
    # Write the updated window back to disk so that generate_forecast() reads
    # the correct anchor timestamp (last hour of the newly appended day).
    # On Streamlit Cloud the ephemeral filesystem holds this write for the
    # duration of the session; the deployed file is untouched on the next run.
    window.to_csv(C.HISTORICAL_WINDOW_FILE)
    return window


# ── Крок 2 — повне перенавчання SARIMAX ─────────────────────────────────────

def warm_start_retrain(window: pd.DataFrame | None = None) -> object:
    """
    Перенавчити SARIMAX на поточному rolling window.
    Повертає fitted results.

    Параметри
    ----------
    window : pd.DataFrame | None
        Якщо None, завантажує з HISTORICAL_WINDOW_FILE автоматично.

    Конфігурація SARIMAX (ідентична Script 05f, debug mode):
      order          = (1, 0, 1)
      seasonal_order = (1, 1, 1, 24)
      exog           = ['fourier_sin', 'fourier_cos']
    """
    if window is None:
        window = pd.read_csv(
            C.HISTORICAL_WINDOW_FILE, parse_dates=["timestamp"], index_col="timestamp"
        )

    window.index = pd.DatetimeIndex(window.index).tz_localize(None)
    window = window[~window.index.duplicated(keep='last')]  # захист від дублікатів
    window = window.asfreq("h")

    # Інтерполювати прогалини (рідко після очищення даних)
    window["hourly_demand_m3h"] = window["hourly_demand_m3h"].interpolate(method="time")
    for col in C.EXOG_COLS:
        window[col] = window[col].interpolate(method="time")
    window = window.dropna()

    train_y = window["hourly_demand_m3h"]
    train_X = window[C.EXOG_COLS]

    model = SARIMAX(
        train_y,
        exog=train_X,
        order=C.SARIMAX_ORDER,
        seasonal_order=C.SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit_kwargs = dict(disp=False, maxiter=C.RETRAIN_MAXITER)

    new_results = model.fit(**fit_kwargs)

    # Roll back the on-disk window by removing the last 24 rows (the
    # just-appended day) so that the next session starts from the same
    # state as the GitHub version — one day before the demo input date.
    # The in-memory window passed to generate_forecast() is unaffected;
    # only the file on the ephemeral filesystem is trimmed here.
    rolled_back = window.iloc[:-24]
    rolled_back.index.name = "timestamp"
    rolled_back.to_csv(C.HISTORICAL_WINDOW_FILE)

    return new_results

# ── Крок 3 — генерація 24-годинного прогнозу ────────────────────────────────

def generate_forecast(model_results, window: pd.DataFrame | None = None) -> np.ndarray:
    """
    Згенерувати 24-годинний прогноз попиту (м³/год).

    ВАЖЛИВО (демо-режим): прогнозує наступні 24 години від КІНЦЯ навчального
    набору (останньої мітки часу в historical_window.csv), а НЕ від поточної
    дати. Це усуває будь-який часовий розрив між навчальними даними та прогнозом.

    Повертає np.ndarray форми (24,) в м³/год.
    """
    # Визначити останню мітку часу з навчального вікна
    if window is not None:
        last_ts = pd.DatetimeIndex(window.index).tz_localize(None).max()
    else:
        w = pd.read_csv(
            C.HISTORICAL_WINDOW_FILE, parse_dates=["timestamp"], index_col="timestamp"
        )
        last_ts = pd.DatetimeIndex(w.index).max()

    # Прогнозувати наступні 24 години ОДРАЗУ ПІСЛЯ кінця навчальних даних
    # (НЕ від поточної дати — усуває часовий розрив)
    next_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h"
    )
    exog_future = _fourier_exog(next_timestamps)

    forecast = model_results.get_forecast(steps=24, exog=exog_future)
    Q_forecast = np.maximum(forecast.predicted_mean.values, 0.0)  # попит ≥ 0

    return Q_forecast