"""
config.py — Vitruka Dashboard: central constants and file paths.
All other modules import from here; change values only in this file.
"""

import os
import numpy as np

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data_files")
NETWORKS_DIR  = os.path.join(BASE_DIR, "networks")

# ── File paths ────────────────────────────────────────────────────────────────
NETWORK_FILE          = os.path.join(NETWORKS_DIR, "Vitruka_Model.inp")
# historical_window.csv is the static pre-built dataset loaded at runtime.
HISTORICAL_WINDOW_FILE= os.path.join(DATA_DIR, "SARIMAX_Historical_Window.csv")

# ── SARIMAX parameters (debug: identical to provided Script 05f) ──────────────
SARIMAX_ORDER          = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 24)
EXOG_COLS              = ["fourier_sin", "fourier_cos"]
ROLLING_WINDOW_SIZE    = 4320          # 180 days × 24 h
RETRAIN_MAXITER        = 50
FOURIER_PERIOD_H       = 168           # weekly cycle in hours

# ── GA parameters (debug values per user specification) ──────────────────────
POP_SIZE     = 80
N_GEN        = 1
N_STEPS      = 24
WARM_FRAC    = 0.10
ENERGY_WEIGHT= 80
PEN_BEP      = 5

# ── Pump — Grundfos CRE 45-2-2 ───────────────────────────────────────────────
Q_BEP       = 43.0    # m³/h — best efficiency point
H_SHUTOFF   = 60.0    # m    — shutoff head at Q=0, speed=1.0
F_RATED     = 50.0    # Hz
F_MIN       = 30.0    # Hz   — VFD minimum
PUMP_IDS    = ["PU1", "PU2"]

# ── EPANET geometry ───────────────────────────────────────────────────────────
ELEV_R1 = 217.8   # m — reservoir R1 (back-calculated from SCADA)
ELEV_J2 = 220.1   # m — pump outlet junction J2

# ── Pressure domains ──────────────────────────────────────────────────────────
# Domain 1: GA chromosome / station outlet (gauge pressure, bar)
P_OUTLET_MIN = 3.5    # bar — lower search bound (SCADA-confirmed; J39 hydraulics)
P_OUTLET_MAX = 4.4    # bar — upper search bound (SCADA P_90%)

# Domain 2: Consumer-node regulatory limits (ДБН В.2.5-74:2013 §6.3.1)
P_MIN_CONSUMER = 2.5  # bar — minimum service pressure
P_MAX_CONSUMER = 6.0  # bar — maximum (eliminates elevation false-penalties)

# ── EFF curve — Grundfos CRE 45-2-2 ([CURVES] from .inp) ─────────────────────
EFF_CURVE_Q   = np.array([0.0, 10.0, 20.0, 30.0, 43.0, 50.0, 60.0, 70.0])
EFF_CURVE_ETA = np.array([0.00, 39.23, 59.23, 70.00,
                           75.80, 75.38, 70.00, 56.16]) / 100.0

# ── Tariff (debug: fixed constant, UAH per kWh) ───────────────────────────────
TARIFF_UAH_PER_KWH = 10.0
ANNUAL_DAYS        = 365