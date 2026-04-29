"""
core/optimisation.py
====================
Single-objective GA for Vitruka pump scheduling — adapted from the
provided GA Optimisation Script (v3 — pymoo).

All fitness functions, operators, and the VitrukaScheduleProblem class
are reproduced verbatim from the provided script.  The only additions are:

  1. Constants imported from config (POP_SIZE=80, N_GEN=30 for debug run).
  2. p_target_to_speed() uses config.ELEV_J2 / ELEV_R1 instead of literals.
  3. run_ga() accepts an optional Streamlit progress_bar + status_text,
     updated per generation via StreamlitProgressCallback (pymoo.Callback).

Architecture references (from provided script — unchanged):
  Savic & Walters (1997) / Mala-Jetmarova et al. (2017) / Kazimipour et al. (2014)
"""
import os, sys
import numpy as np
import wntr

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

# ── EPANET helpers (unchanged from provided script) ──────────────────────────

def p_target_to_speed(p_bar: float) -> float:
    """P_target (bar, gauge at J2) → VFD speed ratio.  Affinity law."""
    H_outlet_abs  = p_bar * 10.197 + C.ELEV_J2
    H_pump_needed = max(H_outlet_abs - C.ELEV_R1, 1.0)
    ratio = np.sqrt(H_pump_needed / C.H_SHUTOFF)
    return float(np.clip(ratio, C.F_MIN / C.F_RATED, 1.0))

def simulate_schedule(inp_path: str, P_schedule, Q_forecast, total_base_demand):
    """Run EPANET EPS for one P_target schedule + demand forecast."""
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.get_pattern("PATTERN").multipliers = list(Q_forecast / total_base_demand)

    speeds   = [p_target_to_speed(p) for p in P_schedule]
    pat_name = "ga_speed"
    if pat_name in wn.pattern_name_list:
        wn.get_pattern(pat_name).multipliers = speeds
    else:
        wn.add_pattern(pat_name, speeds)

    for pid in C.PUMP_IDS:
        pump = wn.get_link(pid)
        pump.speed_timeseries.base_value   = 1.0
        pump.speed_timeseries.pattern_name = pat_name

    return wntr.sim.EpanetSimulator(wn).run_sim(), wn

def load_network(inp_path: str):
    """Load EPANET model; compute total PATTERN base demand (CMH)."""
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"EPANET model not found: {inp_path}")
    wn  = wntr.network.WaterNetworkModel(inp_path)
    tbd = 0.0
    for jname in wn.junction_name_list:
        for d in wn.get_node(jname).demand_timeseries_list:
            if d.pattern is not None and d.pattern.name == "PATTERN":
                tbd += d.base_value * 3600.0
    return wn, tbd

# Fitness components
def epanet_energy(results) -> float:
    """Total pump energy kWh via EFF curve (identical to provided script)."""
    total = 0.0
    for pid in C.PUMP_IDS:
        flow_ms = results.link["flowrate"][pid].abs()
        head_m  = results.link["headloss"][pid].abs()
        for t in range(min(C.N_STEPS, len(flow_ms))):
            q_ms = float(flow_ms.iloc[t])
            if q_ms < 1e-4:
                continue
            h_m = float(head_m.iloc[t])
            if not np.isfinite(h_m) or h_m > 500.0:
                continue
            q_cmh    = q_ms * 3600.0
            eta      = float(np.interp(q_cmh, C.EFF_CURVE_Q, C.EFF_CURVE_ETA))
            eta      = max(eta, 0.01)
            total   += (9810.0 * q_ms * h_m) / (eta * 1000.0)
    return total


def loss_pressure(results, wn) -> float:
    """Regulatory penalty — ДБН В.2.5-74:2013 §6.3.1 (unchanged)."""
    penalty   = 0.0
    pressures = results.node["pressure"]
    for jname in wn.junction_name_list:
        if jname not in pressures.columns:
            continue
        p_arr = pressures[jname].values
        for t in range(min(C.N_STEPS, len(p_arr))):
            p_bar = p_arr[t] / 10.197
            if C.P_MIN_CONSUMER <= p_bar <= C.P_MAX_CONSUMER:
                continue
            dev = (C.P_MIN_CONSUMER - p_bar) if p_bar < C.P_MIN_CONSUMER \
                  else (p_bar - C.P_MAX_CONSUMER)
            penalty += 3000 if dev >= 1.0 else (1000 if dev >= 0.5 else 300)
    return penalty

def loss_bep(results, wn) -> float:
    """BEP deviation — soft secondary penalty (unchanged)."""
    penalty = 0.0
    for pid in C.PUMP_IDS:
        flows = results.link["flowrate"][pid]
        for t in range(min(C.N_STEPS, len(flows))):
            q_m3h = abs(flows.iloc[t]) * 3600.0
            if q_m3h > 0.1:
                penalty += abs(q_m3h - C.Q_BEP) / C.Q_BEP * C.PEN_BEP
    return penalty

def loss_energy_vs_baseline(E_ga: float, E_baseline: float) -> float:
    """
    Linear energy penalty relative to the operator's planned-pressure baseline.
    E_baseline = epanet_energy(simulate operator's Step-2 schedule).
    Negative delta (GA saves energy vs operator's plan) reduces fitness.
    """
    return (E_ga - E_baseline) * C.ENERGY_WEIGHT

def loss_overpressure_vs_demand(P_schedule, Q_forecast) -> float:
    """Demand-proportional overpressure penalty (unchanged)."""
    WEIGHT_OVER, WEIGHT_UNDER = 80, 30
    Q_max   = np.max(Q_forecast)
    penalty = 0.0
    for t in range(C.N_STEPS):
        P_ref = C.P_OUTLET_MIN + (Q_forecast[t] / Q_max) * (C.P_OUTLET_MAX - C.P_OUTLET_MIN)
        diff  = P_schedule[t] - P_ref
        penalty += diff * WEIGHT_OVER if diff > 0 else abs(diff) * WEIGHT_UNDER
    return penalty

def fitness_evaluate(P_schedule, inp_path, Q_forecast, total_base_demand, E_baseline):
    """Full fitness → (soft_total, energy_kwh, pen_pressure, pen_bep)."""
    results, wn = simulate_schedule(inp_path, P_schedule, Q_forecast, total_base_demand)
    e   = epanet_energy(results)
    p_p = loss_pressure(results, wn)          # hard constraint G
    p_b = loss_bep(results, wn)
    p_e = loss_energy_vs_baseline(e, E_baseline)
    p_d = loss_overpressure_vs_demand(P_schedule, Q_forecast)
    return p_e + p_b + p_d, e, p_p, p_b      # p_p excluded from soft sum

# ── pymoo operators (unchanged from provided script) ─────────────────────────

def compare_winners(pop, P, **kwargs):
    """Tournament comparison with constraint domination (unchanged)."""
    S = np.full(P.shape[0], np.nan)
    for i, row in enumerate(P):
        best    = row[0]
        best_cv = pop[best].CV[0] if pop[best].CV is not None else np.inf
        best_f  = pop[best].F[0]  if pop[best].F  is not None else np.inf
        for j in row[1:]:
            cv = pop[j].CV[0] if pop[j].CV is not None else np.inf
            f  = pop[j].F[0]  if pop[j].F  is not None else np.inf
            if cv < best_cv:
                best = j; best_cv = cv; best_f = f
            elif cv == best_cv and f < best_f:
                best = j; best_f = f
        S[i] = best
    return S.astype(int)

class WarmSampling(Sampling):
    """90% random + 10% warm-seeded (Kazimipour et al., 2014) — unchanged."""
    def _do(self, problem, n_samples, **kwargs):
        pop    = np.random.uniform(C.P_OUTLET_MIN, C.P_OUTLET_MAX, (n_samples, C.N_STEPS))
        n_warm = int(n_samples * C.WARM_FRAC)
        for i in range(n_warm):
            p = np.full(C.N_STEPS, 4.0)
            p[0:6]   = np.random.uniform(C.P_OUTLET_MIN, 3.9, 6)
            p[6:9]   = np.random.uniform(3.9, 4.2,            3)
            p[9:17]  = np.random.uniform(3.8, 4.1,            8)
            p[17:21] = np.random.uniform(4.1, C.P_OUTLET_MAX, 4)
            p[21:]   = np.random.uniform(3.8, 4.1,            3)
            p += np.random.normal(0, 0.03, C.N_STEPS)
            pop[i]   = np.clip(p, C.P_OUTLET_MIN, C.P_OUTLET_MAX)
        return pop

class VitrukaScheduleProblem(Problem):
    """
    pymoo single-objective problem with hard pressure constraint.
    F  = soft_total (energy + BEP + overpressure)
    G  = loss_pressure (ДБН compliance; G > 0 → infeasible)
    Unchanged from provided script.
    """
    def __init__(self, inp_path, Q_forecast, total_base_demand, E_baseline):
        super().__init__(
            n_var=C.N_STEPS, n_obj=1, n_ieq_constr=1,
            xl=np.full(C.N_STEPS, C.P_OUTLET_MIN),
            xu=np.full(C.N_STEPS, C.P_OUTLET_MAX),
        )
        self.inp_path = inp_path; self.Q_forecast = Q_forecast
        self.total_base_demand = total_base_demand; self.E_baseline = E_baseline

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = np.zeros(len(X)), np.zeros(len(X))
        for i, schedule in enumerate(X):
            soft, e, p_p, p_b = fitness_evaluate(
                schedule, self.inp_path, self.Q_forecast,
                self.total_base_demand, self.E_baseline)
            F[i] = soft; G[i] = p_p
        out["F"] = F; out["G"] = G

# ── Streamlit progress callback ───────────────────────────────────────────────

class StreamlitProgressCallback(Callback):
    """
    Updates a Streamlit progress bar after each GA generation.
    stage_start / stage_end define the fraction of the 7-stage pipeline
    bar that this GA stage occupies (e.g. 0.57 → 0.86).
    """
    def __init__(self, bar, status_text, n_gen, stage_start, stage_end):
        super().__init__()
        self._bar    = bar
        self._status = status_text
        self._n_gen  = n_gen
        self._s0     = stage_start
        self._s1     = stage_end

    def notify(self, algorithm):
        gen  = algorithm.n_gen
        frac = self._s0 + (gen / self._n_gen) * (self._s1 - self._s0)
        if self._bar:
            self._bar.progress(min(frac, self._s1))
        if self._status:
            self._status.markdown(
                f"<div style='text-align:center; font-family:monospace; "
                f"font-size:16px; padding-bottom:20px;'>"
                f"Optimised Energy Consumption Determination — generation "
                f"<strong>{gen} / {self._n_gen}</strong></div>",
                unsafe_allow_html=True,
            )

# Main GA entry point

def run_ga(
    inp_path: str,
    Q_forecast,
    total_base_demand: float,
    E_baseline_kWh: float,
    progress_bar=None,
    status_text=None,
    stage_start: float = 0.57,
    stage_end:   float = 0.86,
):
    """
    Run the single-objective GA.
    Returns (best_schedule, best_fitness, (E_ga_kWh, pen_pressure, pen_bep)).
    Debug: POP_SIZE=80, N_GEN=30.
    """
    problem = VitrukaScheduleProblem(inp_path, Q_forecast, total_base_demand, E_baseline_kWh)

    callback = StreamlitProgressCallback(
        progress_bar, status_text, C.N_GEN, stage_start, stage_end
    )

    algorithm = GA(
        pop_size=C.POP_SIZE,
        sampling=WarmSampling(),
        selection=TournamentSelection(pressure=3, func_comp=compare_winners),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20, prob=0.15),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", C.N_GEN),
        callback=callback,
        seed=42,
        verbose=False,
    )

    # ── Result extraction (unchanged from provided script) ────────────────────
    if res.X is not None:
        best_schedule = res.X
        best_fitness  = res.F[0]
    else:
        # No feasible solution — return least-infeasible chromosome
        pop   = res.pop
        cv    = np.array([ind.CV[0] if ind.CV is not None else np.inf for ind in pop])
        f_val = np.array([ind.F[0]  if ind.F  is not None else np.inf for ind in pop])
        min_cv = cv.min()
        best_idx = np.where(cv <= min_cv + 1e-6, f_val, np.inf).argmin()
        best_schedule = pop[best_idx].X
        best_fitness  = f_val[best_idx]

    # Final re-evaluation for component breakdown
    soft_total, e, p_p, p_b = fitness_evaluate(
        best_schedule, inp_path, Q_forecast, total_base_demand, E_baseline_kWh
    )

    if progress_bar:
        progress_bar.progress(stage_end)

    return best_schedule, best_fitness, (e, p_p, p_b)