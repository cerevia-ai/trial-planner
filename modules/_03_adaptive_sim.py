# 03_adaptive_sim.py
# Streamlit app: Adaptive Trial Simulator — continuous or event-driven endpoints

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from typing import List
import time

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

# -----------------------------
# Alpha spending (Lan-DeMets approximations)
# -----------------------------
def obrien_fleming_spend(t: float) -> float:
    """Approximate O'Brien-Fleming alpha spending"""
    if t <= 0:
        return 0.0
    z_alpha = norm.ppf(1 - 1e-12)
    return 2 * (1 - norm.cdf(z_alpha / np.sqrt(t)))


def pocock_spend(t: float, alpha=0.05) -> float:
    """Approximate Pocock alpha spending"""
    return alpha * t


def lan_demets_spend(t: float, method="obf", alpha=0.05) -> float:
    t = np.clip(t, 1e-9, 1.0)
    if method == "obf":
        z_a = norm.ppf(1 - alpha / 2)
        spend = alpha * (1 - norm.cdf(z_a / np.sqrt(t)))
    else:
        spend = alpha * t
    return float(np.clip(spend, 0.0, alpha))


def boundaries_from_spending(info_frac: List[float], alpha: float, method: str) -> List[float]:
    """Compute two-sided z-critical values for each interim based on cumulative alpha spending"""
    zs = []
    for t in info_frac:
        a_cum = lan_demets_spend(t, method=method, alpha=alpha)
        a_cum = min(max(a_cum, 1e-12), alpha - 1e-12) if t < 1.0 else alpha
        zs.append(float(norm.ppf(1 - a_cum / 2.0)))
    return zs


# -----------------------------
# Adaptive trial design dataclass
# -----------------------------
@dataclass
class AdaptiveDesign:
    n_max_per_group: int
    n_analyses: int
    info_frac: List[float]
    alpha: float
    spend_method: str
    futility_frac: float
    futility_enabled: bool


# -----------------------------
# Simulation functions
# -----------------------------
def simulate_continuous_trial(design: AdaptiveDesign, n_sim: int, enrollment_rate_per_month: float,
                              mean_control: float, effect_size_mean_diff: float, sigma: float, seed: int = None):
    rng = np.random.default_rng(seed)
    n_per_group = design.n_max_per_group
    z_bounds = boundaries_from_spending(design.info_frac, design.alpha, design.spend_method)

    success_flags = np.zeros(n_sim, dtype=bool)
    stop_at = np.full(n_sim, design.n_analyses, dtype=int)
    sample_sizes = np.zeros(n_sim, dtype=int)

    for sim in range(n_sim):
        control = rng.normal(mean_control, sigma, n_per_group)
        treated = rng.normal(mean_control + effect_size_mean_diff, sigma, n_per_group)

        for k, t in enumerate(design.info_frac):
            m = max(2, int(np.ceil(n_per_group * t)))
            x_ctrl, x_trt = control[:m], treated[:m]
            mean_diff = np.mean(x_trt) - np.mean(x_ctrl)
            s_p = np.sqrt(((m - 1) * np.var(x_trt, ddof=1) + (m - 1) * np.var(x_ctrl, ddof=1)) / (2 * (m - 1)))
            se = s_p * np.sqrt(2.0 / m) if s_p > 0 else 1e-9
            z = mean_diff / se
            z_crit = z_bounds[k]
            fut_z = -abs(design.futility_frac * z_crit) if design.futility_enabled else None

            if z > z_crit:
                success_flags[sim], stop_at[sim], sample_sizes[sim] = True, k, 2 * m
                break
            if fut_z is not None and z < fut_z:
                success_flags[sim], stop_at[sim], sample_sizes[sim] = False, k, 2 * m
                break
        else:
            stop_at[sim], sample_sizes[sim] = design.n_analyses - 1, 2 * n_per_group
            z_final = (np.mean(treated) - np.mean(control)) / (np.std(np.concatenate([treated, control]), ddof=1) * np.sqrt(2 / n_per_group))
            success_flags[sim] = z_final > z_bounds[-1]

    return {"success": success_flags, "stop_at": stop_at, "sample_size": sample_sizes}


def simulate_event_driven_trial(design: AdaptiveDesign, n_sim: int, enrollment_rate_per_month: float,
                                hazard_control: float, hazard_ratio: float, max_follow_up_months: float, seed: int = None):
    rng = np.random.default_rng(seed)
    n_per_group = design.n_max_per_group
    n_total = n_per_group * 2
    info_fracs = design.info_frac
    z_bounds = boundaries_from_spending(info_fracs, design.alpha, design.spend_method)

    success_flags = np.zeros(n_sim, dtype=bool)
    stop_at = np.full(n_sim, design.n_analyses, dtype=int)
    sample_sizes = np.zeros(n_sim, dtype=int)
    total_events = np.zeros(n_sim, dtype=int)

    avg_follow_yrs = max_follow_up_months / 12.0

    for sim in range(n_sim):
        # accrual times
        accrual_times = np.zeros(n_total) if enrollment_rate_per_month <= 0 else np.cumsum(
            rng.exponential(1.0 / enrollment_rate_per_month, n_total))
        accr_ctrl, accr_trt = accrual_times[:n_per_group], accrual_times[n_per_group:]

        ev_ctrl = rng.exponential(1.0 / hazard_control, n_per_group)
        ev_trt = rng.exponential(1.0 / (hazard_control * hazard_ratio), n_per_group)

        accr_ctrl_yrs, accr_trt_yrs = accr_ctrl / 12.0, accr_trt / 12.0
        event_ctrl = accr_ctrl_yrs + ev_ctrl
        event_trt = accr_trt_yrs + ev_trt

        censor_time = np.max(accrual_times) / 12.0 + avg_follow_yrs
        observed_time = np.minimum(np.concatenate([event_ctrl, event_trt]), censor_time)
        observed_event = ((np.concatenate([event_ctrl, event_trt]) <= censor_time)).astype(int)
        arms = np.array([0] * n_per_group + [1] * n_per_group)

        cum_events = np.cumsum(observed_event[np.argsort(observed_time)])
        total_final_events = int(np.sum(observed_event))
        if total_final_events < 1:
            sample_sizes[sim] = n_total
            continue

        for k, t in enumerate(info_fracs):
            target_ev = int(np.ceil(t * total_final_events))
            idx = np.searchsorted(cum_events, target_ev)
            analysis_time = observed_time[idx] if idx < len(observed_time) else censor_time
            mask = observed_time <= analysis_time

            e_ctrl = np.sum(observed_event[(arms == 0) & mask])
            e_trt = np.sum(observed_event[(arms == 1) & mask])
            n_ctrl_obs, n_trt_obs = np.sum(arms == 0), np.sum(arms == 1)
            p1, p2 = e_ctrl / n_ctrl_obs, e_trt / n_trt_obs
            p_bar = (e_ctrl + e_trt) / (n_ctrl_obs + n_trt_obs)
            se = np.sqrt(p_bar * (1 - p_bar) * (1 / n_ctrl_obs + 1 / n_trt_obs))
            z = (p2 - p1) / (se if se > 0 else 1e-9)
            z_crit = z_bounds[k]
            fut_z = -abs(design.futility_frac * z_crit) if design.futility_enabled else None

            if z > z_crit:
                success_flags[sim], stop_at[sim], sample_sizes[sim] = True, k, n_total
                total_events[sim] = np.sum(observed_event[mask])
                break
            if fut_z is not None and z < fut_z:
                success_flags[sim], stop_at[sim], sample_sizes[sim] = False, k, n_total
                total_events[sim] = np.sum(observed_event[mask])
                break
        else:
            stop_at[sim], sample_sizes[sim] = design.n_analyses - 1, n_total
            success_flags[sim] = (np.sum(observed_event[arms == 1]) / n_per_group - np.sum(observed_event[arms == 0]) / n_per_group) > z_bounds[-1]

    return {"success": success_flags, "stop_at": stop_at, "sample_size": sample_sizes, "total_events": total_events}

def run():
    st.title("Adaptive Trial Simulator — group-sequential with interim stopping")
    st.markdown(
        "Simulate adaptive trials with interim analyses, early stopping for efficacy/futility, "
        "and visualize boundaries, probability of success, and average sample size."
    )

    # --- Two-column layout ---
    col1, col2 = st.columns([2, 1])  # left: results, right: settings

    # --- Settings column (was sidebar) ---
    with col2:
        with st.expander("Simulation settings", expanded=True):
            endpoint = st.selectbox("Primary endpoint", ["Continuous biomarker", "Event-driven (time-to-event)"])
            n_sim = st.number_input("Number of Monte Carlo simulations", 200, 20000, 2000, step=100)
            seed = st.number_input("RNG seed (0=random)", 0, 9999999, 0)
            n_analyses = st.selectbox("Number of analyses (incl. final)", [2, 3, 4, 5], index=2)
            default_fracs = np.linspace(1.0 / n_analyses, 1.0, n_analyses).tolist()
            info_frac = st.text_input("Cumulative information fractions (comma-separated)",
                                      value=",".join([f"{f:.2f}" for f in default_fracs]))
            try:
                info_frac_list = [float(x.strip()) for x in info_frac.split(",")]
                if len(info_frac_list) != n_analyses:
                    info_frac_list = default_fracs
            except Exception:
                info_frac_list = default_fracs
            spend_method = st.selectbox("Alpha-spending approximation", ["obf", "pocock"], index=0)
            alpha = st.number_input("Overall two-sided alpha", 0.001, 0.10, 0.05, 0.001, "%.3f")
            futility_enabled = st.checkbox("Enable futility stopping", True)
            futility_frac = st.slider("Futility fraction of efficacy boundary", 0.0, 1.0, 0.5, 0.05)

        with st.expander("Design & enrollment", expanded=False):
            n_per_group_plan = st.number_input("Planned sample size per group (max)", 10, 20000, 500, step=10)
            enrollment_rate = st.number_input("Enrollment rate (subjects/month total)", 1.0, 1000.0, 50.0, step=1.0)
            max_follow_up_months = st.number_input("Average follow-up (months, event-driven)", 1, 120, 36, step=1)

        with st.expander("Endpoint parameters", expanded=False):
            if endpoint == "Continuous biomarker":
                mean_control = st.number_input("Control mean", value=0.0, step=0.1)
                mean_diff = st.number_input("True mean difference (treatment - control)", value=0.2, step=0.05)
                sigma = st.number_input("Standard deviation", value=1.0, step=0.1)
            else:
                annual_event_rate_control = st.number_input("Control annual event rate (fraction)", 0.001, 1.0, 0.10, step=0.005)
                hazard_control = -np.log(1 - annual_event_rate_control)
                hazard_ratio = st.number_input("True hazard ratio (treatment/control)", 0.1, 5.0, 0.8, 0.05)

    # --- Results column ---
    with col1:
        if st.button("Run simulation"):
            start_time = time.time()
            design = AdaptiveDesign(
                n_max_per_group=int(n_per_group_plan),
                n_analyses=len(info_frac_list),
                info_frac=info_frac_list,
                alpha=float(alpha),
                spend_method=spend_method,
                futility_frac=float(futility_frac),
                futility_enabled=bool(futility_enabled),
            )
            rng_seed = None if seed == 0 else int(seed)

            with st.spinner("Simulating..."):
                if endpoint == "Continuous biomarker":
                    res = simulate_continuous_trial(design, int(n_sim), float(enrollment_rate),
                                                    float(mean_control), float(mean_diff), float(sigma), seed=rng_seed)
                else:
                    res = simulate_event_driven_trial(design, int(n_sim), float(enrollment_rate),
                                                      float(hazard_control), float(hazard_ratio), float(max_follow_up_months), seed=rng_seed)

            elapsed = time.time() - start_time
            success_rate = np.mean(res["success"])
            avg_sample_size = np.mean(res["sample_size"])
            stop_counts = pd.Series(res["stop_at"]).value_counts().sort_index()

            st.success(f"Simulation done in {elapsed:.1f}s — estimated probability of success: {success_rate:.3%}")
            st.metric("Estimated probability of success", f"{success_rate:.3%}")
            st.metric("Average sample size (per sim)", f"{avg_sample_size:.1f}")

            # Stopping distribution
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            stops = stop_counts.reindex(range(design.n_analyses), fill_value=0)
            ax1.bar([f"A{idx+1}" for idx in stops.index], stops.values)
            ax1.set_xlabel("Analysis")
            ax1.set_ylabel("Number of sims stopped")
            st.pyplot(fig1)

            # Boundaries
            z_boundaries = boundaries_from_spending(info_frac_list, alpha, spend_method)
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(info_frac_list, z_boundaries, marker="o", label="Efficacy z-boundary")
            if futility_enabled:
                ax2.plot(info_frac_list, [-futility_frac * z for z in z_boundaries], marker="o", label="Futility boundary")
            ax2.set_xlabel("Information fraction")
            ax2.set_ylabel("z-critical value")
            ax2.set_title("Stopping boundaries")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

    add_footer()
