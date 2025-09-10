# 05_endpoint_feasibility.py
# Streamlit: Endpoint Feasibility Explorer for CVD trials
#
# Provides event-rate prediction, sample-size calculations, enrollment/follow-up estimates,
# budget feasibility check, and composite endpoint suggestions.

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from statsmodels.stats.power import zt_ind_solve_power, NormalIndPower
import math

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

# -----------------------------
# Utility functions
# -----------------------------
def _effect_size_from_rates(p1: float, p2: float) -> float:
    """
    Cohen's h-like effect for proportions (two-sample z-test approximation).
    (p1 - p2) / sqrt(p_avg*(1-p_avg))
    """
    p_avg = (p1 + p2) / 2.0
    denom = max(p_avg * (1 - p_avg), 1e-9)
    return (p1 - p2) / np.sqrt(denom)


def _sample_size_for_proportions(p_control: float, p_treat: float, alpha: float, power: float) -> int:
    """
    Approximate required sample size per group for two-sample comparison of proportions
    using z-test approximation via statsmodels' zt_ind_solve_power on effect size.
    """
    eff = _effect_size_from_rates(p_control, p_treat)
    # zt_ind_solve_power expects effect_size and returns nobs1
    try:
        n = zt_ind_solve_power(effect_size=eff, alpha=alpha, power=power, alternative="two-sided")
        return int(np.ceil(n))
    except Exception:
        # fallback: use NormalIndPower
        analysis = NormalIndPower()
        try:
            n = analysis.solve_power(effect_size=eff, power=power, alpha=alpha, alternative='two-sided')
            return int(np.ceil(n))
        except Exception:
            return -1


def _sample_size_for_continuous(delta_mean: float, sigma: float, alpha: float, power: float) -> int:
    """
    Two-sample t (large-sample z) approximate sample size per group for continuous endpoint.
    Uses NormalIndPower.
    """
    analysis = NormalIndPower()
    # effect size (Cohen's d) = delta / sigma
    if sigma <= 0:
        return -1
    d = float(delta_mean) / float(sigma)
    try:
        n = analysis.solve_power(effect_size=d, power=power, alpha=alpha, alternative='two-sided')
        return int(np.ceil(n))
    except Exception:
        return -1


def _annual_to_multi_year(p_annual, years, dropout_rate=0.0):
    """
    Convert annual event probability to multi-year probability assuming constant hazard.
    Approximate effective follow-up with linear dropout: t_eff = years * (1 - dropout_rate/2).
    Works with scalars or arrays.
    """
    p_annual = np.clip(p_annual, 1e-6, 0.95)
    hazard = -np.log(1 - p_annual)            # scalar or vector
    t_eff = years * (1 - dropout_rate / 2.0)
    return 1 - np.exp(-hazard * t_eff)        # array-safe



def _suggest_composites(event_rates: Dict[str, float]) -> Dict[str, float]:
    """
    Suggest simple composites by union of individual event endpoints.
    For two endpoints A and B with rates pA and pB (assuming independence),
    p(A or B) = 1 - (1-pA)(1-pB).
    We'll produce pairwise composites and a 'triple' if available.
    """
    keys = list(event_rates.keys())
    composites = {}
    # pairwise
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            p = 1 - (1 - event_rates[a]) * (1 - event_rates[b])
            composites[f"{a} ∪ {b}"] = p
    # triple (if at least 3)
    if len(keys) >= 3:
        from itertools import combinations

        for comb in combinations(keys, 3):
            probs = [event_rates[k] for k in comb]
            p = 1 - np.prod([1 - x for x in probs])
            composites[" ∪ ".join(comb)] = p
    return composites


# -----------------------------
# Simple risk model for event-rate prediction (transparent)
# -----------------------------
@dataclass
class RiskCoefs:
    intercept: float = -2.2
    age_per10: float = 0.35
    male: float = 0.25
    smoker: float = 0.55
    diabetes: float = 0.50
    htn: float = 0.30
    ldl_per30: float = 0.18
    hdl_per10_inv: float = 0.25
    sbp_per10: float = 0.12


def _annual_event_prob_from_baseline_row(row: dict, coefs: RiskCoefs) -> float:
    x = (
        coefs.intercept
        + coefs.age_per10 * ((row["age"] - 60) / 10.0)
        + coefs.male * (1 if row["sex"] in [1, "M", "Male"] else 0)
        + coefs.smoker * (1 if row.get("smoker", 0) else 0)
        + coefs.diabetes * (1 if row.get("diabetes", 0) else 0)
        + coefs.htn * (1 if row.get("htn", 0) else 0)
        + coefs.ldl_per30 * ((row.get("ldl", 130) - 130) / 30.0)
        + coefs.hdl_per10_inv * ((50 - row.get("hdl", 50)) / 10.0)
        + coefs.sbp_per10 * ((row.get("sbp", 130) - 130) / 10.0)
    )
    prob = 1.0 / (1.0 + math.exp(-x))
    return float(np.clip(prob, 1e-6, 0.99))


def predict_event_rate_from_baseline(df_baseline: pd.DataFrame, follow_up_years: float, dropout_rate: float = 0.0) -> float:
    """
    Given a dataframe of baseline covariates (trial population), predict multi-year event probability averaged across individuals.
    Expected columns: age, sex, smoker, diabetes, htn, ldl, hdl, sbp
    """
    coefs = RiskCoefs()
    # convert to dict rows
    rows = df_baseline.to_dict(orient="records")
    p_ann = np.array([_annual_event_prob_from_baseline_row(r, coefs) for r in rows])
    p_multi = _annual_to_multi_year(p_ann, follow_up_years, dropout_rate)
    return float(np.mean(p_multi))


# -----------------------------
# Streamlit UI
# -----------------------------
def run():
    st.title("Endpoint Feasibility Explorer")
    st.markdown(
        "Estimate expected event rates, sample size needs, enrollment/follow-up timelines, and budget feasibility for common CVD endpoints. "
        "If a single endpoint is underpowered, the app will suggest composite endpoints and re-calc operating characteristics."
    )

    # LEFT: inputs
    left, right = st.columns([1, 1.2])

    with left:
        st.header("1) Endpoint & population")
        endpoint = st.selectbox("Choose primary endpoint", [
            "All-cause mortality (event)",
            "Hospitalization for CV cause (event)",
            "BP reduction (continuous)",
            "LDL change (continuous)"
        ])

        st.markdown("**Provide baseline population** (choose one):")
        baseline_option = st.radio("Baseline input", ["Upload CSV (patient-level)", "Enter cohort summary (quick)"], index=1)
        trial_baseline_df = None
        if baseline_option.startswith("Upload"):
            bl_file = st.file_uploader("Upload baseline CSV (columns: age, sex(0/1), smoker(0/1), diabetes(0/1), htn(0/1), ldl, hdl, sbp)", type=["csv"])
            if bl_file is not None:
                trial_baseline_df = pd.read_csv(bl_file)
                st.success(f"Loaded {len(trial_baseline_df)} baseline rows.")
        else:
            n_demo = st.number_input("Planned sample size (for baseline summary calculations)", min_value=10, max_value=200000, value=500)
            age_mean = st.number_input("Age mean", min_value=30, max_value=90, value=60)
            pct_male = st.slider("Pct male", 0.0, 1.0, 0.50)
            pct_smoker = st.slider("Pct current smoker", 0.0, 1.0, 0.15)
            pct_dm = st.slider("Pct diabetes", 0.0, 1.0, 0.10)
            pct_htn = st.slider("Pct hypertension", 0.0, 1.0, 0.30)
            ldl_mean = st.number_input("LDL mean (mg/dL)", min_value=50, max_value=250, value=130)
            hdl_mean = st.number_input("HDL mean (mg/dL)", min_value=20, max_value=90, value=50)
            sbp_mean = st.number_input("SBP mean (mmHg)", min_value=90, max_value=200, value=130)

            # create summary DF for prediction
            trial_baseline_df = pd.DataFrame({
                "age": np.round(np.full(n_demo, age_mean)),
                "sex": np.round(np.full(n_demo, pct_male)),  # treat as fraction -> predict interprets 1 for male
                "smoker": (np.random.RandomState(0).random(n_demo) < pct_smoker).astype(int),
                "diabetes": (np.random.RandomState(1).random(n_demo) < pct_dm).astype(int),
                "htn": (np.random.RandomState(2).random(n_demo) < pct_htn).astype(int),
                "ldl": np.full(n_demo, ldl_mean),
                "hdl": np.full(n_demo, hdl_mean),
                "sbp": np.full(n_demo, sbp_mean),
            })

        st.header("2) Trial assumptions")
        follow_up_years = st.number_input("Planned follow-up (years)", min_value=0.25, max_value=10.0, value=3.0, step=0.25)
        dropout_rate = st.slider("Annual dropout (fraction)", 0.0, 0.5, 0.10, step=0.01)
        alpha = st.number_input("Alpha (two-sided)", min_value=0.001, max_value=0.2, value=0.05, step=0.001, format="%.3f")
        power = st.number_input("Target power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
        enrollment_rate_per_month = st.number_input("Planned enrollment rate (subjects / month total)", min_value=1.0, max_value=5000.0, value=50.0, step=1.0)
        per_subject_cost = st.number_input("Estimated per-subject total cost (USD)", min_value=100.0, max_value=200000.0, value=5000.0, step=100.0)
        total_budget = st.number_input("Available trial budget (USD)", min_value=1000.0, max_value=1e9, value=2_000_000.0, step=1000.0)

        st.markdown("---")
        st.header("3) Effect assumptions")
        if endpoint.endswith("(event)"):
            effect_mode = st.selectbox("Effect specification", ["Relative risk reduction (%)", "Absolute risk difference (percentage points)"])
            if effect_mode.startswith("Relative"):
                rrr = st.number_input("Assumed relative risk reduction (%)", min_value=0.0, max_value=100.0, value=20.0)
            else:
                ard_pp = st.number_input("Assumed absolute risk difference (percentage points)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        else:
            # continuous
            delta_mean = st.number_input("Assumed true mean difference (treatment - control)", value=0.2, step=0.05)
            sigma = st.number_input("Endpoint SD (assumed)", value=1.0, step=0.1)

        run_button = st.button("Estimate feasibility")


    with right:
        st.header("Results & suggestions")
        placeholder = st.empty()

    if run_button:
        # Predict control event rates or control means
        if endpoint == "All-cause mortality (event)":
            # if baseline file present, use risk model; else ask for direct control annual rate
            if trial_baseline_df is not None:
                p_control_multi = predict_event_rate_from_baseline(trial_baseline_df, follow_up_years, dropout_rate)
                # convert to annual approximate
                # invert multi-year to per-year approx: hazard = -ln(1-p_multi)/t_eff -> p_ann = 1 - exp(-hazard)
                t_eff = follow_up_years * (1 - dropout_rate / 2.0)
                hazard = -np.log(1 - p_control_multi + 1e-12) / max(t_eff, 1e-9)
                p_control_annual = 1 - np.exp(-hazard * 1.0)
            else:
                p_control_annual = st.number_input("Control annual event rate (if no baseline file)", min_value=0.001, max_value=1.0, value=0.05)
                p_control_multi = _annual_to_multi_year_prob(p_control_annual, follow_up_years, dropout_rate)
        elif endpoint == "Hospitalization for CV cause (event)":
            if trial_baseline_df is not None:
                p_control_multi = predict_event_rate_from_baseline(trial_baseline_df, follow_up_years, dropout_rate) * 1.4  # assume more frequent than mortality
                p_control_multi = min(p_control_multi, 0.5)
                t_eff = follow_up_years * (1 - dropout_rate / 2.0)
                hazard = -np.log(1 - p_control_multi + 1e-12) / max(t_eff, 1e-9)
                p_control_annual = 1 - np.exp(-hazard * 1.0)
            else:
                p_control_annual = st.number_input("Control annual hospitalization rate", min_value=0.001, max_value=1.0, value=0.10)
                p_control_multi = _annual_to_multi_year_prob(p_control_annual, follow_up_years, dropout_rate)
        elif endpoint == "BP reduction (continuous)":
            # For continuous endpoints we report required sample size per-group for the specified delta & sigma
            # For 'control mean' we don't need to predict — show default effect size guidance.
            control_mean = st.number_input("Control mean BP (mmHg)", value=140.0)
            # convert mmHg change etc.
        elif endpoint == "LDL change (continuous)":
            control_mean = st.number_input("Control mean LDL (mg/dL)", value=130.0)

        # compute sample size and feasibility
        results = {}
        if endpoint.endswith("(event)"):
            # determine p_control_multi and p_exp
            if effect_mode.startswith("Relative"):
                p_control = float(p_control_multi)
                p_exp = p_control * (1 - rrr / 100.0)
            else:
                p_control = float(p_control_multi)
                p_exp = max(0.0, p_control - (ard_pp / 100.0))

            n_per_group_req = _sample_size_for_proportions(p_control, p_exp, alpha=alpha, power=power)
            if n_per_group_req < 0:
                st.error("Could not compute required sample size with these parameters.")
                n_per_group_req = None

            # enrollment duration
            if n_per_group_req:
                total_required = 2 * n_per_group_req
                enroll_months = max(1.0, total_required / max(1.0, enrollment_rate_per_month))
                enroll_years = enroll_months / 12.0
            else:
                total_required = None
                enroll_years = None

            # budget feasibility
            feasible_by_budget = False
            if total_required is not None:
                est_cost = total_required * per_subject_cost
                feasible_by_budget = est_cost <= total_budget

            results = {
                "p_control_multi": p_control,
                "p_exp_multi": p_exp,
                "n_per_group_req": n_per_group_req,
                "total_required": total_required,
                "enroll_years": enroll_years,
                "est_cost": est_cost if total_required is not None else None,
                "feasible_by_budget": feasible_by_budget,
            }
        else:
            # continuous endpoints
            n_req = _sample_size_for_continuous(delta_mean, sigma, alpha=alpha, power=power)
            total_required = None
            enroll_years = None
            est_cost = None
            feasible_by_budget = False
            if n_req > 0:
                total_required = 2 * n_req
                enroll_months = max(1.0, total_required / max(1.0, enrollment_rate_per_month))
                enroll_years = enroll_months / 12.0
                est_cost = total_required * per_subject_cost
                feasible_by_budget = est_cost <= total_budget

            results = {
                "n_per_group_req": n_req,
                "total_required": total_required,
                "enroll_years": enroll_years,
                "est_cost": est_cost,
                "feasible_by_budget": feasible_by_budget,
            }

        # show results
        with placeholder.container():
            st.subheader("Feasibility summary")
            if endpoint.endswith("(event)"):
                st.metric("Modeled control multi-year event rate", f"{results['p_control_multi']:.2%}")
                st.metric("Modeled experimental multi-year event rate", f"{results['p_exp_multi']:.2%}")
                st.write(f"Required sample size per group: **{results['n_per_group_req']:,}**")
                st.write(f"Total required subjects (both arms): **{results['total_required']:,}**")
                st.write(f"Estimated enrollment time: **{results['enroll_years']:.1f} years** (at {enrollment_rate_per_month:.1f} subj/month)")
                st.write(f"Estimated trial cost: **${results['est_cost']:,.0f}** (per-subject ${per_subject_cost:,.0f})")
                st.write("Feasible by available budget:" , "✅ Yes" if results["feasible_by_budget"] else "❌ No")
            else:
                st.metric("Required sample size per group", f"{results['n_per_group_req']:,}")
                if results["total_required"] is not None:
                    st.write(f"Total required subjects: **{results['total_required']:,}**")
                    st.write(f"Estimated enrollment time: **{results['enroll_years']:.1f} years**")
                    st.write(f"Estimated trial cost: **${results['est_cost']:,.0f}**")
                    st.write("Feasible by available budget:" , "✅ Yes" if results["feasible_by_budget"] else "❌ No")

            st.markdown("---")
            st.subheader("If underpowered: composite endpoint suggestions")
            # For event endpoints only: suggest composites of mortality + hospitalization
            if endpoint.endswith("(event)"):
                baseline_events = {"All-cause mortality": results["p_control_multi"]}
                # if hospitalization option exists, approximate its rate
                # We'll produce some simple suggestions including adding hospitalization (assume higher rate) if user didn't choose it
                # Here: compute hypothetical hospitalization rate from baseline (slightly higher)
                hosp_rate = min(results["p_control_multi"] * 1.4, 0.5)
                baseline_events["Hospitalization"] = hosp_rate
                comp = _suggest_composites(baseline_events)
                comp_df = pd.DataFrame({"composite": list(comp.keys()), "estimated_multi_year_rate": list(comp.values())})
                comp_df["required_n_per_group_if_same_rrr"] = comp_df["estimated_multi_year_rate"].apply(
                    lambda p: _sample_size_for_proportions(p, max(0.0, p * (1 - rrr / 100.0)) , alpha=alpha, power=power)
                )
                st.dataframe(comp_df)
                # highlight suggestions where required size drops substantially (e.g., >25% reduction)
                if results["n_per_group_req"] and results["n_per_group_req"] > 0:
                    comp_df["reduction_pct"] = 100 * (results["n_per_group_req"] - comp_df["required_n_per_group_if_same_rrr"]) / results["n_per_group_req"]
                    st.write("Composites offering largest required-sample reduction:")
                    st.dataframe(comp_df.sort_values("reduction_pct", ascending=False).head(5))
            else:
                st.write("Composite endpoints typically apply to event endpoints. For continuous endpoints consider composite scores (z-scores) combining endpoints — ask me to add examples.")

            st.markdown("---")
            st.subheader("Quick visualization")
            fig, ax = plt.subplots(figsize=(6, 3))
            if endpoint.endswith("(event)"):
                # show control vs experimental event rates
                labels = ["Control", "Experimental"]
                vals = [results["p_control_multi"], results["p_exp_multi"]]
                ax.bar(labels, vals)
                ax.set_ylabel(f"Multi-year event rate (over {follow_up_years} yrs)")
                ax.set_ylim(0, max(0.6, max(vals) * 1.4))
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.01, f"{v:.2%}", ha="center", va="bottom")
            else:
                # show effect size (Cohen's d)
                d = delta_mean / sigma if sigma > 0 else 0.0
                ax.bar(["Cohen's d"], [d])
                ax.set_ylabel("Effect size (Cohen's d)")
                ax.text(0, d + 0.01, f"{d:.2f}", ha="center")
            ax.set_title("Feasibility visualization")
            plt.tight_layout()
            st.pyplot(fig)

            st.info("Notes: This explorer uses transparent approximations. For event-driven time-to-event trials you may want to use dedicated log-rank/event-count sample-size formulas and a full accrual + censoring model. Always confirm with a statistician and regulatory guidance for final designs.")

            # allow download of a short summary CSV
            summary = {
                "endpoint": endpoint,
                "follow_up_years": follow_up_years,
                "dropout_rate": dropout_rate,
                "alpha": alpha,
                "power": power,
                "n_per_group_required": results.get("n_per_group_req"),
                "total_required": results.get("total_required"),
                "enrollment_years_est": results.get("enroll_years"),
                "estimated_cost_usd": results.get("est_cost"),
                "feasible_by_budget": results.get("feasible_by_budget"),
            }
            summary_df = pd.DataFrame([summary])
            st.download_button("Download summary (CSV)", data=summary_df.to_csv(index=False).encode(), file_name="feasibility_summary.csv", mime="text/csv")

    add_footer()

if __name__ == "__main__":
    run()