# utils/ui_helpers.py
import streamlit as st
import numpy as np

# -----------------------------
# Population Inputs
# -----------------------------
def population_inputs(defaults=None, expander_title="🧬 Population Assumptions"):
    """
    Returns a dict of population parameters from Streamlit UI.
    'defaults' can override any of the default values.
    'expander_title' customizes the Streamlit expander header.
    """
    if defaults is None:
        defaults = {}

    with st.expander(expander_title, expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            n_cohort = st.number_input(
                "Simulated cohort size",
                min_value=500, max_value=200000,
                value=defaults.get("n_cohort", 5000),
                step=500
            )
            follow_up = st.slider(
                "Follow-up duration (years)",
                0.5, 10.0,
                value=defaults.get("follow_up", 3.0),
                step=0.5
            )
            dropout = st.slider(
                "Annual dropout rate (fraction)",
                0.0, 0.5,
                value=defaults.get("dropout", 0.10),
                step=0.01
            )
            seed = st.number_input(
                "Random seed",
                min_value=0, max_value=999999,
                value=defaults.get("seed", 42)
            )

        with col2:
            age_mean = st.number_input(
                "Age mean", min_value=18, max_value=90,
                value=defaults.get("age_mean", 60)
            )
            age_sd = st.number_input(
                "Age SD", min_value=1, max_value=20,
                value=defaults.get("age_sd", 8)
            )
            pct_male = st.slider(
                "Pct male", 0.0, 1.0,
                value=defaults.get("pct_male", 0.5), step=0.01
            )
            pct_smoker = st.slider(
                "Pct current smoker", 0.0, 1.0,
                value=defaults.get("pct_smoker", 0.15), step=0.01
            )
            pct_dm = st.slider(
                "Pct diabetes", 0.0, 1.0,
                value=defaults.get("pct_dm", 0.10), step=0.01
            )

        with col3:
            pct_htn = st.slider(
                "Pct hypertension", 0.0, 1.0,
                value=defaults.get("pct_htn", 0.30), step=0.01
            )
            ldl_mean = st.number_input(
                "LDL mean (mg/dL)", 50, 250,
                value=defaults.get("ldl_mean", 130)
            )
            ldl_sd = st.number_input(
                "LDL SD", 1, 80,
                value=defaults.get("ldl_sd", 30)
            )
            hdl_mean = st.number_input(
                "HDL mean (mg/dL)", 20, 90,
                value=defaults.get("hdl_mean", 50)
            )
            hdl_sd = st.number_input(
                "HDL SD", 1, 30,
                value=defaults.get("hdl_sd", 12)
            )
            sbp_mean = st.number_input(
                "SBP mean (mmHg)", 90, 200,
                value=defaults.get("sbp_mean", 130)
            )
            sbp_sd = st.number_input(
                "SBP SD", 1, 50,
                value=defaults.get("sbp_sd", 15)
            )

    return {
        "n_cohort": n_cohort,
        "follow_up": follow_up,
        "dropout": dropout,
        "seed": seed,
        "age_mean": age_mean,
        "age_sd": age_sd,
        "pct_male": pct_male,
        "pct_smoker": pct_smoker,
        "pct_dm": pct_dm,
        "pct_htn": pct_htn,
        "ldl_mean": ldl_mean,
        "ldl_sd": ldl_sd,
        "hdl_mean": hdl_mean,
        "hdl_sd": hdl_sd,
        "sbp_mean": sbp_mean,
        "sbp_sd": sbp_sd,
    }

# -----------------------------
# Trial Design Inputs
# -----------------------------
def trial_design_inputs(defaults=None, expander_title="📐 Trial Design & Effect Specification"):
    """
    Returns a dict of trial design parameters from Streamlit UI.
    'defaults' can override default values.
    'expander_title' customizes the Streamlit expander header.
    """
    if defaults is None:
        defaults = {}

    with st.expander(expander_title, expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            alpha = st.number_input(
                "Alpha (two-sided)",
                min_value=0.001, max_value=0.20,
                value=defaults.get("alpha", 0.05),
                step=0.001, format="%.3f"
            )
            power_target = st.number_input(
                "Target power",
                min_value=0.5, max_value=0.99,
                value=defaults.get("power_target", 0.80),
                step=0.01
            )
            control_override = st.checkbox(
                "Override control event rate manually",
                value=defaults.get("control_override", False)
            )
            control_manual = None
            if control_override:
                control_manual = st.number_input(
                    "Control event rate (multi-year fraction)",
                    min_value=0.0001, max_value=0.9999,
                    value=defaults.get("control_manual", 0.10),
                    step=0.001
                )

        with col2:
            effect_mode = st.selectbox(
                "Effect specification",
                ["Relative risk reduction (%)", "Absolute risk difference (pp)"],
                index=0 if defaults.get("effect_mode") is None else
                      (0 if defaults.get("effect_mode") == "Relative risk reduction (%)" else 1)
            )
            rrr = ard = None
            if effect_mode == "Relative risk reduction (%)":
                rrr = st.number_input(
                    "Assumed relative risk reduction (%)",
                    min_value=0.0, max_value=100.0,
                    value=defaults.get("rrr", 20.0), step=1.0
                )
            else:
                ard = st.number_input(
                    "Assumed absolute risk difference (percentage points)",
                    min_value=0.0, max_value=50.0,
                    value=defaults.get("ard", 2.0), step=0.1
                )

            n_per_group = st.number_input(
                "Planned sample size per group (optional)",
                min_value=0, max_value=500000,
                value=defaults.get("n_per_group", 0),
                step=100
            )

    return {
        "alpha": alpha,
        "power_target": power_target,
        "control_override": control_override,
        "control_manual": control_manual,
        "effect_mode": effect_mode,
        "rrr": rrr,
        "ard": ard,
        "n_per_group": n_per_group,
    }

# -----------------------------
# Plot Section Helper
# -----------------------------
def plot_section(fig, title=None):
    if title:
        st.subheader(title)
    st.pyplot(fig)
