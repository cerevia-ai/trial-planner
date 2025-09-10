# pages/_01_sample_size.py
import streamlit as st
from importlib import import_module

from utils.stats_utils import sample_size_proportions
from utils.plot_utils import plot_power_curve, plot_event_counts
from utils.pdf_utils import create_pdf
from utils.ui_helpers import population_inputs, trial_design_inputs, plot_section

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

def run():
    # -----------------------------
    # Domain selection
    # -----------------------------
    DOMAIN_CONFIGS = {
        "Cardiovascular": "config.cvd_endpoints",
        "Oncology": "config.oncology_endpoints",
        "Neurology": "config.neuro_endpoints"
    }

    domain = st.selectbox("Therapeutic Domain", list(DOMAIN_CONFIGS.keys()))

    # Dynamically import domain-specific endpoints
    module = import_module(DOMAIN_CONFIGS[domain])
    CONDITIONS = getattr(module, "CONDITIONS")
    ENDPOINT_LIST = getattr(module, "ENDPOINT_LIST")
    ENDPOINTS = getattr(module, "ENDPOINTS")
    HISTORICAL_DATA = getattr(module, "HISTORICAL_DATA")

    # -----------------------------
    # Page title/description
    # -----------------------------
    st.title("Sample Size & Power Calculator")
    st.markdown("Estimate trial sample size and event rates across therapeutic domains.")

    # -----------------------------
    # User Inputs
    # -----------------------------
    st.header("1. Trial Design Parameters")
    condition = st.selectbox("Patient Condition", CONDITIONS)
    endpoint = st.selectbox("Primary Endpoint", ENDPOINT_LIST)
    metadata = ENDPOINTS[endpoint]
    st.write(f"Endpoint type: {metadata['type']}")
    st.write(f"Description: {metadata['description']}")

    pop_params = population_inputs()
    trial_params = trial_design_inputs()

    if trial_params["effect_mode"] == "Relative risk reduction (%)":
        experimental_rate = trial_params["control_manual"] * (1 - trial_params["rrr"] / 100) \
            if trial_params["control_override"] else pop_params.get("control_rate", 0.2) * (1 - trial_params["rrr"] / 100)
    else:
        experimental_rate = max(
            0.0,
            (trial_params["control_manual"] if trial_params["control_override"] else pop_params.get("control_rate", 0.2))
            - trial_params["ard"] / 100
        )

    st.write(f"➡️ Experimental group event rate: **{experimental_rate:.1%}**")

    # -----------------------------
    # Calculate sample size
    # -----------------------------
    if st.button("📊 Calculate Sample Size"):
        control_rate = trial_params["control_manual"] if trial_params["control_override"] else pop_params.get("control_rate", 0.2)
        n_per_group = sample_size_proportions(control_rate, experimental_rate, alpha=trial_params["alpha"], power=trial_params["power_target"])
        total_n = 2 * n_per_group

        st.success("### ✅ Required Sample Size")
        st.write(f"- **{n_per_group:,}** patients per group")
        st.write(f"- **{total_n:,}** total patients")
        st.write(f"- Power: {trial_params['power_target']*100:.0f}%, Alpha: {trial_params['alpha']}")
        st.write(f"- Endpoint: {endpoint} in {condition} patients over {pop_params['follow_up']} years")

        control_events = int(n_per_group * control_rate)
        experimental_events = int(n_per_group * experimental_rate)
        st.info(f"Estimated events: **{control_events}** in control, **{experimental_events}** in experimental group")

        fig_power = plot_power_curve(control_rate, experimental_rate, alpha=trial_params["alpha"], max_n=5000)
        plot_section(fig_power, title="📈 Power Curve")

        fig_events = plot_event_counts(control_events, experimental_events)
        plot_section(fig_events, title="📊 Expected Events by Group")

        pdf_file = create_pdf(
            app_name="Sample Size & Power Calculator",
            control_events=control_events,
            experimental_events=experimental_events,
            sample_size=n_per_group,
            effect_size=abs(control_rate - experimental_rate),
            fig_power=fig_power,
            fig_events=fig_events
        )
        # Offer download WITHOUT saving to disk
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_file,
            file_name="clinical_trial_report.pdf",
            mime="application/pdf"
        )

    # Add footer
    add_footer()
