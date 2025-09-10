# pages/02_risk_adjusted.py
import streamlit as st
from statsmodels.stats.power import NormalIndPower

from utils.risk_model import (
    simulate_population,
    risk_adjusted_event_rate,
    RiskCoefficients,
    effect_size_from_rates,
    sample_size_from_effect,
    power_curve,
    event_bar,
    create_pdf
)
from utils.ui_helpers import population_inputs, trial_design_inputs, plot_section

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

def run():
    st.header("🧪 Risk-Adjusted Trial Designer")

    # -----------------------------
    # User Inputs
    # -----------------------------
    pop_params = population_inputs()  # removed expander_title
    trial_params = trial_design_inputs()  # removed expander_title

    # -----------------------------
    # Simulate population & compute risk
    # -----------------------------
    pop = simulate_population(
        n=pop_params["n_cohort"], age_mean=pop_params["age_mean"], age_sd=pop_params["age_sd"],
        pct_male=pop_params["pct_male"], pct_smoker=pop_params["pct_smoker"], pct_dm=pop_params["pct_dm"],
        pct_htn=pop_params["pct_htn"], ldl_mean=pop_params["ldl_mean"], ldl_sd=pop_params["ldl_sd"],
        hdl_mean=pop_params["hdl_mean"], hdl_sd=pop_params["hdl_sd"], sbp_mean=pop_params["sbp_mean"],
        sbp_sd=pop_params["sbp_sd"], seed=pop_params["seed"]
    )

    coefs = RiskCoefficients()
    control_event_rate = risk_adjusted_event_rate(pop, pop_params["follow_up"], pop_params["dropout"], coefs)
    if trial_params["control_override"] and trial_params["control_manual"] is not None:
        control_event_rate = trial_params["control_manual"]

    # Determine experimental event rate
    if trial_params["effect_mode"] == "Relative risk reduction (%)":
        p_control = control_event_rate
        p_experimental = p_control * (1 - trial_params["rrr"]/100.0)
    else:
        p_control = control_event_rate
        p_experimental = max(0.0, p_control - (trial_params["ard"]/100.0))

    # Compute effect size & required sample size
    effect_size = effect_size_from_rates(p_control, p_experimental)
    try:
        n_required = sample_size_from_effect(effect_size, alpha=trial_params["alpha"], power=trial_params["power_target"])
    except Exception:
        n_required = None

    # Achieved power if user supplied sample size
    achieved_power = None
    if trial_params["n_per_group"] > 0:
        analysis = NormalIndPower()
        try:
            achieved_power = analysis.power(
                effect_size=effect_size,
                nobs1=int(trial_params["n_per_group"]),
                alpha=trial_params["alpha"],
                ratio=1.0
            )
        except Exception:
            achieved_power = None

    # Expected events
    n_for_events = int(trial_params["n_per_group"] or n_required or 0)
    control_events_exp = p_control * n_for_events
    experimental_events_exp = p_experimental * n_for_events

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Results")
    st.markdown(f"**Modeled control event rate:** {control_event_rate:.3%}")
    st.markdown(f"**Modeled experimental event rate:** {p_experimental:.3%}")
    st.markdown(f"**Cohen's h (approx):** {effect_size:.4f}")
    if n_required:
        st.markdown(f"**Required sample size per group (power {trial_params['power_target']:.2%}, alpha={trial_params['alpha']}):** {n_required}")
    else:
        st.markdown("**Required sample size per group:** could not be computed.")
    if achieved_power:
        st.markdown(f"**Achieved power with planned n={int(trial_params['n_per_group'])}:** {achieved_power:.3f}")

    # -----------------------------
    # Plots
    # -----------------------------
    fig_power = power_curve(effect_size, alpha=trial_params["alpha"],
                            max_n=max(2000, n_for_events*2),
                            step=max(25, n_for_events//20))
    plot_section(fig_power, title="📈 Power Curve")

    fig_events = event_bar(control_events_exp, experimental_events_exp)
    plot_section(fig_events, title="📊 Expected Events by Group")

    # -----------------------------
    # PDF export
    # -----------------------------
    with st.expander("Export report (PDF)"):
        logo_file = st.file_uploader("Upload logo (optional)", type=["png","jpg","jpeg"])
        app_name = st.text_input("Report title", "Risk-Adjusted Trial Design Report")
        if st.button("Generate PDF"):
            logo_bytes = logo_file.read() if logo_file else None
            pdf_bytes = create_pdf(app_name, logo_bytes, control_events_exp, experimental_events_exp,
                                   n_for_events, effect_size, fig_power, fig_events)
            st.success("PDF generated")
            st.download_button("Download report PDF", pdf_bytes, "risk_adjusted_report.pdf", "application/pdf")

    add_footer()

if __name__ == '__main__':
    run()
