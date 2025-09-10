# rbm_planner.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

# -----------------------------
# Utilities
# -----------------------------
def simulate_rbm(
    n_sites: int,
    n_patients_per_site: int,
    high_risk_fraction: float,
    remote_monitoring_capacity: float,
    seed: int = 42
):
    """
    Simulate risk-based monitoring assignments for sites and patients.
    """
    rng = np.random.default_rng(seed)
    site_risk = rng.random(n_sites)  # site-level risk score 0-1
    patient_risk = rng.random((n_sites, n_patients_per_site))  # patient-level risk 0-1

    # Determine monitoring allocation
    monitor_plan = []
    for i in range(n_sites):
        n_patients = n_patients_per_site
        high_risk_patients = int(high_risk_fraction * n_patients)
        # Assign high-risk patients for more intensive monitoring
        intensive_monitoring = np.zeros(n_patients)
        # top x% patients by risk get intensive monitoring
        top_idx = patient_risk[i].argsort()[-high_risk_patients:]
        intensive_monitoring[top_idx] = 1
        # remote monitoring coverage reduces onsite visits
        intensive_monitoring *= (1 - remote_monitoring_capacity)
        monitor_plan.append(intensive_monitoring)

    return np.array(monitor_plan), site_risk

# -----------------------------
# Streamlit UI
# -----------------------------
def run():
    st.header("📈 Risk-Based Monitoring (RBM) Planner")

    st.markdown("""
    Design a risk-based monitoring strategy before trial launch.
    Prioritize high-risk sites and patients while simulating remote vs on-site monitoring.
    """)

    # Inputs
    n_sites = st.number_input("Number of Sites", min_value=1, max_value=100, value=10)
    n_patients = st.number_input("Patients per Site", min_value=5, max_value=500, value=50)
    high_risk_frac = st.slider("Fraction of High-Risk Patients for Intensive Monitoring", 0.0, 1.0, 0.2)
    remote_capacity = st.slider("Remote Monitoring Capacity (fraction reducing onsite visits)", 0.0, 1.0, 0.3)

    if st.button("Generate RBM Plan"):
        with st.spinner("Simulating RBM plan..."):
            monitor_plan, site_risk = simulate_rbm(
                n_sites, n_patients, high_risk_frac, remote_capacity
            )

        # Summarize site-level monitoring
        summary_df = pd.DataFrame({
            "Site ID": [f"S{i+1}" for i in range(n_sites)],
            "Site Risk Score": site_risk,
            "Patients Under Intensive Monitoring": monitor_plan.sum(axis=1).astype(int)
        }).sort_values(by="Site Risk Score", ascending=False)

        st.subheader("📊 Site Monitoring Summary")
        st.dataframe(summary_df)

        # Plot: Intensive monitoring per site
        st.subheader("Intensive Monitoring Allocation per Site")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(summary_df["Site ID"], summary_df["Patients Under Intensive Monitoring"], color='salmon')
        ax.set_xlabel("Site ID")
        ax.set_ylabel("Patients under Intensive Monitoring")
        ax.set_title("RBM Allocation by Site")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("### Notes")
        st.markdown("""
        - Sites with higher risk scores get more intensive monitoring.
        - Fraction of high-risk patients and remote monitoring capacity affect onsite visit allocation.
        - This is a **simulation**; integrate with real patient-level risk metrics for operational planning.
        """)

    add_footer()

if __name__ == "__main__":
    run()
