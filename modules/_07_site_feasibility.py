# site_feasibility.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

# -----------------------------
# Utility: simulate site recruitment
# -----------------------------
def simulate_site_recruitment(
    n_sites: int,
    mean_eligible_patients: int,
    sd_eligible_patients: int,
    mean_enrollment_rate: float,
    sd_enrollment_rate: float,
    seed: int = 42
):
    """
    Simulate number of enrolled patients per site.
    """
    rng = np.random.default_rng(seed)
    eligible = np.clip(rng.normal(mean_eligible_patients, sd_eligible_patients, n_sites), 10, None)
    enroll_rate = np.clip(rng.normal(mean_enrollment_rate, sd_enrollment_rate, n_sites), 0.05, 1.0)
    enrolled = np.round(eligible * enroll_rate)

    df = pd.DataFrame({
        "Site ID": [f"S{i+1}" for i in range(n_sites)],
        "Eligible Patients": eligible.astype(int),
        "Enrollment Rate": enroll_rate,
        "Projected Enrollment": enrolled.astype(int)
    })
    df["Enrollment Score"] = (enrolled / enrolled.max()) * 100  # normalized score
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
def run():
    st.header("🏥 Site Feasibility & Recruitment Predictor")

    st.markdown("""
    Estimate recruitment potential for clinical trial sites based on historical performance and population estimates.
    """)

    # User inputs
    n_sites = st.number_input("Number of Sites", min_value=1, max_value=500, value=20, step=1)
    mean_eligible = st.number_input("Mean Eligible Patients per Site", value=100)
    sd_eligible = st.number_input("SD of Eligible Patients per Site", value=30)
    mean_rate = st.slider("Mean Enrollment Rate (fraction of eligible)", 0.05, 1.0, 0.25)
    sd_rate = st.slider("SD of Enrollment Rate", 0.0, 0.5, 0.1)

    if st.button("Simulate Recruitment"):
        with st.spinner("Simulating site recruitment..."):
            df_sites = simulate_site_recruitment(
                n_sites, mean_eligible, sd_eligible, mean_rate, sd_rate
            )

        st.subheader("📊 Recruitment Table")
        st.dataframe(df_sites)

        # Plot: projected enrollment
        st.subheader("Projected Enrollment by Site")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_sites["Site ID"], df_sites["Projected Enrollment"], color='skyblue')
        ax.set_xlabel("Site ID")
        ax.set_ylabel("Projected Enrollment")
        ax.set_title("Site-Level Recruitment Projection")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Summary metrics
        st.markdown("### Summary Metrics")
        st.write(f"**Total Projected Enrollment:** {df_sites['Projected Enrollment'].sum()}")
        st.write(f"**Average Enrollment per Site:** {df_sites['Projected Enrollment'].mean():.1f}")
        st.write(f"**Top Site:** {df_sites.loc[df_sites['Projected Enrollment'].idxmax(), 'Site ID']} with {df_sites['Projected Enrollment'].max()} projected enrollments")

    add_footer()

if __name__ == "__main__":
    run()
