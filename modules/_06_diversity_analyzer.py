# pages/_06_diversity_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

def run():
    st.title("Diversity & Representation Analyzer")
    st.markdown(
        """
        Analyze baseline population demographics for clinical trials.
        Explore representation across age, sex, race/ethnicity, and other subgroups.
        """
    )

    # -----------------------
    # Upload or generate baseline
    # -----------------------
    st.header("1) Baseline Population Data")
    baseline_option = st.radio("Provide baseline data", ["Upload CSV (patient-level)", "Generate synthetic cohort"], index=0)
    df = None

    if baseline_option.startswith("Upload"):
        bl_file = st.file_uploader("Upload CSV (columns: age, sex, race, ethnicity, other demographics)", type=["csv"])
        if bl_file is not None:
            df = pd.read_csv(bl_file)
            st.success(f"Loaded {len(df)} patients.")
    else:
        n_demo = st.number_input("Number of patients", min_value=10, max_value=200000, value=500)
        age_mean = st.number_input("Age mean", min_value=30, max_value=90, value=60)
        age_sd = st.number_input("Age SD", min_value=0, max_value=30, value=10)
        pct_male = st.slider("Percent male", 0.0, 1.0, 0.5)
        pct_black = st.slider("Percent Black", 0.0, 1.0, 0.15)
        pct_asian = st.slider("Percent Asian", 0.0, 1.0, 0.1)
        pct_hispanic = st.slider("Percent Hispanic", 0.0, 1.0, 0.15)

        # generate synthetic cohort
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "age": np.clip(rng.normal(age_mean, age_sd, n_demo).round(), 18, 100),
            "sex": rng.choice([0, 1], size=n_demo, p=[1 - pct_male, pct_male]),
            "race_black": rng.choice([0, 1], size=n_demo, p=[1 - pct_black, pct_black]),
            "race_asian": rng.choice([0, 1], size=n_demo, p=[1 - pct_asian, pct_asian]),
            "ethnicity_hispanic": rng.choice([0, 1], size=n_demo, p=[1 - pct_hispanic, pct_hispanic]),
        })
        st.success(f"Generated synthetic cohort of {n_demo} patients.")

    if df is not None:
        # -----------------------
        # Summary statistics
        # -----------------------
        st.header("2) Summary Statistics")
        st.write("Basic demographics:")
        demo_summary = {
            "N": len(df),
            "Mean Age": df["age"].mean(),
            "Median Age": df["age"].median(),
            "Pct Male": df["sex"].mean(),
            "Pct Black": df.get("race_black", pd.Series([0]*len(df))).mean(),
            "Pct Asian": df.get("race_asian", pd.Series([0]*len(df))).mean(),
            "Pct Hispanic": df.get("ethnicity_hispanic", pd.Series([0]*len(df))).mean(),
        }
        st.json({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in demo_summary.items()})

        # -----------------------
        # Visualizations
        # -----------------------
        st.header("3) Visualizations")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Age histogram
        axes[0].hist(df["age"], bins=15, color="#8daed6", edgecolor="black")
        axes[0].set_title("Age distribution")
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Count")

        # Sex pie chart
        axes[1].pie(df["sex"].value_counts(), labels=["Female", "Male"], autopct="%1.1f%%", colors=["#f4b183", "#8daed6"])
        axes[1].set_title("Sex distribution")

        # Race/ethnicity stacked bar
        categories = ["race_black", "race_asian", "ethnicity_hispanic"]
        counts = [df.get(cat, pd.Series([0]*len(df))).sum() for cat in categories]
        axes[2].bar(categories, counts, color=["#e06666", "#6aa84f", "#3d85c6"])
        axes[2].set_title("Race/Ethnicity representation")
        axes[2].set_ylabel("Count")

        plt.tight_layout()
        st.pyplot(fig)

        # -----------------------
        # Export summary
        # -----------------------
        st.header("4) Download")
        summary_df = pd.DataFrame([demo_summary])
        st.download_button(
            "Download summary (CSV)",
            data=summary_df.to_csv(index=False).encode(),
            file_name="diversity_summary.csv",
            mime="text/csv"
        )

        st.info("Notes: This analyzer provides transparent demographics summaries for planning clinical trial representation. For regulatory submissions, consider detailed subgroup analyses per guidance.")

    add_footer()