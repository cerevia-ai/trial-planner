# 04_digital_twin.py
# Streamlit app: Digital Twin of the Control Arm (Historical Comparator)
# Constructs synthetic control arm using PS weighting or NN matching.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import io

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")


# -----------------------------
# Utilities
# -----------------------------
def simulate_observational_reference(n=20000, seed=123):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(60, 12, n), 30, 95).round(0)
    sex = rng.choice([0,1], n)
    smoker = (rng.random(n) < 0.18).astype(int)
    diabetes = (rng.random(n) < 0.10).astype(int)
    htn = (rng.random(n) < 0.30).astype(int)
    ldl = np.clip(rng.normal(130,30,n),50,250)
    hdl = np.clip(rng.normal(50,12,n),20,90)
    sbp = np.clip(rng.normal(130,15,n),90,210)

    intercept = -3.0
    x = (intercept
        + 0.35*((age-60)/10)
        + 0.25*sex
        + 0.55*smoker
        + 0.50*diabetes
        + 0.30*htn
        + 0.18*((ldl-130)/30)
        + 0.25*((50-hdl)/10)
        + 0.12*((sbp-130)/10))
    p_annual = 1/(1+np.exp(-x))
    hazard = -np.log(1-p_annual+1e-12)
    follow_up = np.clip(rng.normal(5,2,n),0.5,12)
    ev_time = rng.exponential(1/np.clip(hazard,1e-9,None))
    event = (ev_time <= follow_up).astype(int)
    time = np.minimum(ev_time, follow_up)

    df = pd.DataFrame({
        "age": age, "sex": sex, "smoker": smoker, "diabetes": diabetes, "htn": htn,
        "ldl": ldl, "hdl": hdl, "sbp": sbp, "follow_up_years": np.round(follow_up,3),
        "time": np.round(time,4), "event": event
    })
    return df

def required_columns():
    return ["age","sex","smoker","diabetes","htn","ldl","hdl","sbp"]

def check_cols_presence(df, required=None):
    if required is None:
        required = required_columns()
    missing = [c for c in required if c not in df.columns]
    return missing

def compute_propensity_weights(ref_df, trial_df, covariates, solver="lbfgs"):
    ref = ref_df.copy().reset_index(drop=True)
    trial = trial_df.copy().reset_index(drop=True)
    ref["__is_trial__"] = 0
    trial["__is_trial__"] = 1
    combined = pd.concat([ref[covariates+["__is_trial__"]], trial[covariates+["__is_trial__"]]], axis=0).reset_index(drop=True)

    X = combined[covariates].astype(float).values
    Xs = StandardScaler().fit_transform(X)
    y = combined["__is_trial__"].values

    lr = LogisticRegression(solver=solver, max_iter=2000)
    lr.fit(Xs, y)
    combined["ps"] = lr.predict_proba(Xs)[:,1]

    ps_ref = combined.loc[combined.index < len(ref), "ps"].values
    eps = 1e-6
    odds_ref = (ps_ref + eps)/(1-ps_ref + eps)
    weights = odds_ref * (len(trial)/np.sum(odds_ref))
    return weights, lr, Xs

def weighted_kaplan_meier(ref_df, weights, time_col="time", event_col="event"):
    kmf = KaplanMeierFitter()
    try:
        kmf.fit(ref_df[time_col], event_observed=ref_df[event_col], weights=weights, label="Synthetic control")
    except:
        kmf.fit(ref_df[time_col], event_observed=ref_df[event_col], label="Synthetic control (unweighted)")
    return kmf

def nearest_neighbor_synthetic(ref_df, trial_df, covariates, n_per_trial=1):
    X_ref = StandardScaler().fit_transform(ref_df[covariates].astype(float).values)
    X_trial = StandardScaler().fit_transform(trial_df[covariates].astype(float).values)
    nbrs = NearestNeighbors(n_neighbors=n_per_trial, algorithm="auto").fit(X_ref)
    idxs = nbrs.kneighbors(X_trial)[1].flatten()
    synthetic = ref_df.iloc[idxs].copy().reset_index(drop=True)
    return synthetic

def show_summary_dashboard(trial_df, ref_df, synthetic_sample, covariates, kmf, follow_up_target, weights=None):
    st.subheader("Synthetic Control Summary Dashboard")

    # Survival curves
    st.markdown("**Survival Curves**")
    fig, ax = plt.subplots(figsize=(6,4))
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Kaplan–Meier: Synthetic Control")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    st.pyplot(fig)

    # Covariate balance
    st.markdown("**Covariate Balance**")
    balance = []
    def weighted_mean(series, w): return float(np.sum(series.values * w) / np.sum(w))
    for c in covariates:
        trial_mean = trial_df[c].astype(float).mean()
        ref_mean = float(ref_df[c].astype(float).mean())
        weighted_ref_mean = weighted_mean(ref_df[c], weights) if weights is not None else ref_mean
        synthetic_mean = float(synthetic_sample[c].astype(float).mean())
        balance.append({
            "covariate": c,
            "trial": trial_mean,
            "reference": ref_mean,
            "weighted_ref": weighted_ref_mean,
            "synthetic": synthetic_mean
        })
    balance_df = pd.DataFrame(balance)
    melted = balance_df.melt(id_vars="covariate", value_vars=["trial","reference","weighted_ref","synthetic"],
                             var_name="Dataset", value_name="Mean")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.barplot(x="covariate", y="Mean", hue="Dataset", data=melted, ax=ax2)
    ax2.set_title("Covariate Means by Dataset")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)

    # Event-rate table
    st.markdown(f"**Estimated Event Rates at {follow_up_target:.2f} years**")
    event_rate = 1.0 - kmf.predict(follow_up_target)
    st.table(pd.DataFrame({
        "Dataset": ["Synthetic Control"],
        "Event Rate": [f"{event_rate:.3%}"]
    }))

    # Download
    st.subheader("Download Synthetic Patient-level Sample")
    buf = io.BytesIO()
    synthetic_sample.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download CSV", buf.getvalue(), file_name="synthetic_control.csv", mime="text/csv")


# -----------------------------
# Main app
# -----------------------------
def run():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Digital Twin Home", "Other Page (placeholder)"])
    if page != "Digital Twin Home":
        st.info("Other page content can be added here.")
        st.stop()

    # Main UI
    st.title("Digital Twin — Synthetic Control Arm")
    st.markdown("""
    Upload trial baseline data and generate a synthetic control arm using **propensity-score weighting** or **nearest-neighbor matching**.
    Outputs include patient-level synthetic controls, event rate estimates, and survival curves.
    """)

    # --- Trial and reference upload ---
    col1, col2 = st.columns([1,1])
    with col1:
        st.header("1) Trial baseline data")
        st.write("Required columns:", required_columns())
        trial_file = st.file_uploader("Trial CSV", type=["csv"], key="trial")
        trial_df = None
        if trial_file:
            trial_df = pd.read_csv(trial_file)
            missing = check_cols_presence(trial_df)
            if missing: st.error(f"Missing columns: {missing}"); st.stop()
            st.success(f"Loaded trial: {len(trial_df)} rows")
            st.dataframe(trial_df.head(10))
        else: st.info("Upload trial CSV to continue.")

    with col2:
        st.header("2) Reference dataset")
        st.write("Required: baseline covariates + `time`,`event`")
        ref_file = st.file_uploader("Reference CSV (optional)", type=["csv"], key="ref")
        use_builtin = st.checkbox("Use built-in simulated reference (demo)", value=True if ref_file is None else False)
        ref_df = None
        if ref_file:
            ref_df = pd.read_csv(ref_file)
            missing = check_cols_presence(ref_df, required_columns()+["time","event"])
            if missing: st.error(f"Missing columns: {missing}"); st.stop()
            st.success(f"Loaded reference: {len(ref_df)} rows")
            st.dataframe(ref_df.head(8))
        elif use_builtin:
            st.info("Generating simulated reference (~20k rows)")
            ref_df = simulate_observational_reference()
            st.success(f"Built-in reference: {len(ref_df)} rows")
            st.dataframe(ref_df.sample(8).reset_index(drop=True))

    st.markdown("---")

    # --- Synthetic control generation ---
    st.header("3) Synthetic control generation")
    with st.form("generation"):
        colA, colB = st.columns([1,1])
        with colA:
            method = st.selectbox("Method", ["Propensity-score weighting", "Nearest-neighbor matching"])
            covariates = st.multiselect("Covariates", required_columns(), default=required_columns())
            follow_up_target = st.number_input("Follow-up horizon (years)", 0.1,10.0,3.0,0.25)
        with colB:
            n_synthetic_per_trial = st.number_input("NN: synthetic per trial", 1,10,1)
            resample_size = st.number_input("Synthetic sample size (download)", 10,50000,1000,10)
            random_seed = st.number_input("Random seed (0=random)", 0,9999999,12345)
        submit = st.form_submit_button("Generate synthetic control")

    if submit:
        if trial_df is None or ref_df is None:
            st.error("Trial and reference datasets required."); st.stop()
        np.random.seed(None if random_seed==0 else int(random_seed))
        st.info("Generating synthetic control...")

        if method.startswith("Propensity"):
            weights, lr_model, _ = compute_propensity_weights(ref_df, trial_df, covariates)
            kmf = weighted_kaplan_meier(ref_df, weights)
            probs = weights/np.sum(weights)
            chosen_idx = np.random.choice(len(ref_df), size=int(resample_size), replace=True, p=probs)
            synthetic_sample = ref_df.iloc[chosen_idx].copy().reset_index(drop=True)
            synthetic_sample["source"]="synthetic_control"; synthetic_sample["weight"]=weights[chosen_idx]
        else:
            synthetic_sample = nearest_neighbor_synthetic(ref_df, trial_df, covariates, n_synthetic_per_trial)
            kmf = KaplanMeierFitter()
            kmf.fit(synthetic_sample["time"], event_observed=synthetic_sample["event"], label="Synthetic control (NN)")

        # Show summary dashboard
        show_summary_dashboard(trial_df, ref_df, synthetic_sample, covariates, kmf, follow_up_target,
                               weights=weights if method.startswith("Propensity") else None)

    add_footer()

# Run app if executed directly
if __name__ == "__main__":
    run()
