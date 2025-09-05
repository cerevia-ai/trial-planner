# app.py - Cardiovascular Trial Sample Size Calculator
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import zt_ind_solve_power, NormalIndPower
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import tempfile


# -----------------------------
# PDF Report Generator
# -----------------------------
def create_pdf(app_name, control_events, experimental_events, sample_size, effect_size, fig_power, fig_events):
    """Generate a polished PDF report, date, sample size results, and charts."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)

    styles = getSampleStyleSheet()
    custom_title = ParagraphStyle("CustomTitle", parent=styles['Title'], alignment=1, fontSize=20, spaceAfter=20)

    story = []

    story.append(Paragraph(f"<b>{app_name}</b>", custom_title))
    story.append(Paragraph(f"Report generated on: {datetime.today().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Title
    story.append(Paragraph("📄 Clinical Trial Sample Size Report", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Parameters
    story.append(Paragraph(f"<b>Effect Size:</b> {effect_size:.3f}", styles['Normal']))
    story.append(Paragraph(f"<b>Required Sample Size per Group:</b> {int(sample_size)}", styles['Normal']))
    story.append(Paragraph(f"<b>Expected Control Events:</b> {control_events}", styles['Normal']))
    story.append(Paragraph(f"<b>Expected Experimental Events:</b> {experimental_events}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Save charts as temp PNGs
    power_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    events_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig_power.savefig(power_path, bbox_inches="tight")
    fig_events.savefig(events_path, bbox_inches="tight")

    # Add images
    story.append(Paragraph("Power Curve:", styles['Heading2']))
    story.append(Image(power_path, width=5*inch, height=3.5*inch))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Expected Events by Group:", styles['Heading2']))
    story.append(Image(events_path, width=5*inch, height=3.5*inch))

    # Build PDF
    doc.build(story)
    return temp_file.name


# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data
def load_historical_data():
    """Simulated historical CVD trial dataset."""
    return pd.DataFrame({
        'condition': [
            'HFrEF', 'HFrEF', 'HFpEF', 'Stable CAD', 'Post-MI', 'Atrial Fibrillation',
            'Hypertension', 'Stable CAD', 'HFrEF', 'Peripheral Artery Disease'
        ],
        'endpoint': [
            'Hospitalization', 'CV Death', 'HF Hospitalization', 'MACE', 'MACE',
            'Stroke', 'MACE', 'MI', 'CV Death', 'Amputation'
        ],
        'follow_up_years': [1.0, 2.0, 1.5, 1.0, 2.0, 1.0, 3.0, 1.0, 1.8, 2.0],
        'control_event_rate': [0.22, 0.15, 0.18, 0.10, 0.14, 0.08, 0.06, 0.09, 0.17, 0.12]
    })


@st.cache_resource
def train_event_rate_predictor():
    """Train a simple RandomForest to suggest control event rates."""
    df = load_historical_data()
    X = pd.get_dummies(df[['condition', 'endpoint']], columns=['condition', 'endpoint'])
    X['follow_up_years'] = df['follow_up_years']
    y = df['control_event_rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist()


def calculate_sample_size(effect_size, alpha=0.05, power=0.8):
    """Calculate sample size per group for two-sample proportion test."""
    n = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n))


def plot_power_curve(effect_size, alpha=0.05, max_n=5000):
    """Plot statistical power as a function of sample size per group."""
    analysis = NormalIndPower()
    sample_sizes = np.arange(50, max_n, 50)
    powers = [analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0)
              for n in sample_sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sample_sizes, powers, label="Power Curve", linewidth=2)
    ax.axhline(0.8, color="red", linestyle="--", label="80% Power Threshold")
    ax.set_xlabel("Sample Size per Group")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power vs. Sample Size")
    ax.legend()
    ax.grid(True)
    return fig


def plot_event_counts(control_events, experimental_events):
    """Bar chart comparing expected events in control vs experimental groups."""
    labels = ["Control Group", "Experimental Group"]
    values = [control_events, experimental_events]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Expected Number of Events")
    ax.set_title("Expected Events per Group")
    for i, v in enumerate(values):
        ax.text(i, v + 1, str(v), ha="center", va="bottom", fontweight="bold")
    return fig


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="CVD Trial Sample Size Calculator", layout="centered")
st.title("Cardiovascular Trial Sample Size Calculator")
st.markdown("An AI-powered tool to estimate sample size and suggest realistic event rates.")

# Load data and model
historical_df = load_historical_data()
model, feature_columns = train_event_rate_predictor()

st.sidebar.header("Instructions")
st.sidebar.info("Fill in the trial details to get AI suggestions and sample size.")

# User Inputs
st.header("1. Trial Design Parameters")

condition = st.selectbox(
    "Patient Condition",
    options=['HFrEF', 'HFpEF', 'Stable CAD', 'Post-MI', 'Atrial Fibrillation',
             'Hypertension', 'Peripheral Artery Disease', 'Myocarditis', 'Valvular Heart Disease']
)

endpoint = st.selectbox(
    "Primary Endpoint",
    options=['MACE', 'CV Death', 'Hospitalization', 'MI', 'Stroke', 'Arrhythmia', 'HF Hospitalization']
)

follow_up = st.slider("Follow-up Duration (years)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# AI Suggestion for Control Event Rate
st.header("2. AI-Powered Suggestion")

input_data = pd.DataFrame([{
    f"condition_{condition}": 1,
    f"endpoint_{endpoint}": 1,
    "follow_up_years": follow_up
}])
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

predicted_rate = model.predict(input_data)[0]
predicted_rate = np.clip(predicted_rate, 0.01, 0.5)  # realistic bounds

st.write(f"🔍 **AI Suggestion**: Expected **control event rate** ≈ **{predicted_rate:.1%}** per {follow_up} year(s).")

use_suggested = st.checkbox("Use AI-suggested event rate", value=True)
control_rate = predicted_rate if use_suggested else st.number_input(
    "Enter your own control event rate", min_value=0.01, max_value=0.99,
    value=predicted_rate, format="%.3f"
)

rrr = st.slider("Expected Relative Risk Reduction (RRR)", min_value=10, max_value=50, value=25, step=5)
experimental_rate = control_rate * (1 - rrr / 100)

st.write(f"➡️ Experimental group event rate: {experimental_rate:.1%}")

# Effect Size
p1, p2 = control_rate, experimental_rate
p_avg = (p1 + p2) / 2
effect_size = (p1 - p2) / np.sqrt(p_avg * (1 - p_avg))

# -----------------------------
# Results Section
# -----------------------------
if st.button("📊 Calculate Sample Size"):
    n_per_group = calculate_sample_size(effect_size=effect_size)
    total_n = 2 * n_per_group

    st.success("### ✅ Required Sample Size")
    st.write(f"- **{n_per_group:,}** patients per group")
    st.write(f"- **{total_n:,}** total patients")
    st.write(f"- Power: 80%, Alpha: 0.05")
    st.write(f"- Endpoint: {endpoint} in {condition} patients over {follow_up} years")

    # Event counts
    control_events = int(n_per_group * control_rate)
    experimental_events = int(n_per_group * experimental_rate)
    st.info(f"Estimated events: **{control_events}** in control, **{experimental_events}** in experimental group")

    # Plots
    st.subheader("📈 Power Curve")
    fig_power = plot_power_curve(effect_size=effect_size, alpha=0.05, max_n=5000)
    st.pyplot(fig_power)

    st.subheader("📊 Expected Events by Group")
    fig_events = plot_event_counts(control_events, experimental_events)
    st.pyplot(fig_events)

    # PDF Report Download
    pdf_file = create_pdf(
        control_events=control_events,
        experimental_events=experimental_events,
        sample_size=n_per_group,
        effect_size=effect_size,
        fig_power=fig_power,
        fig_events=fig_events
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="sample_size_report.pdf",
            mime="application/pdf"
        )


# Optional: Show historical data
with st.expander("📚 View Example Historical Trials"):
    st.dataframe(historical_df, use_container_width=True)
