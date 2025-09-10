# utils/risk_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from statsmodels.stats.power import zt_ind_solve_power, NormalIndPower
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import tempfile
import math

# -----------------------------
# Risk Model & Stats
# -----------------------------

def effect_size_from_rates(p1, p2):
    p_avg = (p1 + p2) / 2.0
    return (p1 - p2) / np.sqrt(max(p_avg * (1 - p_avg), 1e-9))

def sample_size_from_effect(effect_size, alpha=0.05, power=0.80):
    n = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n))

def power_curve(effect_size, alpha=0.05, max_n=5000, step=50):
    analysis = NormalIndPower()
    n_grid = np.arange(step, max_n + step, step)
    power = [analysis.power(effect_size=effect_size, nobs1=int(n), alpha=alpha, ratio=1.0) for n in n_grid]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(n_grid, power, linewidth=2)
    ax.axhline(0.80, linestyle="--")
    ax.set_xlabel("Sample Size per Group")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power vs. Sample Size (Risk-Adjusted)")
    ax.grid(True)
    return fig

def event_bar(control_events, experimental_events):
    labels = ["Control", "Experimental"]
    values = [control_events, experimental_events]
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(labels, values)
    ax.set_ylabel("Expected Number of Events")
    ax.set_title("Expected Events per Group")
    for i, v in enumerate(values):
        ax.text(i, v + max(values)*0.02 + 0.5, str(int(round(v))), ha="center", va="bottom", fontweight="bold")
    return fig

def annual_to_multi_year_prob(p_annual, years, dropout_rate=0.0):
    p_annual = np.clip(p_annual, 1e-9, 0.95)
    hazard = -np.log(1 - p_annual)
    t_eff = years * (1 - dropout_rate/2.0)
    return 1 - np.exp(-hazard * t_eff)

# -----------------------------
# Risk Coefficients & Simulation
# -----------------------------

@dataclass
class RiskCoefficients:
    intercept: float = -2.2
    age_per10: float = 0.35
    male: float = 0.25
    smoker: float = 0.55
    diabetes: float = 0.50
    htn: float = 0.30
    ldl_per30: float = 0.18
    hdl_per10_inv: float = 0.25
    sbp_per10: float = 0.12

def annual_event_probability(row, coefs: RiskCoefficients):
    x = (
        coefs.intercept
        + coefs.age_per10 * ((row["age"] - 60)/10.0)
        + coefs.male * (1 if row["sex"]=="Male" else 0)
        + coefs.smoker * (1 if row["smoker"] else 0)
        + coefs.diabetes * (1 if row["diabetes"] else 0)
        + coefs.htn * (1 if row["htn"] else 0)
        + coefs.ldl_per30 * ((row["ldl"]-130)/30.0)
        + coefs.hdl_per10_inv * ((50-row["hdl"])/10.0)
        + coefs.sbp_per10 * ((row["sbp"]-130)/10.0)
    )
    return 1.0 / (1.0 + math.exp(-x))

def simulate_population(n:int, age_mean:float=60, age_sd:float=8, pct_male:float=0.5,
                        pct_smoker:float=0.15, pct_dm:float=0.1, pct_htn:float=0.3,
                        ldl_mean:float=130, ldl_sd:float=30, hdl_mean:float=50, hdl_sd:float=12,
                        sbp_mean:float=130, sbp_sd:float=15, seed:int=42):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(age_mean, age_sd, n),30,95).round(0)
    sex = rng.choice(["Male","Female"], size=n, p=[pct_male,1-pct_male])
    smoker = rng.random(n) < pct_smoker
    diabetes = rng.random(n) < pct_dm
    htn = rng.random(n) < pct_htn
    ldl = np.clip(rng.normal(ldl_mean, ldl_sd, n),50,250)
    hdl = np.clip(rng.normal(hdl_mean, hdl_sd, n),20,90)
    sbp = np.clip(rng.normal(sbp_mean, sbp_sd, n),90,200)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "smoker": smoker,
        "diabetes": diabetes,
        "htn": htn,
        "ldl": ldl,
        "hdl": hdl,
        "sbp": sbp
    })

def risk_adjusted_event_rate(pop: pd.DataFrame, follow_up_years: float, dropout_rate: float, coefs: RiskCoefficients):
    p_ann = pop.apply(lambda r: annual_event_probability(r, coefs), axis=1).to_numpy()
    p_multi = annual_to_multi_year_prob(p_ann, follow_up_years, dropout_rate)
    return float(np.mean(p_multi))

# -----------------------------
# PDF Export
# -----------------------------
def create_pdf(app_name, logo_bytes, control_events, experimental_events, n_per_group, effect_size, fig_power, fig_events):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=1, fontSize=20, spaceAfter=20)

    story = []
    if logo_bytes:
        logo_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        logo_tmp.write(logo_bytes)
        logo_tmp.flush()
        logo_tmp.close()
        story.append(Image(logo_tmp.name, width=1.5*inch, height=1.5*inch))
    story.append(Paragraph(f"<b>{app_name}</b>", title_style))
    story.append(Paragraph(f"Report generated on: {datetime.today().strftime('%Y-%m-%d')}", styles["Normal"]))
    story.append(Spacer(1,16))

    story.append(Paragraph("Risk-Adjusted Trial Design Report", styles["Heading1"]))
    story.append(Paragraph(f"<b>Effect Size (Cohen's h):</b> {effect_size:.3f}", styles["Normal"]))
    story.append(Paragraph(f"<b>Required Sample Size per Group:</b> {int(n_per_group)}", styles["Normal"]))
    story.append(Paragraph(f"<b>Expected Events:</b> Control {int(round(control_events))}, Experimental {int(round(experimental_events))}", styles["Normal"]))
    story.append(Spacer(1,16))

    p_power = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    p_events = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig_power.savefig(p_power, bbox_inches="tight")
    fig_events.savefig(p_events, bbox_inches="tight")

    story.append(Paragraph("Power Curve", styles["Heading2"]))
    story.append(Image(p_power, width=5*inch, height=3.5*inch))
    story.append(Spacer(1,12))

    story.append(Paragraph("Expected Events by Group", styles["Heading2"]))
    story.append(Image(p_events, width=5*inch, height=3.5*inch))

    doc.build(story)
    with open(tmp.name, "rb") as f:
        return f.read()
