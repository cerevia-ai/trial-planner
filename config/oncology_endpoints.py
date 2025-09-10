# config/oncology_endpoints.py
import pandas as pd

# -----------------------------
# Cancer types / patient cohorts
# -----------------------------
CONDITIONS = [
    "Non-Small Cell Lung Cancer (NSCLC)",
    "Small Cell Lung Cancer (SCLC)",
    "Breast Cancer",
    "Colorectal Cancer",
    "Prostate Cancer",
    "Ovarian Cancer",
    "Melanoma",
    "Glioblastoma",
]

# -----------------------------
# Oncology endpoints
# -----------------------------
# type: 'proportion' | 'time' | 'continuous'
ENDPOINTS = {
    "Overall Survival (OS)": {"type": "time", "description": "Time to death from any cause"},
    "Progression-Free Survival (PFS)": {"type": "time", "description": "Time to disease progression or death"},
    "Objective Response Rate (ORR)": {"type": "proportion", "description": "Proportion of patients with tumor size reduction"},
    "Complete Response (CR)": {"type": "proportion", "description": "Proportion of patients achieving complete response"},
    "Partial Response (PR)": {"type": "proportion", "description": "Proportion of patients achieving partial response"},
    "Tumor Volume Change": {"type": "continuous", "description": "Continuous measurement of tumor size change"},
    "Biomarker Level Change": {"type": "continuous", "description": "Change in biomarker levels"},
}

# -----------------------------
# Historical / Example Data
# -----------------------------
# Optional small dataset for AI suggestions or priors
HISTORICAL_DATA = pd.DataFrame({
    "condition": [
        "NSCLC", "Breast Cancer", "Colorectal Cancer", "Melanoma", "Prostate Cancer"
    ],
    "endpoint": [
        "ORR", "PFS", "OS", "Tumor Volume Change", "CR"
    ],
    "follow_up_years": [1.0, 2.0, 3.0, 1.5, 2.5],
    "control_event_rate": [0.20, 0.15, 0.40, None, 0.10],  # for proportion endpoints
    "baseline_mean": [None, None, None, 100.0, None],      # for continuous endpoints
    "baseline_sd": [None, None, None, 25.0, None],
})

# -----------------------------
# Convenience lists for UI
# -----------------------------
ENDPOINT_LIST = list(ENDPOINTS.keys())
