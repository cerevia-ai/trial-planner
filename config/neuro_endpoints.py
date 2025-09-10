# config/neuro_endpoints.py
import pandas as pd

# -----------------------------
# Neurological conditions / patient cohorts
# -----------------------------
CONDITIONS = [
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Multiple Sclerosis",
    "Epilepsy",
    "Stroke (Ischemic)",
    "Migraine",
    "Amyotrophic Lateral Sclerosis (ALS)",
    "Huntington's Disease",
]

# -----------------------------
# Neurology endpoints
# -----------------------------
# type: 'proportion' | 'time' | 'continuous'
ENDPOINTS = {
    "Cognitive Decline (MMSE)": {"type": "continuous", "description": "Change in Mini-Mental State Exam score"},
    "Time to Relapse": {"type": "time", "description": "Time until neurological relapse or event"},
    "Seizure Frequency": {"type": "continuous", "description": "Number of seizures per period"},
    "Disability Progression (EDSS)": {"type": "continuous", "description": "Change in Expanded Disability Status Scale"},
    "Functional Independence (mRS)": {"type": "proportion", "description": "Proportion of patients with favorable modified Rankin Scale outcome"},
    "Symptom Response Rate": {"type": "proportion", "description": "Proportion of patients showing symptom improvement"},
    "Mortality": {"type": "time", "description": "Time to death from any cause"},
}

# -----------------------------
# Historical / Example Data
# -----------------------------
HISTORICAL_DATA = pd.DataFrame({
    "condition": [
        "Alzheimer's Disease", "Multiple Sclerosis", "Epilepsy", "Stroke (Ischemic)", "Parkinson's Disease"
    ],
    "endpoint": [
        "Cognitive Decline (MMSE)", "Time to Relapse", "Seizure Frequency", "Functional Independence (mRS)", "Symptom Response Rate"
    ],
    "follow_up_years": [1.0, 2.0, 1.5, 1.0, 2.0],
    "control_event_rate": [None, None, None, 0.60, 0.30],  # for proportion endpoints
    "baseline_mean": [25.0, None, 3.0, None, None],         # for continuous endpoints
    "baseline_sd": [3.5, None, 1.2, None, None],
})

# -----------------------------
# Convenience lists for UI
# -----------------------------
ENDPOINT_LIST = list(ENDPOINTS.keys())
