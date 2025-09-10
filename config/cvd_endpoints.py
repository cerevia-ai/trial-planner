# config/cvd_endpoints.py
import pandas as pd

# CVD conditions common in sample-size tool:
CONDITIONS = [
    "HFrEF",
    "HFpEF",
    "Stable CAD",
    "Post-MI",
    "Atrial Fibrillation",
    "Hypertension",
    "Peripheral Artery Disease",
    "Valvular Heart Disease",
    "Myocarditis",
]

# Endpoints mapping: name -> metadata (type: 'proportion' | 'time' | 'continuous', description)
ENDPOINTS = {
    "MACE": {"type": "proportion", "description": "Major adverse cardiovascular events (composite)"},
    "CV Death": {"type": "time", "description": "Cardiovascular death (time-to-event)"},
    "Hospitalization": {"type": "proportion", "description": "Hospitalization for CV cause (binary)"},
    "MI": {"type": "proportion", "description": "Myocardial infarction (event)"},
    "Stroke": {"type": "proportion", "description": "Stroke (event)"},
    "Arrhythmia": {"type": "proportion", "description": "Arrhythmic events (event)"},
    "HF Hospitalization": {"type": "time", "description": "Heart failure hospitalization (time-to-event)"},
    "Blood Pressure Change": {"type": "continuous", "description": "Change in systolic blood pressure"},
    "LDL Change": {"type": "continuous", "description": "Change in LDL cholesterol"},
}

# Small historical example table (for suggestions). In production, replace with curated dataset.
HISTORICAL_DATA = pd.DataFrame({
    "condition": [
        "HFrEF", "HFrEF", "HFpEF", "Stable CAD", "Post-MI",
        "Atrial Fibrillation", "Hypertension", "Stable CAD", "HFrEF", "Peripheral Artery Disease"
    ],
    "endpoint": [
        "HF Hospitalization", "CV Death", "HF Hospitalization", "MACE", "MACE",
        "Stroke", "MACE", "MI", "CV Death", "Amputation"
    ],
    "follow_up_years": [0.5, 2.0, 1.5, 1.0, 2.0, 1.0, 3.0, 1.0, 1.8, 2.0],
    "control_event_rate": [0.12, 0.10, 0.09, 0.08, 0.14, 0.04, 0.06, 0.07, 0.11, 0.05],
})

# Optionally expose convenience lists for UIs
ENDPOINT_LIST = list(ENDPOINTS.keys())
