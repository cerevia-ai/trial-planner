import streamlit as st

st.set_page_config(
    page_title="Clinical Trial Planning Suite",
    layout="wide",
    initial_sidebar_state="expanded"  # 👈 auto-expands sidebar
)

st.title("🧪 Clinical Trial Planning SaaS")

# ------------------------
# Module lists
# ------------------------
core_modules = [
    "Sample Size & Power Calculator",
    "Risk-Adjusted Trial Designer",
    "Adaptive Trial Simulator",
    "Digital Twin Control Arm",
    "Endpoint Feasibility Explorer",
]

advanced_modules = [
    "Diversity & Representation Analyzer",
    "Site Feasibility & Recruitment Predictor",
    "RBM Planner",
    "Decentralized Trial (DCT) Feasibility Engine",
]

# ------------------------
# Sidebar: Category + Module selection
# ------------------------
st.sidebar.header("Navigation")

category = st.sidebar.selectbox(
    "Select module category",
    ["Core Modules", "Advanced Modules"]
)

if category == "Core Modules":
    module_to_run = st.sidebar.selectbox("Select a core module", core_modules)
else:
    module_to_run = st.sidebar.selectbox("Select an advanced module", advanced_modules)

# ------------------------
# Module imports & run
# ------------------------
if module_to_run == "Sample Size & Power Calculator":
    from modules import _01_sample_size as mod
    mod.run()
elif module_to_run == "Risk-Adjusted Trial Designer":
    from modules import _02_risk_adjusted as mod
    mod.run()
elif module_to_run == "Adaptive Trial Simulator":
    from modules import _03_adaptive_sim as mod
    mod.run()
elif module_to_run == "Digital Twin Control Arm":
    from modules import _04_digital_twin as mod
    mod.run()
elif module_to_run == "Endpoint Feasibility Explorer":
    from modules import _05_endpoint_feasibility as mod
    mod.run()
elif module_to_run == "Diversity & Representation Analyzer":
    from modules import _06_diversity_analyzer as mod
    mod.run()
elif module_to_run == "Site Feasibility & Recruitment Predictor":
    from modules import _07_site_feasibility as mod
    mod.run()
elif module_to_run == "RBM Planner":
    from modules import _08_rbm_planner as mod
    mod.run()
elif module_to_run == "Decentralized Trial (DCT) Feasibility Engine":
    from modules import _09_dct_feasibility as mod
    mod.run()
