# modules/_09_dct_feasibility.py

import re
import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data
def get_dct_rules() -> pd.DataFrame:
    """Return DCT feasibility rules database."""
    return pd.DataFrame([
        {"activity": "Informed Consent", "category": "Administrative", "feasibility_score": 95,
         "recommended_tech": "eConsent platforms (Medable, Signant)", "notes": "Video consent + digital signature"},
        {"activity": "Patient Screening", "category": "Enrollment", "feasibility_score": 70,
         "recommended_tech": "Remote surveys, EHR access, home test kits", "notes": "Depends on lab/imaging needs"},
        {"activity": "ECG", "category": "Assessment", "feasibility_score": 80,
         "recommended_tech": "Wearable ECG (Apple Watch, KardiaMobile, Zio)", "notes": "FDA-cleared devices available"},
        {"activity": "Echocardiogram", "category": "Assessment", "feasibility_score": 30,
         "recommended_tech": "Home health + portable echo (limited)", "notes": "Mostly site-based"},
        {"activity": "Blood Draw", "category": "Lab", "feasibility_score": 60,
         "recommended_tech": "Home health phlebotomy, dried blood spots", "notes": "Logistics-dependent"},
        {"activity": "Vital Signs (BP, HR)", "category": "Monitoring", "feasibility_score": 90,
         "recommended_tech": "Smartwatches, connected BP cuffs", "notes": "High accuracy with calibrated devices"},
        {"activity": "Adverse Event Reporting", "category": "Safety", "feasibility_score": 85,
         "recommended_tech": "ePRO apps, SMS, voice assistants", "notes": "Real-time reporting possible"},
        {"activity": "Study Visits (Follow-up)", "category": "Visit", "feasibility_score": 75,
         "recommended_tech": "Telehealth (Zoom, Doxy.me), hybrid visits", "notes": "Can combine remote + local"},
        {"activity": "Medication Dispensing", "category": "Treatment", "feasibility_score": 70,
         "recommended_tech": "Direct-to-patient shipping, smart pillboxes", "notes": "With compliance tracking"},
        {"activity": "Cognitive/Quality of Life Survey", "category": "Endpoint", "feasibility_score": 95,
         "recommended_tech": "eCOA platforms (Medidata, OpenClinica)", "notes": "Fully remote"},
        {"activity": "Wearable Data (Activity, Sleep)", "category": "Digital Biomarker", "feasibility_score": 100,
         "recommended_tech": "Fitbit, Apple Watch, Oura Ring", "notes": "Continuous passive collection"},
        {"activity": "Central Imaging Review", "category": "Assessment", "feasibility_score": 50,
         "recommended_tech": "Cloud-based imaging (Nordic, Sectra)", "notes": "Upload from local clinics"},
    ])

def adjust_score(row: pd.Series, therapeutic_area: str) -> float:
    """Adjust feasibility scores based on therapeutic area logic."""
    if therapeutic_area == "Cardiovascular":
        if "ECG" in row['activity']:
            return min(row['feasibility_score'] + 10, 100)
        if "Echocardiogram" in row['activity']:
            return min(row['feasibility_score'] + 5, 100)
    return row['feasibility_score']

@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


# -----------------------------
# Main App
# -----------------------------
def run():
    st.title("🌐 Decentralized Trial (DCT) Feasibility Engine")
    st.markdown("An AI-powered tool to assess how much of your clinical trial can be virtual.")

    rules_df = get_dct_rules()

    # Sidebar: Trial context
    st.sidebar.header("Trial Context")
    therapeutic_area = st.sidebar.selectbox(
        "Therapeutic Area",
        ["Cardiovascular", "Diabetes", "Neurology", "Oncology", "Rare Disease"]
    )
    trial_phase = st.sidebar.selectbox("Trial Phase", ["II", "III", "IV"])
    duration_months = st.sidebar.slider("Trial Duration (months)", 3, 36, 12)

    # Select activities
    st.header("1. Select Trial Activities")
    selected_activities = st.multiselect(
        "Choose activities in your trial protocol:",
        options=rules_df['activity'].tolist(),
        default=["Informed Consent", "ECG", "Vital Signs (BP, HR)",
                 "Adverse Event Reporting", "Study Visits (Follow-up)"]
    )

    if not selected_activities:
        st.warning("Please select at least one trial activity.")
        return

    selected_df = rules_df[rules_df['activity'].isin(selected_activities)].copy()
    selected_df['adjusted_score'] = selected_df.apply(lambda r: adjust_score(r, therapeutic_area), axis=1)
    dct_score = selected_df['adjusted_score'].mean()
    remote_pct = dct_score
    in_person_pct = 100 - remote_pct

    # -----------------------------
    # DCT Results
    # -----------------------------
    st.header("2. DCT Feasibility Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DCT Feasibility Score", f"{dct_score:.0f}/100")
    with col2:
        st.metric("Est. % Remote", f"{remote_pct:.0f}%")
    with col3:
        st.metric("Est. % In-Person", f"{in_person_pct:.0f}%")

    st.progress(int(dct_score))

    if dct_score >= 80:
        st.success("✅ High DCT potential! Consider a fully decentralized or hybrid model.")
    elif dct_score >= 50:
        st.info("🟡 Moderate DCT potential. Focus on hybrid visits and remote monitoring.")
    else:
        st.warning("⚠️ Low DCT potential. Most activities require site visits. Optimize logistics.")

    # -----------------------------
    # Activity Breakdown
    # -----------------------------
    st.header("3. Activity Breakdown")
    display_df = selected_df[[
        'activity', 'category', 'adjusted_score', 'recommended_tech'
    ]].rename(columns={
        'adjusted_score': 'Remote Feasibility (%)',
        'recommended_tech': 'Recommended Technology'
    })
    st.dataframe(display_df.style.format({"Remote Feasibility (%)": "{:.0f}"}), use_container_width=True)

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.header("4. Recommendations")

    # 🛠️ Recommended Technologies grouped by category
    st.markdown("#### 🛠️ Recommended Technologies")

    category_to_tech = {}
    for _, row in selected_df.iterrows():
        # Split on commas not inside parentheses
        parts = re.split(r',(?![^()]*\))', row['recommended_tech'])
        for t in parts:
            clean = t.strip()
            if clean:
                category_to_tech.setdefault(row['category'], set()).add(clean)

    # Display grouped technologies
    for category, techs in sorted(category_to_tech.items()):
        st.markdown(f"**{category}**")
        st.markdown("\n".join([f"- {t}" for t in sorted(techs)]))
        st.markdown("")

    # 💡 Suggested Improvements
    st.markdown("#### 💡 Suggested Improvements")
    low_feasibility = selected_df[selected_df['adjusted_score'] < 60]
    if len(low_feasibility):
        st.write("Consider alternatives for low-feasibility activities:")
        for _, row in low_feasibility.iterrows():
            st.write(f"- **{row['activity']}**: {row['notes']}")
    else:
        st.write("All activities are highly amenable to decentralization.")

    # 💰 Estimated Benefits
    st.markdown("#### 💰 Estimated Benefits")
    st.markdown("""
    - **Time Savings**: ~15–30% faster enrollment
    - **Cost Reduction**: ~20–40% lower site overhead
    - **Patient Retention**: ~10–25% improvement expected
    """)

    # -----------------------------
    # Export CSV
    # -----------------------------
    csv = convert_df(display_df)
    st.download_button(
        label="📥 Download Report (CSV)",
        data=csv,
        file_name="dct_feasibility_report.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")
