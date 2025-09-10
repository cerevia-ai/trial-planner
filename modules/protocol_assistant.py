# protocol_assistant.py
import streamlit as st
from typing import List
import openai

# -----------------------------
# Configure API key (set via environment variable)
# -----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY")  # or set in your .env

# -----------------------------
# Utility: Generate protocol section
# -----------------------------
def generate_protocol_section(
    section_name: str,
    therapeutic_area: str,
    endpoints: List[str],
    inclusion_criteria: str,
    exclusion_criteria: str,
    statistical_plan: str
) -> str:
    """
    Call LLM to generate a draft protocol section.
    """
    prompt = f"""
    You are a clinical trial protocol writer. Draft the '{section_name}' section for a clinical trial.

    Trial Details:
    - Therapeutic area: {therapeutic_area}
    - Primary and secondary endpoints: {', '.join(endpoints)}
    - Inclusion criteria: {inclusion_criteria}
    - Exclusion criteria: {exclusion_criteria}
    - Statistical plan: {statistical_plan}

    Use standard ICH-GCP/FDA style. Keep it clear, concise, and ready for submission.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        text = response['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        return f"Error generating section: {e}"


# -----------------------------
# Streamlit UI
# -----------------------------
def run():
    st.header("📝 AI Protocol Assistant")

    st.markdown("""
    This tool generates draft clinical trial protocol sections based on your trial inputs.
    """)

    # -----------------------------
    # User inputs
    # -----------------------------
    therapeutic_area = st.selectbox(
        "Therapeutic Area",
        ["Cardiovascular Disease", "HFpEF", "Post-MI", "Diabetes", "Other"]
    )

    endpoints = st.multiselect(
        "Endpoints",
        ["All-cause mortality", "Hospitalization", "BP reduction", "LDL change", "Composite CV outcome"]
    )

    inclusion_criteria = st.text_area("Inclusion Criteria", "Age 18-85, history of MI, LVEF > 35%")
    exclusion_criteria = st.text_area("Exclusion Criteria", "Active cancer, severe CKD, pregnancy")
    statistical_plan = st.text_area("Statistical Plan", "Sample size calculated for 80% power, alpha=0.05")

    section = st.selectbox(
        "Protocol Section to Generate",
        ["Objectives", "Endpoints", "Inclusion/Exclusion", "Study Design", "Statistical Analysis"]
    )

    if st.button("Generate Section"):
        with st.spinner("Generating protocol section..."):
            text = generate_protocol_section(
                section,
                therapeutic_area,
                endpoints,
                inclusion_criteria,
                exclusion_criteria,
                statistical_plan
            )
        st.subheader(f"📄 {section} Section")
        st.text_area(f"{section} Draft", value=text, height=400)

if __name__ == "__main__":
    run()
