import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser


# ==============================================
# Load API Key
# ==============================================
load_dotenv()

# ==============================================
# Pydantic Schema
# ==============================================
class PatientInfo(BaseModel):
    primary_symptom: Optional[str] = None
    severity: Optional[str] = None
    duration: Optional[str] = None
    associated_symptoms: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)

# ==============================================
# Initialize Model
# ==============================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_retries=2
)

# ==============================================
# Prompt + Parser
# ==============================================
parser = PydanticOutputParser(pydantic_object=PatientInfo)

prompt_template = """
Extract the required medical information from the following text.
Do not sound like AI — keep output clean and human-like.
Return data only in JSON using this format:

{format_instructions}

Patient Description:
\"\"\"{patient_text}\"\"\"
"""

prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(prompt_template)],
    input_variables=["patient_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

extraction_chain = prompt | llm | parser

# ==============================================
# Streamlit UI — Modern Glass Style
# ==============================================
st.set_page_config(page_title="🩺 Medical Info Extractor", layout="wide")

# Custom CSS for a modern glass UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f9fafb 0%, #eef1f5 100%);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.75);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🩺 Medical Patient Information Extractor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Convert patient text into clean and structured medical information.</p>", unsafe_allow_html=True)

# ==============================================
# Session State
# ==============================================
if "patient_id" not in st.session_state:
    st.session_state.patient_id = 1
if "patients_data" not in st.session_state:
    st.session_state.patients_data = []

# ==============================================
# Input Section
# ==============================================
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.subheader("✍️ Enter Patient Description")
    patient_text = st.text_area(
        "",
        placeholder="Example: I've had a strong headache for 3 days with nausea...",
        height=150
    )

    cols = st.columns([1,1])
    with cols[0]:
        extract_btn = st.button("🔍 Extract Details", use_container_width=True)
    with cols[1]:
        clear_btn = st.button("🧹 Clear Input", use_container_width=True)

    if clear_btn:
        patient_text = ""

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
# Extraction
# ==============================================
if extract_btn:
    if not patient_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing patient's description..."):
            try:
                result = extraction_chain.invoke({"patient_text": patient_text})
                result_dict = result.dict()

                entry = {
                    "id": st.session_state.patient_id,
                    "patient_text": patient_text,
                    "extracted_info": result_dict
                }

                st.session_state.patients_data.append(entry)
                st.session_state.patient_id += 1

                st.success("Patient information extracted successfully!")

                # Display Output
                with st.container():
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.subheader("📋 Extracted Information")

                    c1, c2 = st.columns(2)

                    with c1:
                        if result.primary_symptom:
                            st.metric("Primary Symptom", result.primary_symptom)
                        if result.severity:
                            st.metric("Severity", result.severity)

                    with c2:
                        if result.duration:
                            st.metric("Duration", result.duration)
                        st.metric("Total Symptoms", len(result.associated_symptoms))

                    if result.associated_symptoms:
                        st.write("### 🤒 Associated Symptoms")
                        for s in result.associated_symptoms:
                            st.write(f"- {s}")

                    if result.medical_history:
                        st.write("### 🩻 Medical History")
                        for h in result.medical_history:
                            st.write(f"- {h}")

                    st.write("### 🧾 Raw JSON")
                    st.json(result_dict)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Save All Data
                with open("all_patients_data.json", "w", encoding="utf-8") as f:
                    json.dump(st.session_state.patients_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==============================================
# Display All Saved Patients
# ==============================================
if st.session_state.patients_data:
    st.markdown("<h2>📊 Patient Records</h2>", unsafe_allow_html=True)

    for p in reversed(st.session_state.patients_data):
        with st.expander(f"Patient #{p['id']} — {p['extracted_info'].get('primary_symptom','No symptom')}"):
            st.write("#### Description")
            st.write(p["patient_text"])

            info = p["extracted_info"]

            if info.get("severity"): st.write(f"- **Severity:** {info['severity']}")
            if info.get("duration"): st.write(f"- **Duration:** {info['duration']}")

            if info.get("associated_symptoms"):
                st.write(f"- **Associated Symptoms:** {', '.join(info['associated_symptoms'])}")

            if info.get("medical_history"):
                st.write(f"- **Medical History:** {', '.join(info['medical_history'])}")

# ==============================================
# Download Button
# ==============================================
if st.session_state.patients_data:
    st.download_button(
        "📥 Download JSON",
        json.dumps(st.session_state.patients_data, indent=2, ensure_ascii=False),
        "all_patients_data.json",
        "application/json",
        use_container_width=True
    )


