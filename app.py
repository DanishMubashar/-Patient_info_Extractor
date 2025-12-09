import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

# ------------------ Load API Key ------------------
load_dotenv()

# ------------------ Pydantic Schema ------------------
class PatientInfo(BaseModel):
    primary_symptom: Optional[str] = Field(None, description="The main symptom reported by the patient", example="headache")
    severity: Optional[str] = Field(None, description="Severity of the primary symptom (e.g., mild, moderate, severe)", example="severe")
    duration: Optional[str] = Field(None, description="Duration of the primary symptom", example="3 days")
    associated_symptoms: List[str] = Field(default_factory=list, description="Other symptoms reported", example=["nausea", "dizziness"])
    medical_history: List[str] = Field(default_factory=list, description="Relevant medical history", example=["migraine"])

# ------------------ Initialize LLM ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ------------------ Output Parser & Prompt ------------------
parser = PydanticOutputParser(pydantic_object=PatientInfo)

prompt_template = """
You are a professional medical NLP system. Extract the following information from the patient's text exactly as JSON matching this schema:

{format_instructions}

Patient Text:
\"\"\"{patient_text}\"\"\"
"""

prompt = ChatPromptTemplate(
    input_variables=["patient_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Medical NLP Extractor", layout="centered")
st.title("🩺 Medical NLP Patient Info Extractor")

# Initialize session state
if "patient_id" not in st.session_state:
    st.session_state.patient_id = 1

if "patients_data" not in st.session_state:
    st.session_state.patients_data = []

# Patient text input
patient_text = st.text_area("Enter patient complaint / description", height=150)

# Extract Info Button
if st.button("Extract Info"):
    if not patient_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Extracting patient info..."):
            try:
                result: PatientInfo = chain.run(patient_text)
            except Exception as e:
                st.error(f"Error during extraction: {e}")
                result = PatientInfo()

        # Assign ID and store
        patient_entry = {
            "id": st.session_state.patient_id,
            "patient_text": patient_text,
            "extracted_info": result.dict()
        }
        st.session_state.patients_data.append(patient_entry)
        st.session_state.patient_id += 1

        # Display extracted info **ONLY for current patient**
        st.subheader(f"Extracted Info for Patient ID {patient_entry['id']}")
        st.json(patient_entry["extracted_info"])  # <-- SHOW ONLY EXTRACTED INFO

        # Save all patient data to JSON file
        json_filename = "all_patients_data.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.patients_data, f, ensure_ascii=False, indent=2)

        st.success(f"Patient ID {patient_entry['id']} data saved successfully!")

# ------------------ Download JSON ------------------
if st.session_state.get("patients_data"):
    st.download_button(
        label="📥 Download All Patients JSON",
        data=json.dumps(st.session_state.patients_data, ensure_ascii=False, indent=2),
        file_name="all_patients_data.json",
        mime="application/json"
    )


