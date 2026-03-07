import streamlit as st
from datetime import datetime
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel, Field

# OCR / File processing
from PIL import Image
import pytesseract
import pdf2image
import PyPDF2

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import base64
import os
from dotenv import load_dotenv

# =============================
# 1. LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# =============================
# 2. PYDANTIC SCHEMA
# =============================

class MedicalAnalysisSchema(BaseModel):
    patient_summary: str = Field(description="Brief overview of the patient's current state.")
    extracted_symptoms: List[str] = Field(description="List of specific symptoms identified.")
    severity_level: str = Field(description="Low, Moderate, or High based on symptoms.")
    possible_conditions: List[str] = Field(description="Potential medical conditions matching symptoms.")
    recommended_tests: List[str] = Field(description="Diagnostic tests suggested for further investigation.")
    treatment_suggestions: List[str] = Field(description="General advice or over-the-counter suggestions.")
    urgent_warnings: Optional[str] = Field(description="Red flags that require immediate ER visit.")

# =============================
# 3. OCR FUNCTIONS
# =============================

def extract_text_from_image(file):
    try:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if not text.strip():
            file.seek(0)
            images = pdf2image.convert_from_bytes(file.read())
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {str(e)}"
    return text

# =============================
# 4. AI ANALYZER (LCEL Chain)
# =============================

def analyze_medical_text_structured(patient_text: str) -> MedicalAnalysisSchema:
    if not API_KEY:
        raise ValueError("API Key missing! Please set GOOGLE_API_KEY in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=API_KEY,
        temperature=0.1
    )
    
    structured_llm = llm.with_structured_output(MedicalAnalysisSchema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional medical diagnostic assistant. Extract symptoms and provide a structured medical analysis based on the patient's text."),
        ("human", "{input}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({"input": patient_text})

# =============================
# 5. PDF & UI HELPERS
# =============================

def generate_pdf(data: MedicalAnalysisSchema):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Medical Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Title"]))
    story.append(Spacer(1, 12))

    def add_section(title, content):
        story.append(Paragraph(f"<b>{title}:</b>", styles["Heading3"]))
        if isinstance(content, list):
            for item in content:
                story.append(Paragraph(f"• {item}", styles["Normal"]))
        else:
            story.append(Paragraph(str(content), styles["Normal"]))
        story.append(Spacer(1, 10))

    add_section("Patient Summary", data.patient_summary)
    add_section("Symptoms Extracted", data.extracted_symptoms)
    add_section("Severity Level", data.severity_level)
    add_section("Possible Conditions", data.possible_conditions)
    add_section("Recommended Tests", data.recommended_tests)
    add_section("Treatment Suggestions", data.treatment_suggestions)
    if data.urgent_warnings:
        add_section("⚠️ URGENT WARNINGS", data.urgent_warnings)

    doc.build(story)
    buffer.seek(0)
    return buffer

def display_results(result_data: MedicalAnalysisSchema):
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 Symptoms & Conditions", "💊 Recommendations", "⚠️ Warnings"])
    
    with tab1:
        st.subheader("Patient Summary")
        st.info(result_data.patient_summary)
        
        severity = result_data.severity_level.lower()
        if "high" in severity:
            st.error(f"⚠️ Severity Level: {result_data.severity_level}")
        elif "moderate" in severity:
            st.warning(f"⚠️ Severity Level: {result_data.severity_level}")
        else:
            st.success(f"✅ Severity Level: {result_data.severity_level}")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted Symptoms")
            for symptom in result_data.extracted_symptoms:
                st.write(f"• {symptom}")
        with col2:
            st.subheader("Possible Conditions")
            for condition in result_data.possible_conditions:
                st.write(f"• {condition}")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recommended Tests")
            for test in result_data.recommended_tests:
                st.write(f"• {test}")
        with col2:
            st.subheader("Treatment Suggestions")
            for treatment in result_data.treatment_suggestions:
                st.write(f"• {treatment}")
    
    with tab4:
        if result_data.urgent_warnings:
            st.error("🚨 URGENT MEDICAL ATTENTION")
            st.write(result_data.urgent_warnings)
        else:
            st.info("No urgent warnings identified.")

# =============================
# 6. CLEAR FUNCTION
# =============================

def clear_all():
    """Clear all session state and reset the app"""
    st.session_state.analysis_result = None
    st.session_state.extracted_text = ""
    st.session_state.input_type = "✏️ Type Symptoms"  # Reset to default
    st.rerun()  # Rerun the app to reflect changes

# =============================
# 7. STREAMLIT APP LAYOUT
# =============================

st.set_page_config(page_title="Medical Symptom Analyzer", page_icon="🏥", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .clear-button {
        background-color: #f0f2f6;
        color: #31333F;
        border: 1px solid #d3dae8;
    }
    .clear-button:hover {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title with clear button in same row
col_title, col_clear = st.columns([6, 1])
with col_title:
    st.title("🏥 AI Medical Symptom Analyzer")
with col_clear:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    if st.button("🗑️ Clear All", type="secondary", help="Clear all inputs and results"):
        clear_all()

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'input_type' not in st.session_state:
    st.session_state.input_type = "✏️ Type Symptoms"

# Check if API Key is loaded
if not API_KEY:
    st.error("⚠️ Backend Error: GOOGLE_API_KEY not found in .env file. Please add your API key to continue.")
    st.stop()

# Input area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Patient Data")
    
    # Use session state to remember input type
    input_type = st.radio(
        "Choose Input", 
        ["✏️ Type Symptoms", "📄 Upload Medical Report"], 
        horizontal=True,
        key="input_type_radio",
        index=0 if st.session_state.input_type == "✏️ Type Symptoms" else 1
    )
    st.session_state.input_type = input_type
    
    if input_type == "✏️ Type Symptoms":
        patient_input = st.text_area(
            "How are you feeling?", 
            height=200,
            value=st.session_state.extracted_text if st.session_state.extracted_text else "",
            key="text_input_area"
        )
        st.session_state.extracted_text = patient_input
    else:
        file = st.file_uploader(
            "Upload File (PDF/JPG/PNG)", 
            type=["pdf", "png", "jpg", "jpeg"],
            key="file_uploader"
        )
        if file:
            with st.spinner("Extracting content..."):
                if file.type == "application/pdf":
                    st.session_state.extracted_text = extract_text_from_pdf(file)
                else:
                    st.session_state.extracted_text = extract_text_from_image(file)
                st.success("✅ Content extracted successfully!")

with col2:
    st.header("📊 Overview")
    if st.session_state.extracted_text and st.session_state.extracted_text.strip():
        st.metric("Character Count", len(st.session_state.extracted_text))
        st.metric("Word Count", len(st.session_state.extracted_text.split()))
        st.write("**Preview:**")
        st.info(st.session_state.extracted_text[:150] + "..." if len(st.session_state.extracted_text) > 150 else st.session_state.extracted_text)
        
        # Quick clear button in overview
        if st.button("🧹 Clear Text", use_container_width=True):
            st.session_state.extracted_text = ""
            st.rerun()
    else:
        st.info("👆 Enter symptoms or upload a file to begin")

# Analysis Button and Additional Clear Options
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    analyze_clicked = st.button("🔬 Generate Analysis Report", type="primary", use_container_width=True)

with col_btn1:
    if st.button("🔄 Reset Form", use_container_width=True):
        clear_all()

if analyze_clicked:
    if not st.session_state.extracted_text.strip():
        st.warning("⚠️ Please provide input text or upload a file before generating a report.")
    else:
        with st.spinner("🔄 Processing medical data with AI... This may take a moment."):
            try:
                result = analyze_medical_text_structured(st.session_state.extracted_text)
                st.session_state.analysis_result = result
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"❌ Analysis Failed: {str(e)}")
                st.info("Please check your API key and try again.")

# Display Results & Export
if st.session_state.analysis_result:
    st.markdown("---")
    st.header("📋 Analysis Results")
    
    # Add a small clear results button
    col_res1, col_res2, col_res3 = st.columns([4, 1, 1])
    with col_res2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.analysis_result = None
            st.rerun()
    
    display_results(st.session_state.analysis_result)
    
    st.markdown("---")
    pdf_buffer = generate_pdf(st.session_state.analysis_result)
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    
    # Centered download button with styling
    st.markdown(
        f'''
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <a href="data:application/pdf;base64,{b64}" 
               download="Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf" 
               style="padding:15px 30px; 
                      background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
                      color:white; 
                      border-radius:10px; 
                      text-decoration:none; 
                      font-weight:bold;
                      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                      transition: all 0.3s ease;">
                📥 Download Professional PDF Report
            </a>
        </div>
        ''', 
        unsafe_allow_html=True
    )

