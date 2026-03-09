import streamlit as st
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# OCR / File processing
from PIL import Image
import pytesseract
import pdf2image
import PyPDF2

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import base64
import os
import json
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
    patient_demographics: Optional[Dict] = Field(description="Patient age, gender, etc.")
    duration_of_symptoms: Optional[str] = Field(description="How long symptoms have been present")
    previous_treatments: Optional[str] = Field(description="Any treatments tried before")
    medical_history: Optional[str] = Field(description="Past medical history")
    allergies: Optional[str] = Field(description="Any allergies")
    medications: Optional[str] = Field(description="Current medications")
    family_history: Optional[str] = Field(description="Family medical history")
    lifestyle_factors: Optional[str] = Field(description="Smoking, alcohol, exercise, etc.")
    doctor_notes: Optional[str] = Field(description="Additional clinical notes")
    previous_reports_summary: Optional[str] = Field(description="Summary of previous medical reports")

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

def analyze_medical_report(report_text: str) -> str:
    """Analyze uploaded medical report and provide summary"""
    if not API_KEY:
        return "API Key missing for report analysis"
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=API_KEY,
        temperature=0.2
    )
    
    prompt = f"""As a medical professional, analyze this medical report/text and provide a detailed summary:
    
    {report_text}
    
    Please provide:
    1. Key findings and abnormalities
    2. Relevant medical history
    3. Previous diagnoses
    4. Past treatments mentioned
    5. Any important notes for current consultation
    
    Summary:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except:
        return "Could not analyze report automatically. Please discuss with Dr. Khan."

# =============================
# 4. ENHANCED DOCTOR CHATBOT
# =============================

def get_chat_response(messages: List, user_input: str, uploaded_reports: List = None) -> str:
    """Get response from Gemini with detailed doctor-like behavior"""
    if not API_KEY:
        raise ValueError("API Key missing!")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=API_KEY,
        temperature=0.3
    )
    
    # Add context about uploaded reports
    reports_context = ""
    if uploaded_reports and len(uploaded_reports) > 0:
        reports_context = "\n\nPREVIOUS MEDICAL REPORTS UPLOADED:\n"
        for i, report in enumerate(uploaded_reports, 1):
            reports_context += f"Report {i} ({report['filename']}): {report['summary'][:300]}...\n"
    
    # Detailed system prompt for thorough history taking
    system_prompt = f"""You are Dr. Khan, an experienced physician with 20+ years of practice. Your consultation style is thorough, empathetic, and detailed.

{reports_context}

**CRITICAL INSTRUCTION: Always ask ONE question at a time. Never ask multiple questions in one message.**

Your consultation MUST cover ALL these aspects systematically:

1. **CHIEF COMPLAINT** (Current symptoms)
   - "What exactly is bothering you?"
   - "When did this first start?"
   - "Can you describe how it feels?"
   - "Where exactly is the problem located?"
   - "Does it spread anywhere else?"

2. **SYMPTOM DETAILS** (For each symptom)
   - "On a scale of 1-10, how severe is it?"
   - "Is it constant or does it come and go?"
   - "What makes it worse?"
   - "What makes it better?"
   - "Have you noticed it at specific times?"

3. **PAST TREATMENTS**
   - "Have you tried any medications for this?"
   - "Did you see any other doctor for this?"
   - "What treatments have you tried before?"
   - "Did those treatments help?"

4. **MEDICAL HISTORY**
   - "Do you have any existing medical conditions?"
   - "Have you had any surgeries in the past?"
   - "Are you currently taking any medications?"
   - "Do you have any allergies?"

5. **FAMILY HISTORY**
   - "Does anyone in your family have similar issues?"
   - "Any family history of diabetes, BP, heart problems?"

6. **LIFESTYLE**
   - "Do you smoke or drink alcohol?"
   - "What does your daily diet look like?"
   - "Do you exercise regularly?"
   - "How is your sleep?"

7. **REVIEW PREVIOUS REPORTS** (if uploaded)
   - "I see from your previous reports that you had [finding]. How is that now?"
   - "Your past reports show [condition]. Are you still managing that?"

**Example of proper conversation:**
Dr. Khan: "Tell me about your headache. When did it start?"
Patient: "Yesterday"
Dr. Khan: "I see. And how would you describe the pain - is it throbbing, sharp, or dull?"
Patient: "Throbbing"
Dr. Khan: "Thank you. On a scale of 1-10, how severe is it?"

Remember: 
- Be warm and professional
- Show empathy ("I understand", "That must be difficult")
- Use simple language, explain medical terms
- NEVER ask multiple questions at once
- Build conversation naturally based on patient's answers"""
    
    # Create messages list
    langchain_messages = [HumanMessage(content=system_prompt)]
    
    # Add chat history (last 6 messages for focused context)
    for msg in messages[-6:]:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=f"Patient: {msg['content']}"))
        else:
            langchain_messages.append(AIMessage(content=f"Dr. Khan: {msg['content']}"))
    
    # Add current user input
    langchain_messages.append(HumanMessage(content=f"Patient: {user_input}"))
    
    try:
        response = llm.invoke(langchain_messages)
        return response.content
    except Exception as e:
        return f"I apologize for the technical difficulty. Could you please repeat that?"

def generate_detailed_report(chat_history: List, uploaded_reports: List = None) -> MedicalAnalysisSchema:
    """Generate comprehensive medical report from chat"""
    if not API_KEY:
        raise ValueError("API Key missing!")
    
    # Combine all information
    conversation = "COMPREHENSIVE MEDICAL CONSULTATION\n"
    conversation += "=" * 60 + "\n\n"
    conversation += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    conversation += "Physician: Dr. Khan\n\n"
    
    # Add uploaded reports
    if uploaded_reports and len(uploaded_reports) > 0:
        conversation += "PREVIOUS MEDICAL REPORTS:\n"
        conversation += "-" * 30 + "\n"
        for i, report in enumerate(uploaded_reports, 1):
            conversation += f"\nReport {i}: {report['filename']}\n"
            conversation += f"Summary: {report['summary']}\n"
        conversation += "\n" + "=" * 60 + "\n\n"
    
    # Add conversation transcript
    conversation += "DETAILED CONSULTATION TRANSCRIPT:\n"
    conversation += "-" * 30 + "\n\n"
    for msg in chat_history:
        role = "Patient" if msg["role"] == "user" else "Dr. Khan"
        conversation += f"{role}: {msg['content']}\n\n"
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=API_KEY,
        temperature=0.1
    )
    
    structured_llm = llm.with_structured_output(MedicalAnalysisSchema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior physician creating a detailed medical report. Based on the consultation transcript and previous reports, create a comprehensive report covering:

        1. Patient Summary: Complete overview of the case
        2. Extracted Symptoms: Detailed list of all symptoms with characteristics
        3. Severity Level: Overall severity assessment (Low/Moderate/High)
        4. Possible Conditions: Differential diagnoses with reasoning
        5. Recommended Tests: Specific investigations needed
        6. Treatment Suggestions: Detailed management plan
        7. Urgent Warnings: Any red flags requiring immediate attention
        8. Patient Demographics: Age, gender, relevant details
        9. Duration of Symptoms: Timeline of current illness
        10. Previous Treatments: Any treatments already tried
        11. Medical History: Past medical conditions, surgeries
        12. Allergies: Any known allergies
        13. Current Medications: Ongoing medications
        14. Family History: Relevant family medical history
        15. Lifestyle Factors: Smoking, alcohol, diet, exercise
        16. Doctor Notes: Additional clinical observations
        17. Previous Reports Summary: Key findings from past reports

        Be thorough and professional, as this will be used for clinical decision-making."""),
        ("human", "{input}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({"input": conversation})

# =============================
# 5. PDF GENERATION
# =============================

def generate_pdf(data: MedicalAnalysisSchema, chat_history: List = None, uploaded_reports: List = None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        textColor=colors.HexColor('#2c3e50')
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#34495e'),
        borderWidth=1,
        borderColor=colors.HexColor('#bdc3c7'),
        borderPadding=5,
        borderRadius=3
    ))
    
    story = []

    # Title
    story.append(Paragraph("Complete Medical Consultation Report", styles["CustomTitle"]))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph("<b>Physician:</b> Dr. Khan", styles["Normal"]))
    story.append(Spacer(1, 20))

    def add_section(title, content, emergency=False):
        if content and str(content).strip():
            story.append(Paragraph(title, styles["CustomHeading"]))
            if isinstance(content, list):
                for item in content:
                    if item and item.strip():
                        if emergency:
                            story.append(Paragraph(f'• <font color="red">{item}</font>', styles["Normal"]))
                        else:
                            story.append(Paragraph(f"• {item}", styles["Normal"]))
            elif isinstance(content, dict):
                for key, value in content.items():
                    if value and str(value).strip():
                        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
            else:
                if emergency:
                    story.append(Paragraph(f'<font color="red">{content}</font>', styles["Normal"]))
                else:
                    story.append(Paragraph(str(content), styles["Normal"]))
            story.append(Spacer(1, 12))

    # All sections
    add_section("PATIENT DEMOGRAPHICS", data.patient_demographics)
    add_section("PREVIOUS REPORTS SUMMARY", data.previous_reports_summary)
    add_section("CHIEF COMPLAINT & SUMMARY", data.patient_summary)
    add_section("DURATION OF SYMPTOMS", data.duration_of_symptoms)
    add_section("DETAILED SYMPTOMS", data.extracted_symptoms)
    add_section("SEVERITY ASSESSMENT", data.severity_level)
    add_section("PREVIOUS TREATMENTS TRIED", data.previous_treatments)
    add_section("PAST MEDICAL HISTORY", data.medical_history)
    add_section("ALLERGIES", data.allergies)
    add_section("CURRENT MEDICATIONS", data.medications)
    add_section("FAMILY HISTORY", data.family_history)
    add_section("LIFESTYLE FACTORS", data.lifestyle_factors)
    add_section("DIFFERENTIAL DIAGNOSIS", data.possible_conditions)
    add_section("RECOMMENDED INVESTIGATIONS", data.recommended_tests)
    add_section("TREATMENT PLAN", data.treatment_suggestions)
    
    if data.urgent_warnings:
        add_section("⚠️ URGENT WARNINGS", data.urgent_warnings, emergency=True)
    
    add_section("ADDITIONAL DOCTOR NOTES", data.doctor_notes)
    
    # Uploaded reports list
    if uploaded_reports:
        story.append(Paragraph("UPLOADED DOCUMENTS", styles["CustomHeading"]))
        for report in uploaded_reports:
            story.append(Paragraph(f"• <b>{report['filename']}</b>", styles["Normal"]))
        story.append(Spacer(1, 12))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("<i>This comprehensive report is generated by AI assistant Dr. Khan. Please review by a qualified healthcare provider.</i>", 
                          styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

def display_detailed_results(result_data: MedicalAnalysisSchema):
    tabs = st.tabs([
        "📋 Summary", "🔍 Symptoms", "📊 History", "🏥 Diagnosis", 
        "💊 Treatment", "📄 Reports", "⚠️ Warnings"
    ])
    
    with tabs[0]:  # Summary
        st.subheader("👨‍⚕️ Clinical Summary")
        st.info(result_data.patient_summary)
        
        col1, col2 = st.columns(2)
        with col1:
            severity = result_data.severity_level.lower()
            if "high" in severity:
                st.error(f"⚠️ Severity: {result_data.severity_level}")
            elif "moderate" in severity:
                st.warning(f"⚠️ Severity: {result_data.severity_level}")
            else:
                st.success(f"✅ Severity: {result_data.severity_level}")
        
        with col2:
            if result_data.duration_of_symptoms:
                st.write(f"**Duration:** {result_data.duration_of_symptoms}")
        
        if result_data.patient_demographics:
            st.write("**Patient Info:**", result_data.patient_demographics)
    
    with tabs[1]:  # Symptoms
        st.subheader("🩺 Detailed Symptoms")
        for symptom in result_data.extracted_symptoms:
            st.write(f"• {symptom}")
    
    with tabs[2]:  # History
        col1, col2 = st.columns(2)
        with col1:
            if result_data.previous_treatments:
                st.subheader("Previous Treatments")
                st.write(result_data.previous_treatments)
            
            if result_data.medical_history:
                st.subheader("Medical History")
                st.write(result_data.medical_history)
            
            if result_data.allergies:
                st.subheader("Allergies")
                st.write(result_data.allergies)
        
        with col2:
            if result_data.medications:
                st.subheader("Current Medications")
                st.write(result_data.medications)
            
            if result_data.family_history:
                st.subheader("Family History")
                st.write(result_data.family_history)
            
            if result_data.lifestyle_factors:
                st.subheader("Lifestyle")
                st.write(result_data.lifestyle_factors)
    
    with tabs[3]:  # Diagnosis
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Possible Conditions")
            for condition in result_data.possible_conditions:
                st.write(f"• {condition}")
        with col2:
            st.subheader("Recommended Tests")
            for test in result_data.recommended_tests:
                st.write(f"• {test}")
    
    with tabs[4]:  # Treatment
        st.subheader("💊 Treatment Plan")
        for treatment in result_data.treatment_suggestions:
            st.write(f"• {treatment}")
        
        if result_data.doctor_notes:
            st.subheader("📝 Additional Notes")
            st.info(result_data.doctor_notes)
    
    with tabs[5]:  # Previous Reports
        st.subheader("📄 Previous Medical Reports")
        if result_data.previous_reports_summary:
            st.write(result_data.previous_reports_summary)
        else:
            st.info("No previous reports uploaded")
    
    with tabs[6]:  # Warnings
        if result_data.urgent_warnings:
            st.error("🚨 URGENT WARNING")
            st.warning(result_data.urgent_warnings)
            st.info("🆘 Please seek immediate medical attention!")
        else:
            st.success("✅ No urgent warnings identified")

# =============================
# 6. CLEAR FUNCTION
# =============================

def clear_all():
    """Clear all session state and reset for new patient"""
    for key in ['messages', 'analysis_result', 'extracted_text', 
                'report_generated', 'show_report', 'uploaded_reports']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# =============================
# 7. STREAMLIT APP
# =============================

st.set_page_config(page_title="Dr. Khan - Detailed Medical Consultation", page_icon="👨‍⚕️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    
    .doctor-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stChatMessage {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    [data-testid="chatMessage"]:has(div[data-testid="assistant-message"]) {
        background-color: #e8f4fd;
        border-left: 5px solid #3498db;
    }
    
    [data-testid="chatMessage"]:has(div[data-testid="user-message"]) {
        background-color: #f0f4c3;
        border-right: 5px solid #cddc39;
    }
    
    .stButton > button {
        border-radius: 20px;
        transition: all 0.3s;
    }
    
    .status-badge {
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        display: inline-block;
    }
    
    .consultation-area {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .upload-box {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9ff;
        margin: 10px 0;
    }
    
    .quick-info-btn {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 8px;
        border-radius: 8px;
        margin: 2px;
        cursor: pointer;
        font-size: 12px;
    }
    
    .quick-info-btn:hover {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👨‍⚕️ **Good morning! I'm Dr. Khan.** Please have a seat. Tell me, what brings you to see me today? Take your time and describe what you've been experiencing."}
    ]
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'show_report' not in st.session_state:
    st.session_state.show_report = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'uploaded_reports' not in st.session_state:
    st.session_state.uploaded_reports = []

# Check API Key
if not API_KEY:
    st.error("⚠️ Configuration Error: GOOGLE_API_KEY not found. Please add it to .env file.")
    st.stop()

# Header
st.markdown("""
<div class="doctor-header">
    <h1 style="margin:0">👨‍⚕️ Dr. Khan's Medical Consultation</h1>
    <p style="margin:5px 0 0; opacity:0.9">Experienced Physician | 20+ Years in Clinical Practice | Thorough & Detailed Consultation</p>
    <p style="margin:5px 0 0; font-size:14px; opacity:0.8">⚠️ AI Assistant - For informational purposes only</p>
</div>
""", unsafe_allow_html=True)

# Status Bar
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    if not st.session_state.show_report:
        st.markdown("""
        <div class="status-badge" style="background-color: #27ae60; color: white;">
            🟢 Detailed Consultation in Progress
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge" style="background-color: #3498db; color: white;">
            📋 Comprehensive Report Ready
        </div>
        """, unsafe_allow_html=True)

with col4:
    if st.button("🆕 New Patient", use_container_width=True):
        clear_all()

# Main consultation area
chat_col, tools_col = st.columns([2, 1])

with chat_col:
    st.markdown('<div class="consultation-area">', unsafe_allow_html=True)
    st.subheader("💬 Detailed Consultation with Dr. Khan")
    st.caption("Dr. Khan will ask you questions one by one to understand your condition thoroughly.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if not st.session_state.show_report:
        prompt = st.chat_input("Type your response to Dr. Khan...")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get doctor's response
            with st.chat_message("assistant"):
                with st.spinner("Dr. Khan is listening and thinking..."):
                    try:
                        response = get_chat_response(
                            st.session_state.messages[:-1], 
                            prompt,
                            st.session_state.uploaded_reports
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = "I apologize, but I'm having a technical difficulty. Could you please repeat that?"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with tools_col:
    st.markdown('<div class="consultation-area">', unsafe_allow_html=True)
    
    # Upload Reports Section
    st.subheader("📁 Upload Medical Reports")
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload previous reports (PDF, Images)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
        key="report_uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if not any(f['filename'] == file.name for f in st.session_state.uploaded_reports):
                with st.spinner(f"Analyzing {file.name}..."):
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                    else:
                        text = extract_text_from_image(file)
                    
                    if text and not text.startswith("Error"):
                        summary = analyze_medical_report(text)
                        
                        st.session_state.uploaded_reports.append({
                            'filename': file.name,
                            'text': text[:500] + "...",
                            'summary': summary
                        })
                        
                        report_msg = f"I've uploaded my previous medical report: {file.name}"
                        st.session_state.messages.append({"role": "user", "content": report_msg})
                        
                        st.success(f"✅ {file.name} uploaded!")
                    else:
                        st.error(f"Could not read {file.name}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display uploaded reports
    if st.session_state.uploaded_reports:
        st.subheader("📋 Uploaded Reports")
        for i, report in enumerate(st.session_state.uploaded_reports):
            with st.expander(f"📄 {report['filename'][:30]}"):
                st.info(report['summary'][:200] + "...")
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.uploaded_reports.pop(i)
                    st.rerun()
    
    st.markdown("---")
    
    # Quick Information Buttons
    st.subheader("📝 Quick Info")
    st.caption("Click to quickly add information")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔴 Emergency", use_container_width=True):
            msg = "I think this might be an emergency"
            st.session_state.messages.append({"role": "user", "content": msg})
            st.rerun()
        
        if st.button("💊 Medications", use_container_width=True):
            msg = "I'm currently taking some medications"
            st.session_state.messages.append({"role": "user", "content": msg})
            st.rerun()
        
        if st.button("🩺 Past History", use_container_width=True):
            msg = "I have some past medical history to share"
            st.session_state.messages.append({"role": "user", "content": msg})
            st.rerun()
    
    with col2:
        if st.button("🤧 Allergies", use_container_width=True):
            msg = "I have some allergies"
            st.session_state.messages.append({"role": "user", "content": msg})
            st.rerun()
        
        if st.button("👨‍👩 Family History", use_container_width=True):
            msg = "I want to share my family medical history"
            st.session_state.messages.append({"role": "user", "content": msg})
            st.rerun()
        
        if st.button("🏃 Lifestyle", use_container_width=True):
            msg = "I want to share my lifestyle habits"
            st.session_state.messages.append({"role": "user", "content": msg})
            st.rerun()
    
    st.markdown("---")
    
    # End Consultation
    if st.button("📋 End Consultation & Generate Report", type="primary", use_container_width=True):
        if len(st.session_state.messages) > 3 or st.session_state.uploaded_reports:
            st.session_state.show_report = True
            st.rerun()
        else:
            st.warning("Please tell Dr. Khan about your symptoms first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Report Generation Section
if st.session_state.show_report:
    st.markdown("---")
    st.markdown('<div class="consultation-area">', unsafe_allow_html=True)
    st.header("📋 Comprehensive Medical Report")
    
    if not st.session_state.report_generated:
        with st.spinner("🔄 Dr. Khan is preparing your detailed medical report..."):
            try:
                result = generate_detailed_report(
                    st.session_state.messages,
                    st.session_state.uploaded_reports
                )
                st.session_state.analysis_result = result
                st.session_state.report_generated = True
                st.balloons()
                st.success("✅ Comprehensive report ready!")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    if st.session_state.analysis_result:
        display_detailed_results(st.session_state.analysis_result)
        
        # Download PDF
        pdf_buffer = generate_pdf(
            st.session_state.analysis_result, 
            st.session_state.messages,
            st.session_state.uploaded_reports
        )
        b64 = base64.b64encode(pdf_buffer.read()).decode()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f'''
                <div style="text-align: center; margin: 30px 0;">
                    <a href="data:application/pdf;base64,{b64}" 
                       download="DrKhan_Complete_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf">
                        <button style="padding:15px 40px; 
                                      background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                                      color:white; 
                                      border:none; 
                                      border-radius:25px; 
                                      cursor:pointer;
                                      font-size:18px;
                                      font-weight:bold;
                                      box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            📥 Download Complete Medical Report
                        </button>
                    </a>
                </div>
                ''', 
                unsafe_allow_html=True
            )
        
        if st.button("🆕 Start New Consultation", use_container_width=True):
            clear_all()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <small>
            👨‍⚕️ Dr. Khan - AI Medical Assistant<br>
            ⚠️ This is an AI consultation tool. Always consult with a real healthcare provider for medical advice.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
