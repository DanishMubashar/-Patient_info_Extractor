

#  **Medical Patient Information Extraction System using LLMs**

### *A Complete End-to-End Project Report*


## **1. Introduction**

In modern healthcare, doctors often receive patient complaints in unstructured text form such as:

> *“I’ve had a severe headache for 3 days with dizziness and nausea. I also have a history of migraines.”*

Extracting structured, clinically useful data from such descriptions is time-consuming.

However, this task can be performed extremely well using **LLMs (Large Language Models)**.
By using **Google Gemini + LangChain + Streamlit**, we can build an automated system that:

* Extracts medical symptoms
* Identifies severity
* Reads duration
* Detects associated symptoms
* Understands medical history
* Converts everything into structured JSON

This project implements a full **Medical NLP Patient Information Extractor** using modern LLM techniques.


## **2. Project Objective**

The goal of this project is to develop a **streamlined, user-friendly system** that:

1. Takes **raw unstructured patient text** as input
2. Uses an **LLM** to extract essential medical information
3. Formats the extracted data into a **Pydantic schema**
4. Displays the information in a beautiful **Streamlit interface**
5. Saves all patient extractions into a **JSON database**
6. Allows the user to download or clear the stored data


## **3. System Architecture**

The system uses the following components:

### **3.1 Frontend (Streamlit UI)**

* A clean and interactive interface
* Text area for patient input
* Extraction button
* Auto-generated metrics, expandable sections, and JSON viewer
* Download button for all extracted patient data

### **3.2 Backend (LLM + LangChain)**

* Gemini 2.5 Flash model for fast LLM inference
* LangChain Prompt Template
* Pydantic schema for structured output parsing
* Validations and error handling

### **3.3 Storage**

* Stores patient extractions in:

  * Streamlit session state
  * A JSON file (`all_patients_data.json`)

### **3.4 Deployment**

* Fully deployed on **Streamlit Cloud**
* Requirements file ensures correct versions
* LangChain compatibility fixes applied


## 🧩 **4. Data Extraction Schema**

A **Pydantic model** defines exactly what the LLM must extract:

```python
class PatientInfo(BaseModel):
    primary_symptom: Optional[str]
    severity: Optional[str]
    duration: Optional[str]
    associated_symptoms: List[str] = []
    medical_history: List[str] = []
```

### Why Pydantic?

* Enforces strict structure
* Validates LLM output
* Converts results into a JSON-friendly format


##  **5. Prompt Engineering**

A custom template guides the LLM to extract information correctly:

```
You are a professional medical NLP system. Extract the following information 
from the patient's text exactly as JSON matching this schema:

{format_instructions}

Patient Text:
"""{patient_text}"""

Extracted Information:
```

### Why this works?

* Clear instructions
* Schema is injected into the prompt
* Ensures consistent output
* Avoids hallucinations
* Suitable for LLMs used in medical language tasks


## **6. LLM Model Used**

We used **Gemini 2.5 Flash**, which is:

* Fast
* Cost-effective
* Accurate for medical text extraction
* Easy to integrate with LangChain

Configured as:

```python
ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)
```

Low temperature ensures **high accuracy**.


## **7. The Extraction Chain**

We combine:

1. Prompt
2. LLM
3. Output Parser

Into one powerful pipeline:

```python
extraction_chain = prompt | llm | parser
```

This lets us run:

```python
result = extraction_chain.invoke({"patient_text": user_input})
```

The output is a fully validated `PatientInfo` object.


##  **8. Streamlit User Interface**

The UI includes:

### ✔️ Text input area

### ✔️ Buttons for extract / clear

### ✔️ Progress loader

### ✔️ Auto-generated analysis

### ✔️ Interactive JSON viewer

### ✔️ Patient database display

### ✔️ Download & clear buttons

### ✔️ Sidebar with instructions

The UI is designed to be:

* Clean
* Medical-themed
* Beginner-friendly
* Professional enough for a real clinic setting


##  **9. Data Storage**

All patient extractions are stored in:

###  Session State

* Temporary storage per session

###  JSON File (Permanent)

`all_patients_data.json`

Contents example:

```json
{
  "id": 1,
  "patient_text": "I have a severe headache...",
  "extracted_info": {
    "primary_symptom": "headache",
    "severity": "severe",
    "duration": "3 days",
    "associated_symptoms": ["nausea", "dizziness"],
    "medical_history": ["migraine"]
  },
  "timestamp": 1
}
```


##  **10. Required packages `requirements.txt`**



```
streamlit
python-dotenv
pydantic
langchain
langchain-core
langchain-community
langchain-google-genai
google-generativeai
```

After fixing imports → deployment succeeded successfully.


## **11. Final Output Example**

Input:

> “Severe chest pain for 2 hours with sweating and shortness of breath.”

Output:

```json
{
  "primary_symptom": "chest pain",
  "severity": "severe",
  "duration": "2 hours",
  "associated_symptoms": [
    "sweating",
    "shortness of breath"
  ],
  "medical_history": []
}
```
