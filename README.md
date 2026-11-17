Here is a complete, production-ready `README.md` covering the entire project lifecycle—from architecture and local environment setup to model training, sample test generation, GitHub/Streamlit Cloud deployment, and every real-world deployment issue solved along the way.

Save this directly as `README.md` in your root folder (`C:\Users\chait\Downloads\medisense-ai\README.md`).

```markdown
# 🏥 MediSense AI: Clinical Decision Support & Multi-Disease Triage System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medisense-ai-clinical-decision-support-system.streamlit.app)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-green.svg)

**MediSense AI** is an end-to-end assistive Clinical Decision Support System (CDSS) designed to bridge the gap between unstructured medical reports and real-time clinical prioritization. It ingests multimodal diagnostic files (PDFs, images, text notes, spreadsheets), evaluates patient risk across **five major disease models**, computes calibrated confidence safety tiers, renders local **SHAP biomarker explanations**, flags **pharmacological contraindications**, auto-routes patients to specialist hospital wards, and supports **batch emergency triage** in **English, हिंदी (Hindi), and मराठी (Marathi)**.

---

## 📋 Table of Contents
1. [System Architecture & Key Features](#1-system-architecture--key-features)
2. [Machine Learning Models & Benchmark Performance](#2-machine-learning-models--benchmark-performance)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Step-by-Step Local Setup & Execution (Windows)](#4-step-by-step-local-setup--execution-windows)
5. [Generating Synthetic Patient PDFs for Testing](#5-generating-synthetic-patient-pdfs-for-testing)
6. [GitHub & Streamlit Community Cloud Deployment](#6-github--streamlit-community-cloud-deployment)
7. [Troubleshooting & Deployment Challenges Solved](#7-troubleshooting--deployment-challenges-solved)

---

## 1. System Architecture & Key Features

```text
[Input Reports: PDF / PNG / JPG / TXT / CSV / XLSX]
                       │
                       ▼
[Stage 1: Universal Multimodal Parser (pypdf + Tesseract OCR + Regex)]
         Extracts demographics, vitals, glycemic, renal, hepatic & cardiac markers
                       │
                       ▼
[Stage 2: Multi-Disease ML Inference Engine (5 Calibrated Classifiers)]
         Evaluates P(Diabetes), P(Kidney), P(Liver), P(Stroke), P(Heart)
                       │
                       ▼
[Stage 3: Clinical Safety, Explainability & Pharmacology Guardrails]
         ├─ Safety Gating (Tier 1 High Confidence → Tier 3 Escalate)
         ├─ SHAP Waterfall Attribution Plots (Biomarker impact audit)
         └─ Drug Contraindication Checks (Metformin/NSAID/Statin alerts)
                       │
                       ▼
[Stage 4: High-Throughput Batch Triage & Urgency Ranking]
         Urgency Score = (Max Disease Risk × 0.40) + (Acute Distress Triggers × 0.60)
                       │
                       ▼
[Stage 5: Multilingual UI & Hospital Ward Referral Generation]
         English | हिंदी (Hindi) | मराठी (Marathi) + Printable Admission Slip

```

### Core Capabilities

* **Universal Multimodal Document Parser:** Extracts clinical values from horizontal and multiline/vertical PDF lab tables, scanned images via OCR, and CSV/Excel sheets while filtering out reference range bounds.
* **Multi-Disease Comorbidity Screening:** Simultaneously scores risk across **Diabetes, Chronic Kidney Disease (CKD), Liver Disease, Stroke, and Heart Disease**.
* **Calibrated Action Tiers:** Separates predictions by certainty margin (`|P - 0.5| * 2`) into **Tier 1 (High Confidence)**, **Tier 2 (Moderate Uncertainty)**, and **Tier 3 (Inconclusive / Escalate to Full Lab Workup)**.
* **Explainable AI (SHAP):** Visualizes positive and negative biomarker contributions for every prediction so clinicians can audit model reasoning.
* **Automated Prescription Safety Alerts:** Cross-references organ function markers against clinical pharmacology rules (e.g., flagging Metformin and NSAIDs when Serum Creatinine $\ge 2.0\text{ mg/dL}$).
* **Hospital Ward Routing & Referral Slips:** Automatically assigns high-risk patients to specialist departments (e.g., *Nephrology ICU*, *Cardiology CCU*, *Neurology Stroke Acute Care*) and generates a downloadable referral slip.
* **Batch Emergency Triage Queue:** Processes unlimited patient files simultaneously, ranking them into **🔴 Priority 1 (Immediate)**, **🟡 Priority 2 (Urgent)**, and **🟢 Priority 3 (Routine)**.
* **Multilingual Localization:** Real-time UI and clinical explanation translation across **English, हिंदी, and मराठी**.

---

## 2. Machine Learning Models & Benchmark Performance

| Disease Condition | Clinical Cohort Dataset | Algorithm Architecture | Test ROC-AUC | Accuracy | Sensitivity (Recall) | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| **Diabetes** | PIMA Indian Diabetes Registry | `HistGradientBoostingClassifier` | **0.949** | 88.3% | 81.5% | 84.6% |
| **Kidney Disease (CKD)** | Chronic Kidney Disease Cohort | `HistGradientBoostingClassifier` | **1.000** | 98.8% | 98.0% | 100.0% |
| **Liver Disease (ILPD)** | Indian Liver Patient Dataset | `RandomForestClassifier` (Balanced) | **0.802** | 72.6% | 71.1% | 88.1% |
| **Stroke Risk** | Clinical Stroke Registry | `RandomForestClassifier` (Balanced) | **0.835** | 71.6% | 82.0% | 12.7%* |
| **Heart Disease** | Cleveland Clinic Cohort | `HistGradientBoostingClassifier` | **0.982** | 91.8% | 95.5% | 93.3% |

**Note on Stroke Precision: Because stroke prevalence in clinical cohorts is ~5%, the classifier is intentionally tuned for high Sensitivity (82.0%) to minimize false negatives in emergency triage.*

---

## 3. Project Folder Structure

```text
medisense-ai/
├── .venv/                       # Local Python virtual environment (excluded from Git)
├── .gitignore                   # Git exclusion rules (.venv, __pycache__, temp files)
├── requirements.txt             # Clean, BOM-free Python dependencies for Cloud & Local
├── packages.txt                 # Linux apt system dependency (tesseract-ocr)
├── README.md                    # Complete documentation & setup guide
├── data/
│   └── raw/                     # Training clinical datasets (.csv)
│       ├── diabetes.csv
│       ├── kidney_disease.csv
│       ├── Indian Liver Patient Dataset (ILPD).csv
│       └── healthcare-dataset-stroke-data.csv
└── demo_app/
    ├── app.py                   # Main Streamlit web application
    ├── train_all.py             # Automated ML training & model serialization script
    ├── saved_models/            # Serialized .joblib model bundles & cohort medians
    │   ├── diabetes_model.joblib
    │   ├── kidney_model.joblib
    │   ├── liver_model.joblib
    │   ├── stroke_model.joblib
    │   └── heart_model.joblib
    ├── temp/                    # Generated synthetic patient PDF test files
    └── modules/
        ├── __init__.py          # Package initializer
        ├── parser.py            # PDF, OCR image, CSV, and regex biomarker parser
        ├── predictor.py         # Model loading, median imputation & confidence tiers
        ├── explain.py           # SHAP waterfall plot generator
        ├── triage.py            # Multi-patient batch urgency scoring & ranking
        └── clinical_safety.py   # Drug contraindication guardrails & ward routing

```

---

## 4. Step-by-Step Local Setup & Execution (Windows)

### Step 1: Prerequisites

1. Install **Python 3.11** (recommended for pre-compiled binary wheels).
2. *(Optional for image OCR)* Install Tesseract OCR via Windows PowerShell:
```powershell
winget install --id UB-Mannheim.TesseractOCR

```



### Step 2: Create & Activate Virtual Environment

Open PowerShell inside your project root (`C:\Users\<username>\Downloads\medisense-ai`) and run:

```powershell
# Create virtual environment
python -m venv .venv

# Allow local PowerShell script execution for the current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

```

### Step 3: Install Required Python Libraries

Always invoke `pip` through the `.venv` Python executable to prevent installing into global Python:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

```

### Step 4: Train & Serialize All 5 Disease Models

Generate the `.joblib` model artifacts inside `demo_app/saved_models/`:

```powershell
.\.venv\Scripts\python.exe demo_app/train_all.py

```

### Step 5: Launch the Local Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo_app/app.py

```

The application will open automatically at **`http://localhost:8501`**.

---

## 5. Generating Synthetic Patient PDFs for Testing

Run the following command in PowerShell to generate 4 realistic clinical laboratory reports (Critical, Severe, Urgent, and Healthy profiles) inside `demo_app/temp/` for testing Single Intake and Batch Triage:

```powershell
.\.venv\Scripts\python.exe -c "
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

os.makedirs('demo_app/temp', exist_ok=True)

patients = [
    {
        'filename': 'demo_app/temp/patient_1_ramesh_critical.pdf',
        'name': 'Ramesh Patil', 'age': 65, 'gender': 'Male', 'id': 'MED-8091',
        'status': 'CRITICAL / HIGH RISK',
        'tests': [
            ('Fasting Blood Glucose', '245.0', 'mg/dL', '70 - 99', 'Critical High'),
            ('Blood Pressure', '165/98', 'mmHg', '90-120 / 60-80', 'Severe High'),
            ('Body Mass Index (BMI)', '38.2', 'kg/m2', '18.5 - 24.9', 'Obese Class II'),
            ('Serum Creatinine (sc)', '3.2', 'mg/dL', '0.7 - 1.3', 'Critical High'),
            ('Blood Urea (bu)', '75.0', 'mg/dL', '15 - 40', 'Critical High'),
            ('Total Bilirubin (tot_bilirubin)', '3.4', 'mg/dL', '0.2 - 1.2', 'Elevated'),
            ('Total Cholesterol (chol)', '265.0', 'mg/dL', '< 200', 'High'),
            ('Hemoglobin (hemo)', '9.2', 'g/dL', '13.5 - 17.5', 'Low')
        ]
    },
    {
        'filename': 'demo_app/temp/patient_2_sunita_urgent.pdf',
        'name': 'Sunita Deshmukh', 'age': 52, 'gender': 'Female', 'id': 'MED-8092',
        'status': 'MODERATE RISK / URGENT',
        'tests': [
            ('Fasting Blood Glucose', '155.0', 'mg/dL', '70 - 99', 'Elevated'),
            ('Blood Pressure', '138/88', 'mmHg', '90-120 / 60-80', 'Pre-hypertension'),
            ('Body Mass Index (BMI)', '28.1', 'kg/m2', '18.5 - 24.9', 'Overweight'),
            ('Serum Creatinine (sc)', '1.4', 'mg/dL', '0.6 - 1.1', 'Slightly Elevated'),
            ('Blood Urea (bu)', '35.0', 'mg/dL', '15 - 40', 'Normal'),
            ('Total Bilirubin (tot_bilirubin)', '1.1', 'mg/dL', '0.2 - 1.2', 'Normal'),
            ('Total Cholesterol (chol)', '210.0', 'mg/dL', '< 200', 'Borderline High'),
            ('Hemoglobin (hemo)', '12.0', 'g/dL', '12.0 - 15.5', 'Normal')
        ]
    },
    {
        'filename': 'demo_app/temp/patient_3_amit_healthy.pdf',
        'name': 'Amit Kulkarni', 'age': 26, 'gender': 'Male', 'id': 'MED-8093',
        'status': 'NORMAL / ROUTINE',
        'tests': [
            ('Fasting Blood Glucose', '82.0', 'mg/dL', '70 - 99', 'Normal'),
            ('Blood Pressure', '118/74', 'mmHg', '90-120 / 60-80', 'Optimal'),
            ('Body Mass Index (BMI)', '21.5', 'kg/m2', '18.5 - 24.9', 'Normal'),
            ('Serum Creatinine (sc)', '0.8', 'mg/dL', '0.7 - 1.3', 'Normal'),
            ('Blood Urea (bu)', '18.0', 'mg/dL', '15 - 40', 'Normal'),
            ('Total Bilirubin (tot_bilirubin)', '0.7', 'mg/dL', '0.2 - 1.2', 'Normal'),
            ('Total Cholesterol (chol)', '165.0', 'mg/dL', '< 200', 'Normal'),
            ('Hemoglobin (hemo)', '15.1', 'g/dL', '13.5 - 17.5', 'Normal')
        ]
    },
    {
        'filename': 'demo_app/temp/patient_4_priya_severe.pdf',
        'name': 'Priya Joshi', 'age': 61, 'gender': 'Female', 'id': 'MED-8094',
        'status': 'HIGH RISK / SEVERE',
        'tests': [
            ('Fasting Blood Glucose', '210.0', 'mg/dL', '70 - 99', 'Critical High'),
            ('Blood Pressure', '155/92', 'mmHg', '90-120 / 60-80', 'Stage 2 Hypertension'),
            ('Body Mass Index (BMI)', '34.0', 'kg/m2', '18.5 - 24.9', 'Obese Class I'),
            ('Serum Creatinine (sc)', '2.6', 'mg/dL', '0.6 - 1.1', 'Critical High'),
            ('Blood Urea (bu)', '62.0', 'mg/dL', '15 - 40', 'Critical High'),
            ('Total Bilirubin (tot_bilirubin)', '2.2', 'mg/dL', '0.2 - 1.2', 'Elevated'),
            ('Total Cholesterol (chol)', '248.0', 'mg/dL', '< 200', 'High'),
            ('Hemoglobin (hemo)', '10.4', 'g/dL', '12.0 - 15.5', 'Low')
        ]
    }
]

for p in patients:
    c = canvas.Canvas(p['filename'], pagesize=letter)
    width, height = letter
    c.setFillColor(colors.HexColor('#1A365D'))
    c.rect(0, height - 70, width, 70, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 15)
    c.drawString(40, height - 35, 'METROPOLIS CLINICAL & DIAGNOSTIC LABORATORIES')
    c.setFont('Helvetica', 9)
    c.drawString(40, height - 52, 'Automated Clinical Pathology & Patient Triage System | NABL Accredited')
    
    c.setFillColor(colors.HexColor('#F7FAFC'))
    c.setStrokeColor(colors.HexColor('#CBD5E0'))
    c.roundRect(40, height - 155, width - 80, 70, 4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#2D3748'))
    c.setFont('Helvetica-Bold', 9)
    c.drawString(55, height - 100, f'Patient Name : {p[\"name\"]}')
    c.drawString(55, height - 118, f'Patient Age  : {p[\"age\"]} yrs')
    c.drawString(55, height - 136, f'Gender       : {p[\"gender\"]}')
    c.drawString(330, height - 100, f'Patient ID : {p[\"id\"]}')
    c.drawString(330, height - 118, 'Date       : 17-Oct-2026')
    c.drawString(330, height - 136, f'Status     : {p[\"status\"]}')
    
    y = height - 190
    c.setFont('Helvetica-Bold', 9)
    c.drawString(50, y, 'TEST PARAMETER')
    c.drawString(250, y, 'VALUE')
    c.drawString(360, y, 'UNIT')
    c.drawString(430, y, 'REFERENCE')
    c.drawString(515, y, 'FLAG')
    c.line(40, y - 5, width - 40, y - 5)
    
    y -= 22
    c.setFont('Helvetica', 9)
    for t, val, u, r, flg in p['tests']:
        if 'Critical' in flg or 'Severe' in flg or 'High' in flg:
            c.setFillColor(colors.HexColor('#C53030'))
        else:
            c.setFillColor(colors.HexColor('#2D3748'))
        c.drawString(50, y, t)
        c.drawString(250, y, str(val))
        c.drawString(360, y, u)
        c.drawString(430, y, r)
        c.drawString(515, y, flg)
        y -= 20
    c.save()
print('Sample PDFs generated successfully in demo_app/temp/')
"

```

---

## 6. GitHub & Streamlit Community Cloud Deployment

### Step 1: Create Clean, BOM-Free Dependency Files & `.gitignore`

```powershell
# 1. Create requirements.txt without UTF-8 BOM
$reqContent = @"
streamlit
pandas
numpy
scikit-learn
shap
matplotlib
pypdf
pillow
pytesseract
joblib
reportlab
openpyxl
"@
[System.IO.File]::WriteAllText("$PWD\requirements.txt", $reqContent, (New-Object System.Text.UTF8Encoding $false))

# 2. Create packages.txt for Linux Tesseract OCR
[System.IO.File]::WriteAllText("$PWD\packages.txt", "tesseract-ocr`n", (New-Object System.Text.UTF8Encoding $false))

# 3. Create .gitignore
$ignoreContent = @"
.venv/
__pycache__/
*.pyc
demo_app/temp/
"@
[System.IO.File]::WriteAllText("$PWD\.gitignore", $ignoreContent, (New-Object System.Text.UTF8Encoding $false))

```

### Step 2: Push Code to GitHub

```powershell
git init
git add .
git commit -m "Complete MediSense AI Clinical Decision Support System"
git branch -M main
git remote add origin [https://github.com/chaitanyaskr5/MediSense-AI-Clinical-Decision-Support-System.git](https://github.com/chaitanyaskr5/MediSense-AI-Clinical-Decision-Support-System.git)
git push -u origin main

```

### Step 3: Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **Create app** and configure:
* **Repository:** `chaitanyaskr5/MediSense-AI-Clinical-Decision-Support-System`
* **Branch:** `main`
* **Main file path:** `demo_app/app.py` *(must use forward slash `/`)*


3. Click **Advanced settings** and set **Python version** to **`3.11`**.
4. Click **Save** and **Deploy**.

---

## 7. Troubleshooting & Deployment Challenges Solved

During development and cloud deployment, the following environment and runtime challenges were identified and resolved:

| # | Issue / Error Encountered | Root Cause | Permanent Solution Applied |
| --- | --- | --- | --- |
| **1** | `Permission denied: '.venv\Scripts\python.exe'` | Running `python -m venv .venv` while `.venv` was already active attempted to overwrite a locked Windows executable. | Skip `python -m venv .venv` once `(.venv)` appears in the prompt. |
| **2** | `ModuleNotFoundError: No module named 'joblib'` despite active `.venv` | `pip` resolved to the global Windows Python path (`AppData\Local\Programs\Python\Python311`) instead of `.venv`. | Explicitly run `.\.venv\Scripts\python.exe -m pip install ...` and `.\.venv\Scripts\python.exe -m streamlit run ...`. |
| **3** | `git push` rejected (`fetch first`) | GitHub repository was initialized with a remote `README.md` commit not present locally. | Used `git push -u origin main --force` to synchronize the local project cleanly. |
| **4** | `This file does not exist` on Streamlit Cloud (`demo_app\app.py`) | Streamlit Cloud runs on Linux containers, which do not recognize Windows backslashes (`\`). | Changed path to Linux forward slash: `demo_app/app.py`. |
| **5** | `Error installing requirements` on Linux | PowerShell's `Out-File -Encoding utf8` inserted invisible UTF-8 Byte Order Marks (`\xef\xbb\xbf`), breaking Linux `pip`/`apt-get`. | Rewrote `requirements.txt` and `packages.txt` using `[System.IO.File]::WriteAllText(..., (New-Object System.Text.UTF8Encoding $false))`. |
| **6** | `Oh no. Error running app` (`ModuleNotFoundError: No module named 'modules'`) | Streamlit Cloud executes from the repo root rather than inside `demo_app/`, so relative imports failed. | Added `APP_DIR = os.path.dirname(os.path.abspath(__file__))`, `sys.path.insert(0, APP_DIR)`, and `os.chdir(APP_DIR)` at the top of `demo_app/app.py`. |
| **7** | `streamlit: ERROR (spawn error)` & infinite restart loop | Including `xgboost` in `requirements.txt` pulled in ~2.5 GB of `nvidia-nccl-cu13` CUDA GPU binaries, exceeding Streamlit's 1 GB RAM limit. | Removed `xgboost` from `requirements.txt` (all models run on `scikit-learn` CPU estimators) and rebooted the app container. |
| **8** | Build stalled on `Building numpy==2.2.6` | Defaulting to Python 3.14 with a `numpy<2.3` pin forced slow C-source compilation inside a 1 GB RAM container. | Removed version pins in `requirements.txt` and selected **Python 3.11** in Streamlit Cloud's *Advanced settings* to install pre-built wheels in ~25 seconds. |

```

```
