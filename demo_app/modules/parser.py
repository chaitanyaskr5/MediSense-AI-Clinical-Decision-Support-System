import os
import re
import io
import pandas as pd
from PIL import Image

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Comprehensive clinical targets and their clinical aliases
TARGET_ALIASES = [
    ("Glucose", [r"fasting\s*(?:blood\s*)?glucose", r"\bbsl\b", r"\bfbs\b", r"\bglucose\b", r"\bbgr\b", r"avg\s*glucose", r"mean\s*glucose"]),
    ("BMI", [r"body\s*mass\s*index", r"\bbmi\b"]),
    ("sc", [r"serum\s*creatinine", r"creatinine", r"\bsc\b"]),
    ("bu", [r"blood\s*urea", r"\burea\b", r"\bbu\b"]),
    ("tot_bilirubin", [r"total\s*bilirubin", r"tot_bilirubin", r"bilirubin"]),
    ("chol", [r"total\s*cholesterol", r"cholesterol", r"\bchol\b"]),
    ("hemo", [r"hemoglobin", r"\bhemo\b", r"\bhb\b"]),
    ("SkinThickness", [r"skin\s*thickness", r"triceps"]),
    ("Insulin", [r"serum\s*insulin", r"\binsulin\b"]),
    ("DiabetesPedigreeFunction", [r"diabetes\s*pedigree", r"\bdpf\b", r"pedigree"]),
    ("Pregnancies", [r"pregnancies", r"gravida"]),
    ("thalach", [r"max\s*heart\s*rate", r"\bthalach\b", r"max\s*hr"]),
    ("oldpeak", [r"st\s*depression", r"\boldpeak\b"]),
    ("direct_bilirubin", [r"direct\s*bilirubin"]),
    ("alkphos", [r"alkaline\s*phosphatase", r"\balkphos\b", r"\balp\b"]),
    ("sgpt", [r"\bsgpt\b", r"\balt\b", r"alamine\s*aminotransferase"]),
    ("sgot", [r"\bsgot\b", r"\bast\b", r"aspartate\s*aminotransferase"]),
    ("albumin", [r"\balbumin\b"]),
    ("ag_ratio", [r"a/g\s*ratio", r"ag_ratio"])
]

def extract_text_from_file(uploaded_file) -> str:
    """Universal text reader across PDF, Images, TXT, CSV, and Excel."""
    filename = uploaded_file.name.lower()
    extracted_text = ""
    try:
        if filename.endswith(".pdf"):
            if PdfReader:
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
                    
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(uploaded_file)
            if pytesseract:
                try:
                    extracted_text = pytesseract.image_to_string(image)
                except Exception:
                    extracted_text = ""
                    
        elif filename.endswith((".csv", ".txt")):
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            # If CSV, format as readable key-values
            if filename.endswith(".csv"):
                try:
                    df = pd.read_csv(io.StringIO(extracted_text))
                    lines = []
                    for col in df.columns:
                        lines.append(f"{col}: {df[col].iloc[0]}")
                    extracted_text = "\n".join(lines)
                except Exception:
                    pass

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
            lines = []
            for col in df.columns:
                lines.append(f"{col}: {df[col].iloc[0]}")
            extracted_text = "\n".join(lines)

    except Exception as e:
        extracted_text = f"File Read Error: {e}"

    return extracted_text

def parse_medical_record(text: str) -> dict:
    """Universal biomarker extractor for horizontal, vertical, and unstructured clinical layouts."""
    data = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. Patient Name (skips header tokens like 'Triage', 'System', 'Laboratory')
    for line in lines:
        m_name = re.search(r"(?:Patient\s*Name|Name\s*of\s*Patient|Patient)\s*[:\-]\s*([^\n\r\|]+)", line, re.IGNORECASE)
        if m_name:
            cand = m_name.group(1).strip()
            cand = re.split(r"[|\n\t]", cand)[0].strip()
            if cand and not any(w in cand.lower() for w in ["triage", "system", "laboratory", "division", "accredited", "profile", "panel"]):
                data["patient_name"] = cand
                break

    # 2. Patient Age
    for line in lines:
        m_age = re.search(r"(?:Patient\s*)?Age\s*[:\-]?\s*([0-9]{1,3})", line, re.IGNORECASE)
        if m_age:
            val = float(m_age.group(1))
            data["Age"] = val
            data["age"] = val
            break

    # 3. Blood Pressure (strictly isolates observed BP like '168/102' from reference ranges like '90-120/60-80')
    bp_match = re.findall(r"(?<![\d\-])\b([0-9]{2,3})\s*/\s*[0-9]{2,3}\b", text)
    if bp_match:
        val = float(bp_match[0])
        data["BloodPressure"] = val
        data["bp"] = val
        data["trestbps"] = val
    else:
        m_bp_single = re.search(r"(?:Blood\s*Pressure|diastolic\s*bp|resting\s*bp|trestbps|bp)[\s:\n\t]*([0-9]{2,3})", text, re.IGNORECASE)
        if m_bp_single:
            val = float(m_bp_single.group(1))
            data["BloodPressure"] = val
            data["bp"] = val
            data["trestbps"] = val

    # 4. Biomarkers - Inline Matching (e.g. Glucose: 180, Glucose = 180)
    for key, aliases in TARGET_ALIASES:
        if key in data:
            continue
        for alias in aliases:
            m = re.search(rf"(?:{alias})[\s:=\(\)]*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    data[key] = val
                    if key == "Glucose":
                        data["avg_glucose_level"] = val
                    elif key == "BMI":
                        data["bmi"] = val
                    break
                except ValueError:
                    pass

    # 5. Biomarkers - Vertical Table Block Matching (stripping reference ranges like 70-99 and <200)
    for key, aliases in TARGET_ALIASES:
        if key in data:
            continue
        for alias in aliases:
            m_block = re.search(rf"(?:{alias})[\s\S]{{1,60}}", text, re.IGNORECASE)
            if m_block:
                snip = m_block.group(0)
                snip_clean = re.sub(r"[0-9]+(?:\.[0-9]+)?\s*-\s*[0-9]+(?:\.[0-9]+)?", " ", snip)
                snip_clean = re.sub(r"[<>]\s*[0-9]+(?:\.[0-9]+)?", " ", snip_clean)
                nums = re.findall(r"\b[0-9]+(?:\.[0-9]+)?\b", snip_clean)
                if nums:
                    val = float(nums[0])
                    data[key] = val
                    if key == "Glucose":
                        data["avg_glucose_level"] = val
                    elif key == "BMI":
                        data["bmi"] = val
                    break

    return data