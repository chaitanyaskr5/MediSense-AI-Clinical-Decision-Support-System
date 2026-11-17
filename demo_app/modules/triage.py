import os
import re
import pandas as pd
from modules.parser import extract_text_from_file, parse_medical_record
from modules.predictor import evaluate_patient

DISEASE_KEYS = ["Kidney", "Heart", "Stroke", "Diabetes", "Liver"]

LOCAL_LABELS = {
    "English": {
        "Kidney": "Chronic Kidney Disease",
        "Heart": "Acute Heart Disease",
        "Stroke": "Stroke Risk",
        "Diabetes": "Severe Type 2 Diabetes",
        "Liver": "Hepatic/Liver Disorder"
    },
    "हिंदी": {
        "Kidney": "क्रोनिक किडनी रोग (गुर्दे की बीमारी)",
        "Heart": "गंभीर हृदय रोग",
        "Stroke": "स्ट्रोक / पक्षाघात जोखिम",
        "Diabetes": "गंभीर मधुमेह (डायबिटीज)",
        "Liver": "यकृत / लिवर विकार"
    },
    "मराठी": {
        "Kidney": "मूत्रपिंडाचा जुनाट आजार (किडनी विकार)",
        "Heart": "गंभीर हृदयविकार",
        "Stroke": "पक्षाघात (स्ट्रोक) जोखीम",
        "Diabetes": "गंभीर मधुमेह (डायबेटिस)",
        "Liver": "यकृताचा आजार (लिव्हर विकार)"
    }
}

def process_batch_reports(uploaded_files, target_condition: str = "All", language: str = "English") -> pd.DataFrame:
    """
    Evaluates every patient against all 5 machine learning models.
    Ranks patients based on maximum predicted disease probability,
    comorbidity burden, and model certainty.
    """
    records = []

    for file in uploaded_files:
        raw_text = extract_text_from_file(file)
        biomarkers = parse_medical_record(raw_text)

        # 1. Resolve Patient Name
        patient_name = biomarkers.get("patient_name")
        if not patient_name:
            base = os.path.splitext(file.name)[0]
            cleaned = re.sub(r"(?:patient[_\-\s]*)?(?:critical|urgent|severe|healthy|normal|moderate|high)?[_\-\s]*", "", base, flags=re.IGNORECASE)
            patient_name = cleaned.replace("_", " ").replace("-", " ").strip().title()

        # 2. Evaluate risk across ALL 5 ML models
        model_risks = {}
        model_confs = {}

        for d in DISEASE_KEYS:
            try:
                eval_out = evaluate_patient(d, biomarkers)
                model_risks[d] = eval_out["risk_probability"] * 100.0
                model_confs[d] = eval_out["confidence_score"] * 100.0
            except Exception:
                model_risks[d] = 0.0
                model_confs[d] = 0.0

        # 3. Determine Highest-Risk Disease & Comorbidity Count
        primary_disease = max(model_risks, key=model_risks.get)
        max_disease_risk = model_risks[primary_disease]
        highest_conf = model_confs[primary_disease]

        # Comorbidity: number of diseases with predicted risk >= 50%
        active_conditions = [d for d, r in model_risks.items() if r >= 50.0]
        comorbidity_count = len(active_conditions)

        # 4. Disease Risk-Based Composite Score (0 to 100)
        # Base: Max disease risk (70%)
        # Comorbidity escalation: +8 points per additional compromised organ system (up to +20%)
        # Model certainty weight: (10%)
        comorbidity_bonus = min(20.0, max(0, comorbidity_count - 1) * 8.0)
        composite_urgency = (max_disease_risk * 0.70) + comorbidity_bonus + (highest_conf * 0.10)
        final_urgency_score = min(100.0, max(5.0, round(composite_urgency, 1)))

        # 5. Triage Priority & Localized Clinical Explanation
        localized_disease_name = LOCAL_LABELS.get(language, LOCAL_LABELS["English"]).get(primary_disease, primary_disease)

        if final_urgency_score >= 65.0:
            priority_tag = "🔴 Priority 1 (Immediate)"
            if language == "मराठी":
                explanation = f"तातडीचे वैद्यकीय उपचार आवश्यक! प्राथमिक आजार: {localized_disease_name} (जोखीम: {max_disease_risk:.1f}%). बाधित अवयव प्रणाली: {comorbidity_count} आजार."
            elif language == "हिंदी":
                explanation = f"तत्काल चिकित्सीय हस्तक्षेप आवश्यक! प्राथमिक रोग: {localized_disease_name} (जोखिम: {max_disease_risk:.1f}%). प्रभावित अंग प्रणाली: {comorbidity_count} रोग।"
            else:
                explanation = f"Immediate clinical intervention required! Primary risk: {localized_disease_name} ({max_disease_risk:.1f}%). Multiple comorbidity burden: {comorbidity_count} conditions."
        elif final_urgency_score >= 35.0:
            priority_tag = "🟡 Priority 2 (Urgent)"
            if language == "मराठी":
                explanation = f"मध्यम क्लिनिकल जोखीम. प्राथमिक तपासणी: {localized_disease_name} ({max_disease_risk:.1f}%). २४ तासांत डॉक्टर तपासणी आवश्यक."
            elif language == "हिंदी":
                explanation = f"मध्यम क्लिनिकल जोखिम। प्राथमिक स्थिति: {localized_disease_name} ({max_disease_risk:.1f}%). 24 घंटे में विशेषज्ञ समीक्षा अनुशंसित।"
            else:
                explanation = f"Urgent clinical evaluation indicated. Primary condition: {localized_disease_name} ({max_disease_risk:.1f}%). Assess within 24 hours."
        else:
            priority_tag = "🟢 Priority 3 (Routine)"
            if language == "मराठी":
                explanation = f"सर्व ५ आजारांची जोखीम सामान्य मर्यादेत आहे (कमाल जोखीम: {max_disease_risk:.1f}%). नियमित ओपीडी पाठपुरावा पुरेसा."
            elif language == "हिंदी":
                explanation = f"सभी 5 रोग मॉडल सामान्य सीमा में हैं (अधिकतम जोखिम: {max_disease_risk:.1f}%). नियमित बाह्य रोगी विभाग (OPD) समीक्षा पर्याप्त है।"
            else:
                explanation = f"All 5 disease risk scores in low/controlled range (Max: {max_disease_risk:.1f}%). Routine outpatient monitoring."

        records.append({
            "File Name": file.name,
            "Patient Name": patient_name,
            "Urgency Score": final_urgency_score,
            "Triage Priority": priority_tag,
            "Primary Disease Risk": f"{localized_disease_name} ({max_disease_risk:.1f}%)",
            "Kidney Risk": f"{model_risks['Kidney']:.1f}%",
            "Heart Risk": f"{model_risks['Heart']:.1f}%",
            "Stroke Risk": f"{model_risks['Stroke']:.1f}%",
            "Diabetes Risk": f"{model_risks['Diabetes']:.1f}%",
            "Liver Risk": f"{model_risks['Liver']:.1f}%",
            "Clinical Explanation": explanation
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="Urgency Score", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"
    return df