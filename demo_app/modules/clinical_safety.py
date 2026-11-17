def get_drug_contraindications(biomarkers: dict) -> list:
    """
    Evaluates patient organ biomarker levels against clinical pharmacology safety rules.
    Returns targeted contraindication warnings and dosing precautions.
    """
    warnings = []

    # 1. Renal / Kidney Safety Check
    sc = biomarkers.get("sc", 0.0)
    bu = biomarkers.get("bu", 0.0)
    if sc >= 2.0 or bu >= 55.0:
        warnings.append({
            "drug_class": "Metformin & SGLT2 Inhibitors",
            "severity": "CRITICAL",
            "reason": f"Severe renal retention detected (Creatinine {sc:.1f} mg/dL, Urea {bu:.0f} mg/dL). High risk of lactic acidosis.",
            "guidance": "Discontinue Metformin immediately. Shift glycemic management to insulin therapy."
        })
        warnings.append({
            "drug_class": "NSAIDs (Ibuprofen, Naproxen, Diclofenac)",
            "severity": "CRITICAL",
            "reason": "Risk of acute tubular necrosis and sudden decline in glomerular filtration rate (GFR).",
            "guidance": "Avoid NSAIDs. Use low-dose Paracetamol or non-renal analgesics for pain management."
        })
    elif sc >= 1.4:
        warnings.append({
            "drug_class": "ACE Inhibitors / ARBs",
            "severity": "WARNING",
            "reason": f"Borderline renal insufficiency (Creatinine {sc:.1f} mg/dL). Risk of hyperkalemia.",
            "guidance": "Monitor serum potassium and renal panel closely if titrating dose."
        })

    # 2. Hepatic / Liver Safety Check
    bili = biomarkers.get("tot_bilirubin", 0.0)
    sgpt = biomarkers.get("sgpt", 0.0)
    sgot = biomarkers.get("sgot", 0.0)
    if bili >= 2.5 or sgpt >= 80.0 or sgot >= 80.0:
        warnings.append({
            "drug_class": "High-dose Paracetamol / Acetaminophen",
            "severity": "CRITICAL",
            "reason": f"Active hepatic injury/jaundice detected (Bilirubin {bili:.1f} mg/dL, SGPT {sgpt:.0f} U/L).",
            "guidance": "Strictly cap dosage to maximum 2g/24hr or avoid entirely."
        })
        warnings.append({
            "drug_class": "Statins (Atorvastatin, Rosuvastatin)",
            "severity": "WARNING",
            "reason": "Elevated baseline transaminases indicate hepatic inflammation.",
            "guidance": "Hold or adjust statin regimen until liver enzymes stabilize."
        })

    # 3. Glycemic / Metabolic Emergency Check
    glucose = biomarkers.get("Glucose", biomarkers.get("avg_glucose_level", 0.0))
    if glucose >= 250.0:
        warnings.append({
            "drug_class": "Sulfonylureas (Glimepiride, Glibenclamide)",
            "severity": "WARNING",
            "reason": f"Severe acute hyperglycemia ({glucose:.0f} mg/dL). Risk of unpredictable hypoglycemic rebound.",
            "guidance": "Initiate sliding scale IV regular insulin protocol under inpatient observation."
        })

    # 4. Cardiovascular / Hypertensive Crisis Check
    bp = biomarkers.get("BloodPressure", biomarkers.get("bp", biomarkers.get("trestbps", 0.0)))
    if bp >= 170.0:
        warnings.append({
            "drug_class": "Sympathomimetics / Decongestants (Pseudoephedrine)",
            "severity": "CRITICAL",
            "reason": f"Severe hypertension ({bp:.0f} mmHg). Risk of precipitating intracranial hemorrhage.",
            "guidance": "Strictly avoid decongestant vasoconstrictors. Initiate rapid-acting antihypertensives."
        })

    return warnings

def determine_ward_routing(biomarkers: dict, highest_disease: str, urgency_score: float) -> dict:
    """
    Determines specialist hospital ward allocation and clinical referral prioritization.
    """
    sc = biomarkers.get("sc", 0.0)
    bp = biomarkers.get("BloodPressure", biomarkers.get("bp", biomarkers.get("trestbps", 0.0)))
    bili = biomarkers.get("tot_bilirubin", 0.0)
    glucose = biomarkers.get("Glucose", biomarkers.get("avg_glucose_level", 0.0))

    if urgency_score >= 65.0:
        bed_type = "ICU / High Dependency Unit (HDU)"
        priority_level = "Level 1: Emergency Resuscitation"
    elif urgency_score >= 35.0:
        bed_type = "Monitored Step-Down / Acute Care Bed"
        priority_level = "Level 2: Urgent Inpatient Admission"
    else:
        bed_type = "Outpatient Department (OPD) Consultation"
        priority_level = "Level 3: Routine Ambulatory Care"

    # Department mapping
    if highest_disease == "Kidney" or sc >= 2.5:
        specialty = "Nephrology & Renal Replacement Care"
        recommended_actions = "Immediate nephrology consult, 24-hr creatinine clearance, prepare for possible hemodialysis."
    elif highest_disease == "Heart" or bp >= 165.0:
        specialty = "Cardiology Intensive Care (CCU)"
        recommended_actions = "Continuous 12-lead cardiac telemetry, emergency echocardiogram, bedside cardiac troponin test."
    elif highest_disease == "Stroke":
        specialty = "Neurology Acute Stroke Care"
        recommended_actions = "STAT non-contrast brain CT scan, NIHSS neurological deficit evaluation, maintain cerebral perfusion."
    elif highest_disease == "Liver" or bili >= 2.5:
        specialty = "Gastroenterology & Hepatology Ward"
        recommended_actions = "Abdominal ultrasound, coagulation profile (INR/PT), monitoring for hepatic encephalopathy."
    else:
        specialty = "Endocrinology & Internal Medicine Unit"
        recommended_actions = "Serum ketone testing, arterial blood gas (ABG) for ketoacidosis check, fluid rehydration protocol."

    return {
        "specialty": specialty,
        "bed_type": bed_type,
        "priority_level": priority_level,
        "recommended_actions": recommended_actions
    }