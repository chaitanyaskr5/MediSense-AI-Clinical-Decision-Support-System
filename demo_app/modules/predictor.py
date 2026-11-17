import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

def load_disease_model(disease: str):
    path = os.path.join(MODELS_DIR, f"{disease.lower()}_model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def evaluate_patient(disease: str, user_inputs: dict):
    bundle = load_disease_model(disease)
    if not bundle:
        raise FileNotFoundError(f"Model for {disease} not found. Please train models first.")
        
    model = bundle["model"]
    features = bundle["features"]
    medians = bundle.get("medians", {})
    
    # Impute missing values from training population medians
    input_row = {}
    for feat in features:
        if feat in user_inputs and user_inputs[feat] is not None:
            input_row[feat] = float(user_inputs[feat])
        else:
            input_row[feat] = float(medians.get(feat, 0.0))
            
    df_input = pd.DataFrame([input_row])
    
    # Probability prediction
    proba = float(model.predict_proba(df_input)[0, 1])
    
    # Confidence calibration: margin distance from decision boundary
    confidence_score = float(abs(proba - 0.5) * 2.0)
    
    # Validation Tiers (Safety Gate)
    if confidence_score >= 0.70:
        action_tier = "Tier 1: High Confidence"
        recommendation = "Clear clinical biomarker signal. Automated triage accepted."
        badge = "success"
    elif confidence_score >= 0.35:
        action_tier = "Tier 2: Moderate Uncertainty"
        recommendation = "Borderline clinical profile. Secondary clinical review recommended."
        badge = "warning"
    else:
        action_tier = "Tier 3: Low Confidence / Inconclusive"
        recommendation = "Biomarker variance elevated. Escalate for secondary diagnostic workup."
        badge = "danger"
        
    return {
        "risk_probability": proba,
        "confidence_score": confidence_score,
        "action_tier": action_tier,
        "recommendation": recommendation,
        "badge": badge,
        "df_input": df_input,
        "features": features,
        "metrics": bundle["metrics"]
    }