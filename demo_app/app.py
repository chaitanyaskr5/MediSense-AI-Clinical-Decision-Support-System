import os
import streamlit as st
import pandas as pd
from modules.parser import extract_text_from_file, parse_medical_record
from modules.predictor import evaluate_patient, load_disease_model
from modules.explain import generate_shap_plot
from modules.triage import process_batch_reports
from modules.clinical_safety import get_drug_contraindications, determine_ward_routing
import train_all

st.set_page_config(
    page_title="MediSense AI - Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-mode adaptive CSS
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 16px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
        opacity: 0.85;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Multilingual Localization Dictionary
# -------------------------------------------------------------
I18N = {
    "English": {
        "title": "MediSense AI: Clinical Decision Support Platform",
        "caption": "Multimodal Medical Document Extraction, Multi-Disease Risk Prediction & Explainable AI (XAI)",
        "sidebar_header": "Navigation & Controls",
        "theme_toggle": "Dark Mode for Charts",
        "disease_label": "Target Medical Condition",
        "retrain_btn": "Retrain All Disease Models",
        "retrain_success": "Models retrained and serialized successfully!",
        "tab_single": "Single Patient Intake & Diagnosis",
        "tab_batch": "🚨 Batch Triage & Patient Priority Ranking",
        "tab_benchmarks": "Validation & Benchmark Metrics",
        "doc_section": "1. Multimodal Document Ingestion",
        "upload_label": "Upload Lab Report, Scan, or Medical Summary (PDF, PNG, JPG, TXT)",
        "extracted_content": "Extracted Content",
        "auto_extracted_msg": "Auto-extracted {n} clinical parameters from document.",
        "no_params_msg": "No mapped parameters identified. Use manual input or sample profiles.",
        "demo_presets": "Instant Demo Presets",
        "btn_high_risk": "Load High-Risk Profile",
        "btn_healthy": "Load Healthy Profile",
        "biomarker_section": "2. Biomarker Parameter Review",
        "model_not_found": "Model artifact not found. Please click 'Retrain All Disease Models'.",
        "execute_btn": "Execute Clinical Assessment & Explainability",
        "results_header": "3. Diagnostic Results & Safety Stratification",
        "metric_prob": "Predicted Disease Probability",
        "metric_conf": "Prediction Certainty (Confidence)",
        "metric_tier": "Safety Triage Level",
        "status_high": "HIGH RISK / CRITICAL ANOMALY",
        "status_mid": "MODERATE CLINICAL RISK",
        "status_low": "LOW RISK / NORMAL RANGE",
        "xai_header": "4. Explainable AI: Biomarker Attribution",
        "xai_caption": "Quantifies the exact push of each biomarker on the final risk prediction relative to population baseline.",
        "xai_guide": """**Clinical Interpretation Guide:**
* **Positive Values (Extending Right):** Biomarkers actively driving the risk probability upwards above standard baseline.
* **Negative Values (Extending Left):** Normal or protective physiological values pulling the prediction toward the healthy class.""",
        "benchmark_header": "Model Benchmark Validation Table (Held-Out Test Sets)",
        "benchmark_caption": "Evaluated using stratified cross-validation and independent test splits on clinical ground-truth cohorts.",
        "tier1_rec": "Clear clinical biomarker signal. Automated triage accepted.",
        "tier2_rec": "Borderline clinical profile. Secondary clinical review recommended.",
        "tier3_rec": "Biomarker variance elevated. Escalate for secondary diagnostic workup.",
        "comorbidity_btn": "🔍 Run Full Comorbidity Risk Scan (All Diseases)",
        "comorbidity_title": "Patient Cross-Condition Comorbidity Radar",
        "download_report_btn": "📥 Download Clinical Audit Report (.txt)",
        "what_if_header": "🔬 Interactive 'What-If' Intervention Simulator",
        "what_if_caption": "Simulate how modifying physiological factors impacts the risk probability in real-time.",
        "batch_upload_label": "Upload Multiple Patient Reports (Unlimited PDFs, Images, or Text)",
        "batch_run_btn": "⚡ Run High-Throughput Batch Triage",
        "batch_table_title": "📋 Prioritized Clinical Triage Queue (Ranked Highest to Lowest Urgency)",
        "batch_download_btn": "📥 Download Prioritized Ward Triage List (.csv)"
    },
    "हिंदी": {
        "title": "मेडीसेंस एआई: क्लिनिकल निर्णय समर्थन प्रणाली (MediSense AI)",
        "caption": "मल्टीमॉडल मेडिकल दस्तावेज़ विश्लेषण, बहु-रोग जोखिम भविष्यवाणी और व्याख्या योग्य एआई (XAI)",
        "sidebar_header": "नेविगेशन और नियंत्रण",
        "theme_toggle": "चार्ट के लिए डार्क मोड",
        "disease_label": "लक्षित रोग चुनें",
        "retrain_btn": "सभी रोग मॉडलों को पुनः प्रशिक्षित करें",
        "retrain_success": "सभी मॉडल सफलतापूर्वक पुनः प्रशिक्षित और सहेजे गए!",
        "tab_single": "एकल रोगी परीक्षण और निदान",
        "tab_batch": "🚨 बैच प्राथमिकता और वार्ड ट्राइएज रैंकिंग",
        "tab_benchmarks": "मॉडल सत्यापन और बेंचमार्क मेट्रिक्स",
        "doc_section": "1. मेडिकल दस्तावेज़ अपलोड एवं विश्लेषण",
        "upload_label": "लैब रिपोर्ट, स्कैन या सारांश अपलोड करें (PDF, PNG, JPG, TXT)",
        "extracted_content": "निकाली गई सामग्री",
        "auto_extracted_msg": "दस्तावेज़ से {n} बायोमार्कर सफलतापूर्वक निकाले गए।",
        "no_params_msg": "कोई पैरामीटर नहीं मिला। कृपया मैन्युअल रूप से दर्ज करें।",
        "demo_presets": "त्वरित डेमो प्रोफाइल",
        "btn_high_risk": "उच्च जोखिम प्रोफाइल लोड करें",
        "btn_healthy": "सामान्य/स्वस्थ प्रोफाइल लोड करें",
        "biomarker_section": "2. क्लिनिकल बायोमार्कर समीक्षा",
        "model_not_found": "मॉडल नहीं मिला। कृपया साइडबार से मॉडल पुनः प्रशिक्षित करें।",
        "execute_btn": "क्लिनिकल मूल्यांकन और व्याख्या (XAI) चलाएं",
        "results_header": "3. नैदानिक परिणाम और सुरक्षा स्तरीकरण",
        "metric_prob": "रोग की अनुमानित संभावना",
        "metric_conf": "भविष्यवाणी की निश्चितता (विश्वास स्कोर)",
        "metric_tier": "सुरक्षा प्राथमिकता स्तर",
        "status_high": "उच्च जोखिम / गंभीर स्थिति",
        "status_mid": "मध्यम क्लिनिकल जोखिम",
        "status_low": "कम जोखिम / सामान्य श्रेणी",
        "xai_header": "4. व्याख्या योग्य एआई: बायोमार्कर योगदान (XAI)",
        "xai_caption": "यह चार्ट दिखाता है कि प्रत्येक बायोमार्कर ने जोखिम को कितना बढ़ाया या घटाया है।",
        "xai_guide": """**क्लिनिकल व्याख्या गाइड:**
* **सकारात्मक मान (दाईं ओर):** वे बायोमार्कर जो रोग के जोखिम को सामान्य से अधिक बढ़ा रहे हैं।
* **नकारात्मक मान (बाईं ओर):** सामान्य या सुरक्षात्मक बायोमार्कर जो जोखिम को कम कर रहे हैं।""",
        "benchmark_header": "मॉडल बेंचमार्क सत्यापन तालिका (परीक्षण डेटा)",
        "benchmark_caption": "सत्यापित क्लिनिकल डेटासेट पर स्तरीकृत क्रॉस-सत्यापन द्वारा मूल्यांकित।",
        "tier1_rec": "स्पष्ट क्लिनिकल संकेत। स्वचालित मूल्यांकन मान्य है।",
        "tier2_rec": "सीमांत क्लिनिकल प्रोफाइल। डॉक्टर द्वारा द्वितीयक समीक्षा की सलाह दी जाती है।",
        "tier3_rec": "उच्च बायोमार्कर विचलन। विस्तृत जांच के लिए विशेषज्ञ को रेफर करें।",
        "comorbidity_btn": "🔍 पूर्ण सह-रुग्णता (Comorbidity) जोखिम स्कैन चलाएं",
        "comorbidity_title": "रोगी का बहु-रोग सह-रुग्णता रडार",
        "download_report_btn": "📥 क्लिनिकल ऑडिट रिपोर्ट डाउनलोड करें (.txt)",
        "what_if_header": "🔬 इंटरएक्टिव 'What-If' हस्तक्षेप सिम्युलेटर",
        "what_if_caption": "देखें कि बायोमार्कर में बदलाव से जोखिम संभावना कैसे बदलती है।",
        "batch_upload_label": "एक साथ कई मरीजों की रिपोर्ट अपलोड करें (PDF, Images, Text)",
        "batch_run_btn": "⚡ बैच ट्राइएज प्राथमिकता क्रम चलाएं",
        "batch_table_title": "📋 प्राथमिकता प्राप्त रोगी सूची (सबसे गंभीर मरीज पहले)",
        "batch_download_btn": "📥 वार्ड ट्राइएज प्राथमिकता सूची डाउनलोड करें (.csv)"
    },
    "मराठी": {
        "title": "मेडीसेन्स एआय: क्लिनिकल निर्णय समर्थन प्रणाली (MediSense AI)",
        "caption": "वैद्यकीय अहवाल विश्लेषण, बहु-आजार जोखीम अंदाज आणि स्पष्टीकरणात्मक एआय (XAI)",
        "sidebar_header": "नियंत्रण आणि पर्याय",
        "theme_toggle": "चार्टसाठी डार्क मोड",
        "disease_label": "आजार निवडा",
        "retrain_btn": "सर्व मॉडेल्स पुन्हा प्रशिक्षित करा",
        "retrain_success": "सर्व मॉडेल्स यशस्वीरित्या प्रशिक्षित आणि सेव्ह झाले!",
        "tab_single": "एकल रुग्ण तपासणी आणि निदान",
        "tab_batch": "🚨 एकाच वेळी अनेक रुग्णांचे ट्राइएज आणि रँकिंग",
        "tab_benchmarks": "मॉडेल पडताळणी आणि बेंचमार्क",
        "doc_section": "१. वैद्यकीय दस्तऐवज अपलोड आणि विश्लेषण",
        "upload_label": "लॅब रिपोर्ट, स्कॅन किंवा अहवाल अपलोड करा (PDF, PNG, JPG, TXT)",
        "extracted_content": "दस्तऐवजातील मजकूर",
        "auto_extracted_msg": "अहवालातून {n} बायोमार्कर्स यशस्वीरित्या काढण्यात आले.",
        "no_params_msg": "पॅरामीटर्स आढळले नाहीत. कृपया स्वतः मूल्ये भरा.",
        "demo_presets": "डेमो प्रोफाईल्स",
        "btn_high_risk": "उच्च जोखीम प्रोफाइल लोड करा",
        "btn_healthy": "निरोगी प्रोफाइल लोड करा",
        "biomarker_section": "२. क्लिनिकल घटकांची पडताळणी",
        "model_not_found": "मॉडेल उपलब्ध नाही. कृपया आधी मॉडेल्स प्रशिक्षित करा.",
        "execute_btn": "क्लिनिकल मूल्यांकन आणि XAI सुरू करा",
        "results_header": "३. तपासणी निष्कर्ष आणि सुरक्षितता वर्गवारी",
        "metric_prob": "आजाराची संभाव्यता (जोखीम)",
        "metric_conf": "अंदाजाची अचूकता (आत्मविश्वास)",
        "metric_tier": "सुरक्षा श्रेणी (टियर)",
        "status_high": "उच्च जोखीम / गंभीर परिस्थिती",
        "status_mid": "मध्यम क्लिनिकल जोखीम",
        "status_low": "कमी जोखीम / सामान्य श्रेणी",
        "xai_header": "४. स्पष्टीकरणात्मक एआय: घटकांचे योगदान (XAI)",
        "xai_caption": "प्रत्येक वैद्यकीय घटकाने जोखीम वाढवण्यात किंवा कमी करण्यात किती वाटा उचलला हे दर्शवते.",
        "xai_guide": """**क्लिनिकल मार्गदर्शक:**
* **उजवीकडील घटक (Positive):** हे घटक आजाराचा धोका वाढवत आहेत.
* **डावीकडील घटक (Negative):** हे घटक सामान्य पातळीवर असून धोका कमी करत आहेत.""",
        "benchmark_header": "मॉडेल पडताळणी तक्ता (टेस्ट डेटा)",
        "benchmark_caption": "मानक क्लिनिकल डेटासेटवर तपासलेले निष्कर्ष.",
        "tier1_rec": "स्पष्ट क्लिनिकल लक्षणे. स्वयंचलित निष्कर्ष ग्राह्य धरला जाऊ शकतो.",
        "tier2_rec": "मध्यम पातळी. तज्ज्ञ डॉक्टरांकडून पुनर्पडताळणी आवश्यक.",
        "tier3_rec": "घटकांमध्ये तफावत. रुग्णाला पुढील सखोल तपासणीसाठी पाठवा.",
        "comorbidity_btn": "🔍 सर्व आजारांचे एकत्रित (Comorbidity) स्कॅन करा",
        "comorbidity_title": "रुग्णाचा बहु-आजार जोखीम निर्देशांक",
        "download_report_btn": "📥 क्लिनिकल तपासणी अहवाल डाउनलोड करा (.txt)",
        "what_if_header": "🔬 परस्परसंवादी 'What-If' सिम्युलेटर",
        "what_if_caption": "घटकांमध्ये बदल केल्यास आजाराचा धोका कसा बदलतो ते तपासा.",
        "batch_upload_label": "एकाच वेळी अनेक रुग्णांचे अहवाल अपलोड करा (PDF, Images, Text)",
        "batch_run_btn": "⚡ रुग्ण अग्रक्रम क्रमवारी (Triage) तयार करा",
        "batch_table_title": "📋 उपचारासाठी प्राधान्यक्रम यादी (सर्वात गंभीर रुग्ण प्रथम)",
        "batch_download_btn": "📥 वॉर्ड ट्राइएज प्राधान्यक्रम यादी डाउनलोड करा (.csv)"
    }
}

DISEASE_TRANSLATIONS = {
    "English": {"Diabetes": "Diabetes", "Kidney": "Kidney Disease (CKD)", "Liver": "Liver Disease (ILPD)", "Stroke": "Stroke Risk", "Heart": "Heart Disease"},
    "हिंदी": {"Diabetes": "मधुमेह (Diabetes)", "Kidney": "गुर्दे की बीमारी (Kidney / CKD)", "Liver": "यकृत रोग (Liver / ILPD)", "Stroke": "स्ट्रोक / पक्षाघात (Stroke)", "Heart": "हृदय रोग (Heart Disease)"},
    "मराठी": {"Diabetes": "मधुमेह (Diabetes)", "Kidney": "मूत्रपिंडाचे आजार (Kidney / CKD)", "Liver": "यकृताचे आजार (Liver / ILPD)", "Stroke": "पक्षाघात (Stroke)", "Heart": "हृदयविकार (Heart Disease)"}
}

if "user_inputs" not in st.session_state:
    st.session_state["user_inputs"] = {}

# Sidebar Controls
st.sidebar.header("MediSense AI")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "हिंदी", "मराठी"])
T = I18N[selected_lang]

chart_dark_mode = st.sidebar.checkbox(T["theme_toggle"], value=True)

st.sidebar.subheader(T["sidebar_header"])
disease_keys = ["Diabetes", "Kidney", "Liver", "Stroke", "Heart"]
disease_display = [DISEASE_TRANSLATIONS[selected_lang][k] for k in disease_keys]
selected_idx = st.sidebar.selectbox(T["disease_label"], range(len(disease_keys)), format_func=lambda i: disease_display[i])
disease_option = disease_keys[selected_idx]

if st.sidebar.button(T["retrain_btn"], type="primary"):
    with st.spinner("Training models..."):
        df_summary = train_all.train_and_save_all()
        st.sidebar.success(T["retrain_success"])
        st.sidebar.dataframe(df_summary, hide_index=True)

st.title(T["title"])
st.caption(T["caption"])

tab_single, tab_batch, tab_benchmark = st.tabs([
    T["tab_single"], 
    T["tab_batch"], 
    T["tab_benchmarks"]
])

# -------------------------------------------------------------
# TAB 1: Single Patient Assessment
# -------------------------------------------------------------
with tab_single:
    col_file, col_form = st.columns([1, 1], gap="large")
    
    parsed_dict = {}
    with col_file:
        st.markdown(f"### {T['doc_section']}")
        uploaded = st.file_uploader(T["upload_label"], type=["pdf", "png", "jpg", "jpeg", "txt", "csv", "xlsx"])
        
        if uploaded:
            raw_text = extract_text_from_file(uploaded)
            with st.expander(T["extracted_content"], expanded=False):
                st.text_area(T["extracted_content"], raw_text, height=180)
            parsed_dict = parse_medical_record(raw_text)
            if parsed_dict:
                st.success(T["auto_extracted_msg"].format(n=len(parsed_dict)))
            else:
                st.info(T["no_params_msg"])

        st.markdown("---")
        st.markdown(f"#### {T['demo_presets']}")
        p1, p2 = st.columns(2)
        
        if p1.button(T["btn_high_risk"], use_container_width=True):
            st.session_state["user_inputs"] = {
                "Glucose": 185.0, "BloodPressure": 94.0, "SkinThickness": 35.0, "Insulin": 210.0, "BMI": 36.4, "DiabetesPedigreeFunction": 0.68, "Age": 62.0, "Pregnancies": 4.0,
                "age": 64.0, "bp": 90.0, "sg": 1.010, "al": 3.0, "su": 2.0, "bgr": 210.0, "bu": 65.0, "sc": 2.8, "sod": 132.0, "pot": 5.4, "hemo": 9.4, "pcv": 30.0, "wc": 9800.0, "rc": 3.2,
                "tot_bilirubin": 3.4, "direct_bilirubin": 1.4, "tot_proteins": 6.1, "albumin": 2.8, "ag_ratio": 0.70, "sgpt": 92.0, "sgot": 110.0, "alkphos": 240.0,
                "chol": 265.0, "trestbps": 160.0, "thalach": 112.0, "oldpeak": 2.8, "cp": 2.0,
                "avg_glucose_level": 215.0, "hypertension": 1.0, "heart_disease": 1.0, "bmi": 36.4
            }
            st.rerun()

        if p2.button(T["btn_healthy"], use_container_width=True):
            st.session_state["user_inputs"] = {
                "Glucose": 88.0, "BloodPressure": 70.0, "SkinThickness": 20.0, "Insulin": 75.0, "BMI": 22.4, "DiabetesPedigreeFunction": 0.25, "Age": 28.0, "Pregnancies": 0.0,
                "age": 28.0, "bp": 70.0, "sg": 1.025, "al": 0.0, "su": 0.0, "bgr": 92.0, "bu": 22.0, "sc": 0.8, "sod": 140.0, "pot": 4.2, "hemo": 14.8, "pcv": 44.0, "wc": 6500.0, "rc": 5.1,
                "tot_bilirubin": 0.7, "direct_bilirubin": 0.2, "tot_proteins": 7.4, "albumin": 4.2, "ag_ratio": 1.30, "sgpt": 24.0, "sgot": 22.0, "alkphos": 85.0,
                "chol": 175.0, "trestbps": 118.0, "thalach": 165.0, "oldpeak": 0.1, "cp": 0.0,
                "avg_glucose_level": 85.0, "hypertension": 0.0, "heart_disease": 0.0, "bmi": 22.4
            }
            st.rerun()

    with col_form:
        st.markdown(f"### {T['biomarker_section']}: {DISEASE_TRANSLATIONS[selected_lang][disease_option]}")
        bundle = load_disease_model(disease_option)
        if bundle is None:
            st.error(T["model_not_found"])
            st.stop()
            
        features = bundle["features"]
        medians = bundle.get("medians", {})
        display_feats = features[:8]
        current_inputs = {}
        
        c1, c2 = st.columns(2)
        for idx, feat in enumerate(display_feats):
            default_val = float(
                parsed_dict.get(feat, 
                st.session_state["user_inputs"].get(feat, 
                medians.get(feat, 0.0)))
            )
            col_target = c1 if idx % 2 == 0 else c2
            current_inputs[feat] = col_target.number_input(f"{feat}", value=default_val, format="%.2f")

    st.markdown("---")
    eval_clicked = st.button(T["execute_btn"], type="primary", use_container_width=True)
    
    if eval_clicked or "last_eval" in st.session_state:
        if eval_clicked:
            res = evaluate_patient(disease_option, current_inputs)
            st.session_state["last_eval"] = (disease_option, res, current_inputs)
        else:
            disease_option, res, current_inputs = st.session_state["last_eval"]
            bundle = load_disease_model(disease_option)

        risk_pct = res["risk_probability"] * 100
        conf_pct = res["confidence_score"] * 100

        if risk_pct >= 60.0:
            border_color = "#ff4d4f"
            bg_color = "rgba(255, 77, 79, 0.15)"
            status_text = T["status_high"]
            rec_text = T["tier3_rec"] if res["action_tier"].startswith("Tier 3") else T["tier1_rec"]
        elif risk_pct >= 35.0:
            border_color = "#faad14"
            bg_color = "rgba(250, 173, 20, 0.15)"
            status_text = T["status_mid"]
            rec_text = T["tier2_rec"]
        else:
            border_color = "#52c41a"
            bg_color = "rgba(82, 196, 26, 0.15)"
            status_text = T["status_low"]
            rec_text = T["tier1_rec"]

        st.markdown(f"### {T['results_header']}")
        
        st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 6px solid {border_color}; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: {border_color}; margin: 0 0 8px 0; font-size: 1.3rem;">{status_text}: {risk_pct:.1f}%</h3>
                <p style="margin: 0; color: var(--text-color); font-size: 15px;"><b>{T['metric_tier']}:</b> {res['action_tier'].split(':')[0]}</p>
                <p style="margin: 4px 0 0 0; color: var(--text-color); opacity: 0.9; font-size: 14px;"><b>मार्गदर्शन / Guidance:</b> {rec_text}</p>
            </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric(T["metric_prob"], f"{risk_pct:.1f}%")
        m2.metric(T["metric_conf"], f"{conf_pct:.1f}%")
        m3.metric(T["metric_tier"], res["action_tier"].split(":")[0])

        st.caption(f"{T['metric_conf']}:")
        st.progress(min(max(conf_pct / 100.0, 0.0), 1.0))

        report_text = f"""======================================================================
MEDISENSE AI CLINICAL DECISION SUPPORT - SUMMARY AUDIT
======================================================================
Evaluated Condition   : {disease_option}
Interface Language    : {selected_lang}
Assessed Risk         : {risk_pct:.1f}% ({status_text})
Prediction Confidence : {conf_pct:.1f}%
Assigned Safety Tier  : {res['action_tier']}
Clinical Guidance     : {rec_text}
----------------------------------------------------------------------
BIOMARKER PROFILE:
""" + "\n".join([f"  - {k}: {v}" for k, v in current_inputs.items()]) + """
----------------------------------------------------------------------
Disclaimer: Generated by an assistive clinical decision support system.
Correlate with formal laboratory diagnostic testing.
======================================================================
"""
        st.download_button(
            label=T["download_report_btn"],
            data=report_text,
            file_name=f"medisense_{disease_option.lower()}_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("---")
        if st.button(T["comorbidity_btn"], use_container_width=True):
            st.markdown(f"#### {T['comorbidity_title']}")
            c_cols = st.columns(5)
            all_diseases = ["Diabetes", "Kidney", "Liver", "Stroke", "Heart"]
            for i, d in enumerate(all_diseases):
                try:
                    c_res = evaluate_patient(d, current_inputs)
                    p = c_res["risk_probability"] * 100
                    delta_label = "Elevated" if p >= 50 else "Normal"
                    c_cols[i].metric(label=d, value=f"{p:.1f}%", delta=delta_label)
                except Exception:
                    c_cols[i].metric(label=d, value="N/A")

        st.markdown(f"### {T['xai_header']}")
        st.caption(T["xai_caption"])
        fig = generate_shap_plot(bundle["model"], res["df_input"], dark_mode=chart_dark_mode)
        st.pyplot(fig)
        st.info(T["xai_guide"])

        # What-If Interactive Simulator
        with st.expander(T["what_if_header"], expanded=False):
            st.caption(T["what_if_caption"])
            key_feats = [f for f in ["Glucose", "BloodPressure", "BMI", "age", "trestbps", "chol", "avg_glucose_level"] if f in current_inputs]
            if key_feats:
                sim_target = key_feats[0]
                cur_val = float(current_inputs[sim_target])
                min_val = max(10.0, float(cur_val * 0.5))
                max_val = float(cur_val * 1.5 + 20.0)
                sim_val = st.slider(f"Simulate adjustment for {sim_target}", min_val, max_val, cur_val)
                sim_inputs = current_inputs.copy()
                sim_inputs[sim_target] = sim_val
                sim_res = evaluate_patient(disease_option, sim_inputs)
                adj_risk = sim_res["risk_probability"] * 100
                st.write(f"Adjusted Predicted Risk: **{adj_risk:.1f}%** (Baseline: {risk_pct:.1f}%)")

        # -------------------------------------------------------------
        # 5. Automated Prescription Safety & Contraindication Alerts
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 5. Automated Prescription Safety & Contraindication Alerts")
        st.caption("Cross-references patient organ function indicators against pharmacological safety guidelines to prevent adverse drug events.")

        contraindications = get_drug_contraindications(current_inputs)

        if not contraindications:
            st.success("✅ No critical drug contraindications flagged. Biomarkers fall within standard therapeutic tolerance ranges.")
        else:
            for item in contraindications:
                is_crit = item["severity"] == "CRITICAL"
                box_bg = "rgba(255, 77, 79, 0.12)" if is_crit else "rgba(250, 173, 20, 0.12)"
                border_c = "#ff4d4f" if is_crit else "#faad14"
                tag_c = "#ff4d4f" if is_crit else "#d48806"

                st.markdown(f"""
                    <div style="background-color: {box_bg}; border-left: 5px solid {border_c}; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                        <span style="background-color: {border_c}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{item['severity']}</span>
                        <b style="color: {tag_c}; font-size: 15px; margin-left: 8px;">{item['drug_class']}</b>
                        <p style="margin: 6px 0 2px 0; color: var(--text-color); font-size: 13.5px;"><b>Mechanism:</b> {item['reason']}</p>
                        <p style="margin: 0; color: var(--text-color); opacity: 0.9; font-size: 13px;"><b>Clinical Action:</b> {item['guidance']}</p>
                    </div>
                """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 6. Hospital Ward Routing & Admission Referral Slip
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 6. Hospital Ward Routing & Specialist Referral Slip")
        st.caption("Intelligently directs the patient to specialized inpatient or outpatient units based on multi-organ risk.")

        routing_info = determine_ward_routing(current_inputs, disease_option, res["confidence_score"] * 100)

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.info(f"**Allocated Specialist Department:**\n### {routing_info['specialty']}")
            st.write(f"**Recommended Bed Facility:** {routing_info['bed_type']}")
        with r_col2:
            st.warning(f"**Triage Care Level:**\n### {routing_info['priority_level']}")
            st.write(f"**Stat Clinical Directives:** {routing_info['recommended_actions']}")

        patient_name_display = parsed_dict.get("patient_name", "Registered Inpatient")
        referral_doc = f"""================================================================================
           METROPOLIS HEALTHCARE SYSTEM - EMERGENCY REFERRAL SLIP
================================================================================
Patient Name        : {patient_name_display}
Evaluated Condition : {disease_option}
Assessed Risk Level : {risk_pct:.1f}% ({status_text})
Model Confidence    : {conf_pct:.1f}% ({res['action_tier']})
--------------------------------------------------------------------------------
WARD ALLOCATION & TRIAGE
--------------------------------------------------------------------------------
Allocated Specialty : {routing_info['specialty']}
Care Priority Level : {routing_info['priority_level']}
Recommended Bed Type: {routing_info['bed_type']}
Immediate Actions   : {routing_info['recommended_actions']}
--------------------------------------------------------------------------------
CONTRAINDICATED MEDICATIONS:
""" + ("\n".join([f" * [CRITICAL] {c['drug_class']} - {c['reason']}" for c in contraindications]) if contraindications else " * None flagged.") + f"""
--------------------------------------------------------------------------------
KEY CLINICAL BIOMARKERS:
""" + "\n".join([f"  - {k}: {v}" for k, v in current_inputs.items()]) + """
--------------------------------------------------------------------------------
Authorized by: MediSense AI Clinical Support Engine
Physician Signature: ___________________________   Date: 17-Oct-2026
================================================================================
"""

        st.download_button(
            label="📄 Generate Official Ward Admission & Referral Slip (.txt)",
            data=referral_doc,
            file_name=f"referral_slip_{patient_name_display.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
# -------------------------------------------------------------
# TAB 2: Batch Patient Triage & Ranking
# -------------------------------------------------------------
with tab_batch:
    st.markdown(f"### {T['tab_batch']}")
    st.caption("Upload unlimited patient files (PDFs, scans, text notes). The system extracts all clinical indicators, computes multidimensional urgency scores across all 5 disease models, and ranks patients by clinical intervention need.")

    batch_files = st.file_uploader(
        T["batch_upload_label"], 
        type=["pdf", "png", "jpg", "jpeg", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )

    if st.button(T["batch_run_btn"], type="primary", use_container_width=True):
        if not batch_files:
            st.warning("Please upload at least one patient document to run the triage queue.")
        else:
            with st.spinner(f"Ingesting {len(batch_files)} patient reports and evaluating across all 5 disease models..."):
                st.session_state["df_batch_ranked"] = process_batch_reports(
                    batch_files, 
                    target_condition=disease_option, 
                    language=selected_lang
                )

    # Render results if available in session state
    if "df_batch_ranked" in st.session_state and not st.session_state["df_batch_ranked"].empty:
        df_ranked = st.session_state["df_batch_ranked"]

        p1_count = sum(df_ranked["Triage Priority"].str.contains("Priority 1", na=False))
        p2_count = sum(df_ranked["Triage Priority"].str.contains("Priority 2", na=False))
        p3_count = sum(df_ranked["Triage Priority"].str.contains("Priority 3", na=False))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Patients Processed", len(df_ranked))
        m2.metric("🔴 Priority 1 (Immediate)", p1_count)
        m3.metric("🟡 Priority 2 (Urgent)", p2_count)
        m4.metric("🟢 Priority 3 (Routine)", p3_count)

        st.markdown("---")
        st.markdown(f"#### {T['batch_table_title']}")

        # Dynamic name column resolution
        name_col = "Patient Name" if "Patient Name" in df_ranked.columns else ("Patient Identifier" if "Patient Identifier" in df_ranked.columns else "File Name")

        # Full multi-disease risk breakdown columns
        desired_cols = [
            name_col, 
            "Urgency Score", 
            "Triage Priority", 
            "Primary Disease Risk",
            "Kidney Risk",
            "Heart Risk",
            "Stroke Risk",
            "Diabetes Risk",
            "Liver Risk",
            "Clinical Explanation"
        ]
        
        display_cols = [c for c in desired_cols if c in df_ranked.columns]

        # Display prioritized table
        st.dataframe(df_ranked[display_cols], use_container_width=True)

        csv_data = df_ranked[display_cols].to_csv().encode('utf-8')
        st.download_button(
            label=T["batch_download_btn"],
            data=csv_data,
            file_name=f"multidisease_triage_queue_{disease_option.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
# -------------------------------------------------------------
# TAB 3: Model Benchmarks
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown(f"### {T['benchmark_header']}")
    st.caption(T["benchmark_caption"])
    
    summary_data = [
        {"Condition / Cohort": "Diabetes (PIMA Indian)", "ROC-AUC": 0.949, "Accuracy": "88.3%", "Sensitivity (Recall)": "81.5%", "Precision": "84.6%"},
        {"Condition / Cohort": "Chronic Kidney Disease (CKD)", "ROC-AUC": 1.000, "Accuracy": "98.8%", "Sensitivity (Recall)": "98.0%", "Precision": "100.0%"},
        {"Condition / Cohort": "Liver Disease (ILPD)", "ROC-AUC": 0.802, "Accuracy": "72.6%", "Sensitivity (Recall)": "71.1%", "Precision": "88.1%"},
        {"Condition / Cohort": "Stroke Risk (Clinical Registry)", "ROC-AUC": 0.835, "Accuracy": "71.6%", "Sensitivity (Recall)": "82.0%", "Precision": "12.7%"},
        {"Condition / Cohort": "Heart Disease (Cleveland Cohort)", "ROC-AUC": 0.982, "Accuracy": "91.8%", "Sensitivity (Recall)": "95.5%", "Precision": "93.3%"}
    ]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)