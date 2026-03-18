import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


@st.cache_resource
def load_bundle(disease_key: str):
    pipeline_path = MODEL_DIR / f"{disease_key}_pipeline.pkl"
    schema_path = MODEL_DIR / f"{disease_key}_schema.json"
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return pipeline, schema


def build_inputs(schema: dict) -> dict:
    features: dict = schema.get("features", {})
    inputs: dict[str, object] = {}
    cols = st.columns(2)
    items = list(features.items())

    for i, (name, meta) in enumerate(items):
        with cols[i % 2]:
            if meta.get("type") == "categorical":
                options = meta.get("values", [])
                if not options:
                    options = ["Unknown"]
                inputs[name] = st.selectbox(name, options=options, index=0)
            else:
                lo = float(meta.get("min", 0.0))
                hi = float(meta.get("max", lo + 1.0))
                default = float(meta.get("default", (lo + hi) / 2))
                step = (hi - lo) / 100 if hi > lo else 1.0
                inputs[name] = st.number_input(
                    name,
                    min_value=lo,
                    max_value=hi,
                    value=min(max(default, lo), hi),
                    step=step,
                )

    return inputs

# Symptom Checker Data
SYMPTOM_QUESTIONS = [
    ("Do you experience frequent urination?", "diabetes"),
    ("Do you feel excessive thirst?", "diabetes"),
    ("Do you feel chest pain?", "heart disease"),
    ("Do you experience shortness of breath?", "heart disease"),
    ("Do you have a persistent fever?", "dengue"),
    ("Do you have muscle pain and joint aches?", "dengue"),
    ("Do you have white or yellow vaginal discharge?", "uti"),
    ("Do you feel pain or burning while urinating?", "uti"),
    ("Do you have a cough with phlegm or dry cough?", "cold"),
    ("Do you have a sore throat?", "cold"),
    ("Do you feel a lump in the breast?", "breast cancer"),
    ("Have you noticed any discharge or changes in breast shape?", "breast cancer")
]

# Streamlit UI
st.set_page_config(page_title="Healthcare AI", page_icon="🏥", layout="wide")

st.title("🏥 Healthcare AI Prediction System")

# Sidebar Navigation
app_mode = st.sidebar.selectbox(
    "Choose Service",
    ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Cancer Prediction", "Symptom Checker"]
)

if app_mode == "Home":
    st.write("Welcome to the Healthcare AI System. Please select a service from the sidebar.")
    st.markdown("""
    ### Available Services
    - **Diabetes Prediction** — Predict diabetes risk based on 8 health metrics
    - **Heart Disease Prediction** — Predict heart disease using 13 clinical features (UCI Cleveland)
    - **Cancer Prediction** — Predict breast cancer using tumor measurements
    - **Symptom Checker** — Answer questions to identify potential conditions
    """)

elif app_mode in ["Diabetes Prediction", "Heart Disease Prediction", "Cancer Prediction"]:
    if app_mode == "Heart Disease Prediction":
        disease_key = "heart"
    elif app_mode == "Cancer Prediction":
        disease_key = "cancer"
    else:
        disease_key = "diabetes"

    st.header(app_mode)
    pipeline, schema = load_bundle(disease_key)

    with st.form(f"{disease_key}_prediction_form"):
        inputs = build_inputs(schema)
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        df = pd.DataFrame([inputs])
        try:
            prob = float(pipeline.predict_proba(df)[0][1])
            confidence_percent = round(prob * 100, 2)
            prediction = 1 if prob >= 0.5 else 0

            if prediction == 1:
                st.error(f"⚠️ **Higher Risk Detected** — Confidence: {confidence_percent}%")
                st.caption("This is a screening tool, not a diagnosis. Please consult a doctor.")
            else:
                st.success(f"✅ **Lower Risk** — Confidence: {confidence_percent}%")
                st.caption("Maintain a healthy lifestyle. Regular check-ups are recommended.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif app_mode == "Symptom Checker":
    st.header("Symptom Checker")

    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.scores = {}

    step = st.session_state.step

    if step < len(SYMPTOM_QUESTIONS):
        question, disease = SYMPTOM_QUESTIONS[step]
        st.progress((step + 1) / len(SYMPTOM_QUESTIONS), text=f"Question {step + 1} of {len(SYMPTOM_QUESTIONS)}")
        st.write(f"**{question}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes", use_container_width=True):
                st.session_state.scores[disease] = st.session_state.scores.get(disease, 0) + 1
                st.session_state.step += 1
                st.rerun()
        with col2:
            if st.button("❌ No", use_container_width=True):
                st.session_state.step += 1
                st.rerun()

    else:
        scores = st.session_state.scores
        if scores:
            most_likely = max(scores, key=scores.get)
            total_for_disease = sum(1 for q in SYMPTOM_QUESTIONS if q[1] == most_likely)
            confidence = round((scores[most_likely] / total_for_disease) * 100, 1)

            st.success(f"Based on your symptoms, you might have: **{most_likely.title()}**")
            st.write(f"Confidence: {confidence}%")
            st.caption("This is not a medical diagnosis. Please consult a healthcare professional.")
        else:
            st.info("No specific disease detected based on the answers provided.")

        if st.button("🔄 Restart Symptom Checker"):
            st.session_state.step = 0
            st.session_state.scores = {}
            st.rerun()
