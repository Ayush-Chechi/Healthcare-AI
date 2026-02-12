import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load Models
models = {
    "diabetes": pickle.load(open("model/diabetes_model.pkl", "rb")),
    "heart": pickle.load(open("model/heart_model.pkl", "rb")),
    "cancer": pickle.load(open("model/cancer_model.pkl", "rb")),
}

# Input Fields Configuration
input_fields = {
    "diabetes": ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age'],
    "heart": ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
    "cancer": ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'],
}

cancer_field_mapping = {
    'radius_mean': 'mean radius',
    'texture_mean': 'mean texture',
    'perimeter_mean': 'mean perimeter',
    'area_mean': 'mean area'
}

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
st.set_page_config(page_title="Healthcare AI", page_icon="PwA", layout="wide")

st.title("Healthcare AI Prediction System")

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Service", ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Cancer Prediction", "Symptom Checker"])

if app_mode == "Home":
    st.write("Welcome to the Healthcare AI System. Please select a service from the sidebar.")

elif app_mode in ["Diabetes Prediction", "Heart Disease Prediction", "Cancer Prediction"]:
    disease_key = app_mode.split()[0].lower()
    
    st.header(f"{app_mode}")
    
    # Dynamic Input Form
    with st.form("prediction_form"):
        inputs = {}
        cols = st.columns(2)
        fields = input_fields[disease_key]
        
        for i, field in enumerate(fields):
            label = field.replace('_', ' ').title()
            # Special handling for cancer field names mapping if needed for display, 
            # but model expects specific keys. usage of cancer_field_mapping is for model input formatting if needed
            
            with cols[i % 2]:
                inputs[field] = st.number_input(label, value=0.0)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            model = models[disease_key]
            values = [inputs[field] for field in fields]
            
            # Create DataFrame for model
            df = pd.DataFrame([values], columns=fields)
            
            # Apply mapping for cancer model if necessary (based on app.py logic)
            if disease_key == "cancer":
                 df.columns = [cancer_field_mapping[col] for col in df.columns]

            try:
                prediction = model.predict(df)[0]
                confidence = model.predict_proba(df)[0][prediction]
                confidence_percent = round(confidence * 100, 2)
                
                result_text = "⚠️ You may have the disease." if prediction == 1 else "✅ You are likely healthy."
                st.subheader("Prediction Result")
                st.write(f"**Result:** {result_text}")
                st.write(f"**Confidence:** {confidence_percent}%")
                
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
        st.write(f"**Question {step + 1}:** {question}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes"):
                st.session_state.scores[disease] = st.session_state.scores.get(disease, 0) + 1
                st.session_state.step += 1
                st.rerun()
        with col2:
            if st.button("No"):
                st.session_state.step += 1
                st.rerun()
                
    else:
        # Final Result Calculation
        scores = st.session_state.scores
        if scores:
            most_likely = max(scores, key=scores.get)
            total_for_disease = sum(1 for q in SYMPTOM_QUESTIONS if q[1] == most_likely)
            confidence = round((scores[most_likely] / total_for_disease) * 100, 1)
            
            st.success(f"Based on your symptoms, you might have: **{most_likely.title()}**")
            st.write(f"Confidence: {confidence}%")
        else:
             st.info("No specific disease detected based on the answers provided.")
             
        if st.button("Restart Symptom Checker"):
            st.session_state.step = 0
            st.session_state.scores = {}
            st.rerun()
