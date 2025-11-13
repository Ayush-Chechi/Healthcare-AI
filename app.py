from flask import Flask, render_template, request, redirect, session
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'  # Needed for session storage

# ---------------------- ML-Based Model Setup ---------------------- #

models = {
    "diabetes": pickle.load(open("model/diabetes_model.pkl", "rb")),
    "heart": pickle.load(open("model/heart_model.pkl", "rb")),
    "cancer": pickle.load(open("model/cancer_model.pkl", "rb")),
}

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']
    model = models[disease]
    fields = input_fields[disease]
    values = [float(request.form[field]) for field in fields]

    df = pd.DataFrame([values], columns=fields)

    if disease == "cancer":
        df.columns = [cancer_field_mapping[col] for col in df.columns]

    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df)[0][prediction]
    confidence_percent = round(confidence * 100, 2)

    result = "⚠️ You may have the disease." if prediction == 1 else "✅ You are likely healthy."
    return render_template("result.html", prediction=result, confidence=confidence_percent, inputs=dict(zip(fields, values)))

# ---------------------- Symptom-Based Diagnosis ---------------------- #

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

@app.route('/symptom-checker')
def symptom_checker():
    session['step'] = 0
    session['scores'] = {}
    question = SYMPTOM_QUESTIONS[0][0]
    return render_template('symptom_question.html', question=question, progress=1, total=len(SYMPTOM_QUESTIONS))

@app.route('/symptom-answer', methods=['POST'])
def symptom_answer():
    answer = request.form['answer']
    step = session.get('step', 0)
    scores = session.get('scores', {})

    if answer == 'yes':
        disease = SYMPTOM_QUESTIONS[step][1]
        scores[disease] = scores.get(disease, 0) + 1

    step += 1
    session['step'] = step
    session['scores'] = scores

    if step >= len(SYMPTOM_QUESTIONS):
        if scores:
            most_likely = max(scores, key=scores.get)
            total_for_disease = sum(1 for q in SYMPTOM_QUESTIONS if q[1] == most_likely)
            confidence = round((scores[most_likely] / total_for_disease) * 100, 1)
            return render_template("symptom_result.html", result=most_likely.title(), confidence=confidence)
        else:
            return render_template("symptom_result.html", result="None", confidence=0)

    next_question = SYMPTOM_QUESTIONS[step][0]
    return render_template('symptom_question.html', question=next_question, progress=step+1, total=len(SYMPTOM_QUESTIONS))

@app.route('/reset-symptoms')
def reset_symptoms():
    session.pop('step', None)
    session.pop('scores', None)
    return redirect('/symptom-checker')


# ---------------------- Run ---------------------- #

if __name__ == '__main__':
    app.run(debug=True)
