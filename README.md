# 🩺 AI HealthScan – Disease Diagnosis & Prediction

**AI HealthScan** is a smart web application that supports early risk screening using trained ML models and a symptom-based Q&A flow.

## 🔍 Features

### 🤖 AI Model-Based Predictions
- Predicts likelihood of:
  - Diabetes
  - Heart Disease
  - Breast Cancer
- Uses trained ML models (Random Forest)
- Confidence score shown with a circular chart
- PDF report download & share via WhatsApp/Email

### 🧠 Symptom-Based Diagnosis
- One-by-one yes/no symptom questionnaire
- Diagnoses diseases like:
  - Dengue
  - UTI
  - Cold
  - Breast Cancer
  - More...
- Animated progress bar
- Confidence visualization
- Explanation of diagnosis + links to helpful resources

---

## 🖥️ Tech Stack (What + Where)

### Frontend (HTML/CSS/JS)
- **HTML/CSS**: Page layout and styling in `templates/index.html`, `templates/result.html`, `templates/symptom_question.html`, `templates/symptom_result.html`
- **JavaScript**:
  - Dynamic input form + disease info in `templates/index.html`
  - Confidence doughnut chart using `Chart.js` in `templates/result.html` and `templates/symptom_result.html`
  - PDF export via `html2pdf.js` and chart capture via `html2canvas` in `templates/result.html`
  - WhatsApp/Email share links in `templates/result.html`

### Backend (Python + Flask)
- **Flask**: Routing, form handling, and template rendering in `app.py`
- **Session storage**: Tracks symptom-question progress in `app.py`

### ML/Data
- **scikit-learn**: RandomForest models trained in `train_diabetes.py`, `train_heart.py`, `train_cancer.py`
- **pandas**: Data loading/cleanup in training scripts; input transformation for prediction in `app.py`
- **pickle**: Model serialization to `model/*.pkl` and loading in `app.py`

---

## 📁 File & Folder Guide (What Each File Does)

- `app.py` — Flask app with routes for prediction + symptom checker, loads ML models, renders templates
- `train_diabetes.py` — Trains diabetes model from `diabetes.csv` and saves `model/diabetes_model.pkl`
- `train_heart.py` — Trains heart disease model from `heart.csv` and saves `model/heart_model.pkl`
- `train_cancer.py` — Trains breast cancer model using `sklearn.datasets.load_breast_cancer` and saves `model/cancer_model.pkl`
- `requirements.txt` — Python dependencies for the app and ML training
- `diabetes.csv` — Dataset used to train diabetes model
- `heart.csv` — Dataset used to train heart disease model
- `model/` — Saved ML models (`*.pkl`) used by the Flask app
- `templates/` — HTML templates rendered by Flask
- `templates/index.html` — Disease selector + input form + disease info + links
- `templates/result.html` — Prediction results, confidence chart, PDF download, share links
- `templates/symptom_question.html` — Symptom questionnaire UI with progress bar
- `templates/symptom_result.html` — Symptom-based result page with chart and explanations
- `README.md` — Project overview and setup notes

---

## 🚀 Running Locally

1. **Clone the repository**
2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python app.py
```

4. **Open in browser**

```
http://127.0.0.1:5000
```

---

## 📊 Datasets Used

1. PIMA Indians Diabetes Dataset (`diabetes.csv`)
2. Heart Disease UCI Dataset (`heart.csv`)
3. Breast Cancer Wisconsin Dataset (via `sklearn.datasets.load_breast_cancer`)

---

## 📄 License

This project is licensed under the MIT License.

---

## 📚 Useful Resources

1. WHO - Diabetes
2. American Heart Association
3. BreastCancer.org
4. CDC - Dengue Info
5. National Health Portal (India)

---

## 🙌 Acknowledgements

Created with ❤️ for academic and healthcare awareness purposes.
Designed and developed by Utkarsh Singh Parihar.
