# 🩺 AI HealthScan – Disease Diagnosis & Prediction

**AI HealthScan** is a smart web application designed to assist users with early diagnosis and risk prediction of various diseases using trained machine learning models and a symptom-based Q&A system.

## 🔍 Features

### 🤖 AI Model-Based Predictions
- Predicts likelihood of:
  - Diabetes
  - Heart Disease
  - Breast Cancer
- Uses trained ML models (Random Forest, etc.)
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

## 🖥️ Tech Stack

| Frontend   | Backend     | AI/ML              |
|------------|-------------|--------------------|
| HTML, CSS, JS | Flask (Python) | scikit-learn, pandas |

---

## 📁 Folder Structure

📦 AI_HealthScan/
├── app.py
├── templates/
│ ├── index.html
│ ├── result.html
│ ├── symptom_question.html
│ └── symptom_result.html
├── static/
│ └── (optional JS/CSS files)
├── model/
│ ├── diabetes_model.pkl
│ ├── heart_model.pkl
│ └── cancer_model.pkl
├── train_*.py
├── requirements.txt
└── README.md


---

## 🚀 Running Locally

1. **Clone the repository**

Install dependencies
pip install -r requirements.txt

Run the app
python app.py

Visit in browser:
http://127.0.0.1:5000

📊 Datasets Used

1. PIMA Indians Diabetes Dataset
2. Heart Disease UCI
3. Breast Cancer Wisconsin

📄 License
This project is licensed under the MIT License.

📚 Useful Resources

1. WHO - Diabetes
2. American Heart Association
3. BreastCancer.org
4. CDC - Dengue Info
5. National Health Portal (India)

🙌 Acknowledgements

Created with ❤️ for academic and healthcare awareness purposes.
Designed and developed by Utkarsh Singh Parihar.