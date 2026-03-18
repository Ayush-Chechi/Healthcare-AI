# 🏥 Healthcare AI Prediction System

**AI HealthScan** is a smart web application that supports early risk screening for **Diabetes**, **Heart Disease**, and **Breast Cancer** using advanced Machine Learning models (XGBoost & RandomForest). It also features a symptom-based checker for common conditions.

---

## 🚀 Key Features

### 🤖 AI Model-Based Predictions
- **Heart Disease**: Uses a tuned **RandomForest** pipeline trained on the combined UCI processed Heart Disease datasets (920 rows).
- **Diabetes**: Uses a tuned **RandomForest** pipeline trained on the UCI Early Stage Diabetes Risk dataset (520 rows).
- **Breast Cancer**: Uses a tuned **XGBoost** pipeline trained on **WDBC** (569 rows, 30 features).
- **Interactive UI**: Built with **Streamlit** for real-time risk assessment.
- **Auto-adapting inputs**: The app generates inputs from a saved feature schema (`model/*_schema.json`) so it stays compatible as datasets/models evolve.

### 🧠 Symptom-Based Check
- Simple Q&A flow to screen for:
  - Dengue
  - UTI
  - Cold/Flu
  - General symptoms

---

## 📊 Notebook Analysis & Feature Engineering
This project includes a comprehensive Jupyter Notebook (`Healthcare_AI_Analysis.ipynb`) that:
- **Documents dataset sources** (see `data/SOURCES.md`)
- **Shows EDA + sanity-check ROC curves**
- **Summarizes model performance** using Accuracy / Precision / Recall / F1 / AUC-ROC

---

## 🛠️ Tech Stack

- **App UI**: Streamlit (Python-based web UI)
- **ML Models**: XGBoost, Scikit-Learn (RandomForest)
- **Data Processing**: Pandas, NumPy
- **Analysis**: Jupyter Notebook, YData Profiling, Matplotlib/Seaborn

---

## 📂 Project Structure

- `app/` — Streamlit application code
- `streamlit_app.py` — Thin launcher (kept for the same run command)
- `scripts/` — Training + utilities (rebuild models, generate notebook)
- `Healthcare_AI_Analysis.ipynb` — Comprehensive EDA and model training notebook
- `model/` — Trained pipeline artifacts + schemas + metrics
- `data/` — Normalized datasets + dataset sources
- `frontend/` — Optional static landing page (not used by Streamlit)

**Why there is no `templates/` folder**
- In Flask, HTML lives in `templates/` and CSS/JS in `static/`.  
- This project’s **actual app is Streamlit**, so Flask’s `templates/static` structure is unnecessary and confusing.
- The previous HTML UI has been kept only as a **static prototype** in `frontend/`.

---

## 🏃‍♂️ Running Locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in Browser**
   - The app should automatically open at `http://localhost:8501`

---

## 📊 Datasets Used

See `data/SOURCES.md` for direct URLs and citations. Current datasets:

1. **Early Stage Diabetes Risk Prediction Dataset (UCI)** — `data/diabetes.csv` (520 rows, 16 features)
2. **Heart Disease (UCI processed: Cleveland + Hungary + Switzerland + VA)** — `data/heart.csv` (920 rows, 13 features)
3. **Breast Cancer Wisconsin (Diagnostic) (WDBC) (UCI)** — `data/breast_cancer_wdbc.csv` (569 rows, 30 features)

### Current model performance (holdout test split)
- **Diabetes**: Acc 0.971 · Precision 0.984 · Recall 0.969 · F1 0.976 · AUC 0.999
- **Heart**: Acc 0.826 · Precision 0.824 · Recall 0.873 · F1 0.848 · AUC 0.926
- **Cancer**: Acc 0.974 · Precision 1.000 · Recall 0.929 · F1 0.963 · AUC 0.994

These are saved verbatim in `model/*_metrics.json`.

---

## 📄 License
This project is licensed under the MIT License.
Designed for educational and screening purposes. **Not a substitute for professional medical advice.**
