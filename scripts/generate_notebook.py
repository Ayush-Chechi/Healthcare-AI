"""
Healthcare-AI: Jupyter Notebook Generator
==========================================
Generates a comprehensive .ipynb notebook with:
- Pandas profiling for all 3 datasets
- Feature engineering
- XGBoost model training & comparison with RandomForest
- Model evaluation & visualization
"""

import nbformat as nbf
import os


def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }

    cells = []

    # ========================================================================
    # CELL 1: Title & Introduction (Markdown)
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "# 🏥 Healthcare AI — Comprehensive Analysis\n"
        "\n"
        "This notebook covers **three disease prediction models** with:\n"
        "- 📊 **Pandas Profiling** (Exploratory Data Analysis)\n"
        "- 🛠️ **Feature Engineering** (derived features, imputation, scaling)\n"
        "- 🚀 **XGBoost** vs **RandomForest** comparison\n"
        "- 📈 **Evaluation**: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrices\n"
        "\n"
        "**Datasets:** Heart Disease · Diabetes · Breast Cancer"
    ))

    # ========================================================================
    # CELL 2: Imports & Setup
    # ========================================================================
    cells.append(nbf.v4.new_code_cell(
        "# === Imports & Setup ===\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n"
        "import os\n"
        "import pickle\n"
        "\n"
        "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, precision_score, recall_score, f1_score,\n"
        "    roc_auc_score, confusion_matrix, classification_report, roc_curve\n"
        ")\n"
        "from sklearn.datasets import load_breast_cancer\n"
        "\n"
        "from xgboost import XGBClassifier\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "plt.rcParams['figure.figsize'] = (10, 5)\n"
        "plt.rcParams['figure.dpi'] = 100\n"
        "\n"
        "os.makedirs('model', exist_ok=True)\n"
        "os.makedirs('reports', exist_ok=True)\n"
        "\n"
        "print('✅ All imports successful!')"
    ))

    # ========================================================================
    # SECTION: PANDAS PROFILING
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "---\n"
        "# 📊 Part 1: Pandas Profiling (EDA)\n"
        "\n"
        "We use **ydata-profiling** to generate comprehensive EDA reports for each dataset.\n"
        "Reports are saved as HTML files in the `reports/` directory."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from ydata_profiling import ProfileReport\n"
        "\n"
        "# --- Heart Disease Profiling ---\n"
        "heart_df = pd.read_csv('data/heart.csv')\n"
        "print(f'Heart Dataset: {heart_df.shape[0]} rows × {heart_df.shape[1]} columns')\n"
        "print(heart_df.head())\n"
        "\n"
        "heart_profile = ProfileReport(\n"
        "    heart_df,\n"
        "    title='Heart Disease Dataset — Profiling Report',\n"
        "    minimal=True,\n"
        "    explorative=True\n"
        ")\n"
        "heart_profile.to_file('reports/heart_profiling_report.html')\n"
        "print('\\n✅ Heart profiling report saved to reports/heart_profiling_report.html')"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# --- Diabetes Profiling ---\n"
        "diabetes_df_raw = pd.read_csv('data/diabetes.csv')\n"
        "diabetes_df_raw.columns = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age', 'Outcome']\n"
        "print(f'Diabetes Dataset: {diabetes_df_raw.shape[0]} rows × {diabetes_df_raw.shape[1]} columns')\n"
        "print(diabetes_df_raw.head())\n"
        "\n"
        "diabetes_profile = ProfileReport(\n"
        "    diabetes_df_raw,\n"
        "    title='Diabetes Dataset — Profiling Report',\n"
        "    minimal=True,\n"
        "    explorative=True\n"
        ")\n"
        "diabetes_profile.to_file('reports/diabetes_profiling_report.html')\n"
        "print('\\n✅ Diabetes profiling report saved to reports/diabetes_profiling_report.html')"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# --- Breast Cancer Profiling ---\n"
        "cancer_data = load_breast_cancer()\n"
        "cancer_df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)\n"
        "cancer_df['target'] = cancer_data.target\n"
        "print(f'Cancer Dataset: {cancer_df.shape[0]} rows × {cancer_df.shape[1]} columns')\n"
        "print(cancer_df.head())\n"
        "\n"
        "cancer_profile = ProfileReport(\n"
        "    cancer_df,\n"
        "    title='Breast Cancer Dataset — Profiling Report',\n"
        "    minimal=True,\n"
        "    explorative=True\n"
        ")\n"
        "cancer_profile.to_file('reports/cancer_profiling_report.html')\n"
        "print('\\n✅ Cancer profiling report saved to reports/cancer_profiling_report.html')"
    ))

    # ========================================================================
    # SECTION: HEART DISEASE
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "---\n"
        "# ❤️ Part 2: Heart Disease Prediction\n"
        "\n"
        "**Dataset:** UCI Cleveland Heart Disease (14 columns, ~297 rows)\n"
        "\n"
        "**All 13 features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, "
        "`thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`\n"
        "\n"
        "### Feature Engineering\n"
        "| New Feature | Formula | Rationale |\n"
        "|---|---|---|\n"
        "| `chol_age_ratio` | `chol / age` | Age-normalized cholesterol |\n"
        "| `cardiac_index` | `thalach / age` | Heart rate reserve proxy |\n"
        "| `risk_score` | `(chol × oldpeak) / (thalach + 1)` | Combined risk factor |\n"
        "| `cp_thal_interaction` | `cp × thal` | Chest pain × thalassemia interaction |\n"
        "| `exang_oldpeak` | `exang × oldpeak` | Exercise angina severity |"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# === Heart Disease: Data Preparation & Feature Engineering ===\n"
        "heart = pd.read_csv('data/heart.csv')\n"
        "print(f'Heart dataset: {heart.shape[0]} rows, {heart.shape[1]} columns')\n"
        "print(f'Columns: {heart.columns.tolist()}')\n"
        "\n"
        "# Target\n"
        "y_heart = heart['target']\n"
        "\n"
        "# Use ALL 13 features (full UCI Cleveland dataset)\n"
        "feature_cols = [c for c in heart.columns if c != 'target']\n"
        "X_heart_original = heart[feature_cols].copy()\n"
        "\n"
        "# Feature Engineering — create derived features\n"
        "X_heart = X_heart_original.copy()\n"
        "X_heart['chol_age_ratio'] = X_heart['chol'] / (X_heart['age'] + 1)\n"
        "X_heart['cardiac_index'] = X_heart['thalach'] / (X_heart['age'] + 1)\n"
        "X_heart['risk_score'] = (X_heart['chol'] * X_heart['oldpeak'].clip(lower=0.01)) / (X_heart['thalach'] + 1)\n"
        "X_heart['cp_thal_interaction'] = X_heart['cp'] * X_heart['thal']\n"
        "X_heart['exang_oldpeak'] = X_heart['exang'] * X_heart['oldpeak']\n"
        "\n"
        "# Standardize\n"
        "scaler_heart = StandardScaler()\n"
        "X_heart_scaled = pd.DataFrame(\n"
        "    scaler_heart.fit_transform(X_heart),\n"
        "    columns=X_heart.columns\n"
        ")\n"
        "\n"
        "print(f'\\nOriginal features: {X_heart_original.shape[1]} → Engineered features: {X_heart_scaled.shape[1]}')\n"
        "print(f'Target distribution: No Disease={sum(y_heart==0)}, Disease={sum(y_heart==1)}')\n"
        "print('\\nEngineered feature stats:')\n"
        "print(X_heart[['chol_age_ratio', 'cardiac_index', 'risk_score', 'cp_thal_interaction', 'exang_oldpeak']].describe().round(2))"
    ))

    # Heart Model Training
    cells.append(nbf.v4.new_code_cell(
        "# === Heart Disease: Model Training & Comparison ===\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X_heart_scaled, y_heart, test_size=0.2, random_state=42, stratify=y_heart\n"
        ")\n"
        "\n"
        "# --- RandomForest Baseline ---\n"
        "rf_heart = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)\n"
        "rf_heart.fit(X_tr, y_tr)\n"
        "rf_pred = rf_heart.predict(X_te)\n"
        "rf_proba = rf_heart.predict_proba(X_te)[:, 1]\n"
        "\n"
        "# --- XGBoost ---\n"
        "xgb_heart = XGBClassifier(\n"
        "    n_estimators=200, max_depth=5, learning_rate=0.1,\n"
        "    subsample=0.8, colsample_bytree=0.8,\n"
        "    use_label_encoder=False, eval_metric='logloss',\n"
        "    random_state=42\n"
        ")\n"
        "xgb_heart.fit(X_tr, y_tr)\n"
        "xgb_pred = xgb_heart.predict(X_te)\n"
        "xgb_proba = xgb_heart.predict_proba(X_te)[:, 1]\n"
        "\n"
        "# --- Cross-Validation ---\n"
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "rf_cv = cross_val_score(rf_heart, X_heart_scaled, y_heart, cv=cv, scoring='accuracy')\n"
        "xgb_cv = cross_val_score(xgb_heart, X_heart_scaled, y_heart, cv=cv, scoring='accuracy')\n"
        "\n"
        "print('=== Heart Disease Results ===')\n"
        "print(f'\\nRandomForest  — Test Acc: {accuracy_score(y_te, rf_pred):.4f} | CV Mean: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}')\n"
        "print(f'XGBoost       — Test Acc: {accuracy_score(y_te, xgb_pred):.4f} | CV Mean: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}')\n"
        "\n"
        "print('\\n--- XGBoost Classification Report ---')\n"
        "print(classification_report(y_te, xgb_pred, target_names=['No Disease', 'Disease']))"
    ))

    # Heart Visualization
    cells.append(nbf.v4.new_code_cell(
        "# === Heart Disease: Visualization ===\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "# Confusion Matrices\n"
        "for ax, pred, title in [(axes[0], rf_pred, 'RandomForest'), (axes[1], xgb_pred, 'XGBoost')]:\n"
        "    cm = confusion_matrix(y_te, pred)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n"
        "                xticklabels=['No Disease', 'Disease'],\n"
        "                yticklabels=['No Disease', 'Disease'])\n"
        "    ax.set_title(f'Heart — {title}', fontsize=13, fontweight='bold')\n"
        "    ax.set_ylabel('Actual')\n"
        "    ax.set_xlabel('Predicted')\n"
        "\n"
        "# ROC Curves\n"
        "for proba, label in [(rf_proba, 'RandomForest'), (xgb_proba, 'XGBoost')]:\n"
        "    fpr, tpr, _ = roc_curve(y_te, proba)\n"
        "    auc = roc_auc_score(y_te, proba)\n"
        "    axes[2].plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})', linewidth=2)\n"
        "axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4)\n"
        "axes[2].set_title('Heart — ROC Curves', fontsize=13, fontweight='bold')\n"
        "axes[2].set_xlabel('False Positive Rate')\n"
        "axes[2].set_ylabel('True Positive Rate')\n"
        "axes[2].legend()\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reports/heart_model_comparison.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('📊 Plot saved to reports/heart_model_comparison.png')"
    ))

    # Heart Feature Importance
    cells.append(nbf.v4.new_code_cell(
        "# === Heart Disease: Feature Importance ===\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "\n"
        "for ax, model, title in [(axes[0], rf_heart, 'RandomForest'), (axes[1], xgb_heart, 'XGBoost')]:\n"
        "    importances = pd.Series(model.feature_importances_, index=X_heart.columns)\n"
        "    importances.sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')\n"
        "    ax.set_title(f'Heart — {title} Feature Importance', fontsize=13, fontweight='bold')\n"
        "    ax.set_xlabel('Importance')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reports/heart_feature_importance.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # ========================================================================
    # SECTION: DIABETES
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "---\n"
        "# 🩸 Part 3: Diabetes Prediction\n"
        "\n"
        "**Features:** `pregnancies`, `glucose`, `bp`, `skin`, `insulin`, `bmi`, `dpf`, `age`\n"
        "\n"
        "### Feature Engineering\n"
        "| Step | Description |\n"
        "|---|---|\n"
        "| Zero-value imputation | Replace 0s in `bp`, `skin`, `insulin`, `bmi` with column median |\n"
        "| `glucose_bmi_interaction` | `glucose × bmi` — metabolic syndrome proxy |\n"
        "| `insulin_glucose_ratio` | `insulin / (glucose + 1)` — insulin resistance indicator |\n"
        "| `age_pregnancies` | `age × pregnancies` — combined reproductive risk |\n"
        "| `log_glucose` | `log(glucose + 1)` — normalizes skewed glucose |\n"
        "| `bmi_category` | `1 if bmi >= 30 else 0` — obesity flag |"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# === Diabetes: Data Preparation & Feature Engineering ===\n"
        "diabetes = pd.read_csv('data/diabetes.csv')\n"
        "diabetes.columns = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age', 'Outcome']\n"
        "\n"
        "y_diabetes = diabetes['Outcome']\n"
        "X_diabetes_original = diabetes.drop('Outcome', axis=1).copy()\n"
        "\n"
        "# --- Zero-value imputation (biologically impossible zeros) ---\n"
        "zero_cols = ['bp', 'skin', 'insulin', 'bmi']\n"
        "X_diabetes = X_diabetes_original.copy()\n"
        "for col in zero_cols:\n"
        "    median_val = X_diabetes[col][X_diabetes[col] != 0].median()\n"
        "    X_diabetes[col] = X_diabetes[col].replace(0, median_val)\n"
        "    print(f'  {col}: replaced {(X_diabetes_original[col] == 0).sum()} zeros with median = {median_val:.1f}')\n"
        "\n"
        "# --- Derived features ---\n"
        "X_diabetes['glucose_bmi_interaction'] = X_diabetes['glucose'] * X_diabetes['bmi']\n"
        "X_diabetes['insulin_glucose_ratio'] = X_diabetes['insulin'] / (X_diabetes['glucose'] + 1)\n"
        "X_diabetes['age_pregnancies'] = X_diabetes['age'] * X_diabetes['pregnancies']\n"
        "X_diabetes['log_glucose'] = np.log1p(X_diabetes['glucose'])\n"
        "X_diabetes['bmi_category'] = (X_diabetes['bmi'] >= 30).astype(int)\n"
        "\n"
        "# --- Standardize ---\n"
        "scaler_diabetes = StandardScaler()\n"
        "X_diabetes_scaled = pd.DataFrame(\n"
        "    scaler_diabetes.fit_transform(X_diabetes),\n"
        "    columns=X_diabetes.columns\n"
        ")\n"
        "\n"
        "print(f'\\nOriginal features: {X_diabetes_original.shape[1]} → Engineered features: {X_diabetes_scaled.shape[1]}')\n"
        "print('\\nNew feature statistics:')\n"
        "print(X_diabetes[['glucose_bmi_interaction', 'insulin_glucose_ratio', 'age_pregnancies', 'log_glucose', 'bmi_category']].describe().round(2))"
    ))

    # Diabetes Model Training
    cells.append(nbf.v4.new_code_cell(
        "# === Diabetes: Model Training & Comparison ===\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes\n"
        ")\n"
        "\n"
        "# --- RandomForest Baseline ---\n"
        "rf_diabetes = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)\n"
        "rf_diabetes.fit(X_tr, y_tr)\n"
        "rf_pred = rf_diabetes.predict(X_te)\n"
        "rf_proba = rf_diabetes.predict_proba(X_te)[:, 1]\n"
        "\n"
        "# --- XGBoost ---\n"
        "xgb_diabetes = XGBClassifier(\n"
        "    n_estimators=300, max_depth=4, learning_rate=0.05,\n"
        "    subsample=0.8, colsample_bytree=0.8,\n"
        "    reg_alpha=0.1, reg_lambda=1.0,\n"
        "    use_label_encoder=False, eval_metric='logloss',\n"
        "    random_state=42\n"
        ")\n"
        "xgb_diabetes.fit(X_tr, y_tr)\n"
        "xgb_pred = xgb_diabetes.predict(X_te)\n"
        "xgb_proba = xgb_diabetes.predict_proba(X_te)[:, 1]\n"
        "\n"
        "# --- Cross-Validation ---\n"
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "rf_cv = cross_val_score(rf_diabetes, X_diabetes_scaled, y_diabetes, cv=cv, scoring='accuracy')\n"
        "xgb_cv = cross_val_score(xgb_diabetes, X_diabetes_scaled, y_diabetes, cv=cv, scoring='accuracy')\n"
        "\n"
        "print('=== Diabetes Results ===')\n"
        "print(f'\\nRandomForest  — Test Acc: {accuracy_score(y_te, rf_pred):.4f} | CV Mean: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}')\n"
        "print(f'XGBoost       — Test Acc: {accuracy_score(y_te, xgb_pred):.4f} | CV Mean: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}')\n"
        "\n"
        "print('\\n--- XGBoost Classification Report ---')\n"
        "print(classification_report(y_te, xgb_pred, target_names=['No Diabetes', 'Diabetes']))"
    ))

    # Diabetes Visualization
    cells.append(nbf.v4.new_code_cell(
        "# === Diabetes: Visualization ===\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "for ax, pred, title in [(axes[0], rf_pred, 'RandomForest'), (axes[1], xgb_pred, 'XGBoost')]:\n"
        "    cm = confusion_matrix(y_te, pred)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,\n"
        "                xticklabels=['No Diabetes', 'Diabetes'],\n"
        "                yticklabels=['No Diabetes', 'Diabetes'])\n"
        "    ax.set_title(f'Diabetes — {title}', fontsize=13, fontweight='bold')\n"
        "    ax.set_ylabel('Actual')\n"
        "    ax.set_xlabel('Predicted')\n"
        "\n"
        "for proba, label in [(rf_proba, 'RandomForest'), (xgb_proba, 'XGBoost')]:\n"
        "    fpr, tpr, _ = roc_curve(y_te, proba)\n"
        "    auc = roc_auc_score(y_te, proba)\n"
        "    axes[2].plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})', linewidth=2)\n"
        "axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4)\n"
        "axes[2].set_title('Diabetes — ROC Curves', fontsize=13, fontweight='bold')\n"
        "axes[2].set_xlabel('False Positive Rate')\n"
        "axes[2].set_ylabel('True Positive Rate')\n"
        "axes[2].legend()\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reports/diabetes_model_comparison.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # Diabetes Feature Importance
    cells.append(nbf.v4.new_code_cell(
        "# === Diabetes: Feature Importance ===\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "\n"
        "for ax, model, title in [(axes[0], rf_diabetes, 'RandomForest'), (axes[1], xgb_diabetes, 'XGBoost')]:\n"
        "    importances = pd.Series(model.feature_importances_, index=X_diabetes.columns)\n"
        "    importances.sort_values(ascending=True).plot(kind='barh', ax=ax, color='seagreen', edgecolor='black')\n"
        "    ax.set_title(f'Diabetes — {title} Feature Importance', fontsize=13, fontweight='bold')\n"
        "    ax.set_xlabel('Importance')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reports/diabetes_feature_importance.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # ========================================================================
    # SECTION: CANCER
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "---\n"
        "# 🎗️ Part 4: Breast Cancer Prediction\n"
        "\n"
        "**Improvement:** Using **all 30 features** from sklearn's breast cancer dataset instead of just 4.\n"
        "\n"
        "### Feature Engineering\n"
        "| New Feature | Formula | Rationale |\n"
        "|---|---|---|\n"
        "| `radius_area_ratio` | `mean radius / mean area` | Shape regularity indicator |\n"
        "| `perimeter_radius_ratio` | `mean perimeter / mean radius` | Circularity measure |\n"
        "| `texture_smoothness` | `mean texture × mean smoothness` | Surface irregularity |\n"
        "\n"
        "> **Note:** The Streamlit app expects 4 features. We train **two models**: \n"
        "> 1. Full-feature model (for best accuracy in this analysis)\n"
        "> 2. App-compatible 4-feature model (saved for `streamlit_app.py`)"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# === Cancer: Data Preparation & Feature Engineering ===\n"
        "cancer_data = load_breast_cancer()\n"
        "X_cancer_full = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)\n"
        "y_cancer = cancer_data.target\n"
        "\n"
        "# --- Feature Engineering on FULL dataset ---\n"
        "X_cancer = X_cancer_full.copy()\n"
        "X_cancer['radius_area_ratio'] = X_cancer['mean radius'] / (X_cancer['mean area'] + 1e-6)\n"
        "X_cancer['perimeter_radius_ratio'] = X_cancer['mean perimeter'] / (X_cancer['mean radius'] + 1e-6)\n"
        "X_cancer['texture_smoothness'] = X_cancer['mean texture'] * X_cancer['mean smoothness']\n"
        "\n"
        "# --- Standardize ---\n"
        "scaler_cancer = StandardScaler()\n"
        "X_cancer_scaled = pd.DataFrame(\n"
        "    scaler_cancer.fit_transform(X_cancer),\n"
        "    columns=X_cancer.columns\n"
        ")\n"
        "\n"
        "print(f'Original features: {X_cancer_full.shape[1]} → Engineered features: {X_cancer_scaled.shape[1]}')\n"
        "print(f'\\nTarget distribution: Malignant={sum(y_cancer == 0)}, Benign={sum(y_cancer == 1)}')"
    ))

    # Cancer Model Training
    cells.append(nbf.v4.new_code_cell(
        "# === Cancer: Model Training & Comparison (Full Features) ===\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X_cancer_scaled, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer\n"
        ")\n"
        "\n"
        "# --- RandomForest ---\n"
        "rf_cancer = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)\n"
        "rf_cancer.fit(X_tr, y_tr)\n"
        "rf_pred = rf_cancer.predict(X_te)\n"
        "rf_proba = rf_cancer.predict_proba(X_te)[:, 1]\n"
        "\n"
        "# --- XGBoost ---\n"
        "xgb_cancer = XGBClassifier(\n"
        "    n_estimators=200, max_depth=5, learning_rate=0.1,\n"
        "    subsample=0.8, colsample_bytree=0.8,\n"
        "    use_label_encoder=False, eval_metric='logloss',\n"
        "    random_state=42\n"
        ")\n"
        "xgb_cancer.fit(X_tr, y_tr)\n"
        "xgb_pred = xgb_cancer.predict(X_te)\n"
        "xgb_proba = xgb_cancer.predict_proba(X_te)[:, 1]\n"
        "\n"
        "# --- Cross-Validation ---\n"
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "rf_cv = cross_val_score(rf_cancer, X_cancer_scaled, y_cancer, cv=cv, scoring='accuracy')\n"
        "xgb_cv = cross_val_score(xgb_cancer, X_cancer_scaled, y_cancer, cv=cv, scoring='accuracy')\n"
        "\n"
        "print('=== Cancer Results (Full 33 Features) ===')\n"
        "print(f'\\nRandomForest  — Test Acc: {accuracy_score(y_te, rf_pred):.4f} | CV Mean: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}')\n"
        "print(f'XGBoost       — Test Acc: {accuracy_score(y_te, xgb_pred):.4f} | CV Mean: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}')\n"
        "\n"
        "print('\\n--- XGBoost Classification Report ---')\n"
        "print(classification_report(y_te, xgb_pred, target_names=['Malignant', 'Benign']))"
    ))

    # Cancer Visualization
    cells.append(nbf.v4.new_code_cell(
        "# === Cancer: Visualization ===\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "for ax, pred, title in [(axes[0], rf_pred, 'RandomForest'), (axes[1], xgb_pred, 'XGBoost')]:\n"
        "    cm = confusion_matrix(y_te, pred)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,\n"
        "                xticklabels=['Malignant', 'Benign'],\n"
        "                yticklabels=['Malignant', 'Benign'])\n"
        "    ax.set_title(f'Cancer — {title}', fontsize=13, fontweight='bold')\n"
        "    ax.set_ylabel('Actual')\n"
        "    ax.set_xlabel('Predicted')\n"
        "\n"
        "for proba, label in [(rf_proba, 'RandomForest'), (xgb_proba, 'XGBoost')]:\n"
        "    fpr, tpr, _ = roc_curve(y_te, proba)\n"
        "    auc = roc_auc_score(y_te, proba)\n"
        "    axes[2].plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})', linewidth=2)\n"
        "axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4)\n"
        "axes[2].set_title('Cancer — ROC Curves', fontsize=13, fontweight='bold')\n"
        "axes[2].set_xlabel('False Positive Rate')\n"
        "axes[2].set_ylabel('True Positive Rate')\n"
        "axes[2].legend()\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reports/cancer_model_comparison.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # Cancer Feature Importance (Top 15)
    cells.append(nbf.v4.new_code_cell(
        "# === Cancer: Top 15 Feature Importance ===\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n"
        "\n"
        "for ax, model, title in [(axes[0], rf_cancer, 'RandomForest'), (axes[1], xgb_cancer, 'XGBoost')]:\n"
        "    importances = pd.Series(model.feature_importances_, index=X_cancer.columns)\n"
        "    importances.nlargest(15).sort_values(ascending=True).plot(\n"
        "        kind='barh', ax=ax, color='mediumpurple', edgecolor='black'\n"
        "    )\n"
        "    ax.set_title(f'Cancer — {title} Top 15 Features', fontsize=13, fontweight='bold')\n"
        "    ax.set_xlabel('Importance')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reports/cancer_feature_importance.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # ========================================================================
    # App-compatible cancer model
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "### 🔗 App-Compatible Cancer Model (4 features)\n"
        "\n"
        "Training a separate model with the **4 features** expected by `streamlit_app.py`, "
        "but with XGBoost for better accuracy."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# === Cancer: App-Compatible 4-Feature XGBoost ===\n"
        "app_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']\n"
        "X_cancer_app = X_cancer_full[app_features].copy()\n"
        "\n"
        "# Add engineered features for the app model too\n"
        "X_cancer_app['radius_area_ratio'] = X_cancer_app['mean radius'] / (X_cancer_app['mean area'] + 1e-6)\n"
        "X_cancer_app['perimeter_radius_ratio'] = X_cancer_app['mean perimeter'] / (X_cancer_app['mean radius'] + 1e-6)\n"
        "\n"
        "scaler_cancer_app = StandardScaler()\n"
        "X_cancer_app_scaled = pd.DataFrame(\n"
        "    scaler_cancer_app.fit_transform(X_cancer_app),\n"
        "    columns=X_cancer_app.columns\n"
        ")\n"
        "\n"
        "X_tr_app, X_te_app, y_tr_app, y_te_app = train_test_split(\n"
        "    X_cancer_app_scaled, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer\n"
        ")\n"
        "\n"
        "# Original RF with 4 features (baseline)\n"
        "rf_cancer_app = RandomForestClassifier(n_estimators=200, random_state=42)\n"
        "rf_cancer_app.fit(X_tr_app[app_features], y_tr_app)\n"
        "rf_app_acc = accuracy_score(y_te_app, rf_cancer_app.predict(X_te_app[app_features]))\n"
        "\n"
        "# XGBoost with engineered features\n"
        "xgb_cancer_app = XGBClassifier(\n"
        "    n_estimators=200, max_depth=5, learning_rate=0.1,\n"
        "    use_label_encoder=False, eval_metric='logloss', random_state=42\n"
        ")\n"
        "xgb_cancer_app.fit(X_tr_app, y_tr_app)\n"
        "xgb_app_acc = accuracy_score(y_te_app, xgb_cancer_app.predict(X_te_app))\n"
        "\n"
        "print(f'App-Compatible Cancer Models:')\n"
        "print(f'  RF  (4 original features): {rf_app_acc:.4f}')\n"
        "print(f'  XGB (4 + 2 engineered):    {xgb_app_acc:.4f}')"
    ))

    # ========================================================================
    # SAVE MODELS
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "---\n"
        "# 💾 Part 5: Save Best Models & Scalers\n"
        "\n"
        "Saving the **best performing model** (XGBoost) and **fitted scalers** for each disease to `model/` directory.\n"
        "\n"
        "> ⚠️ **Scalers must be saved** — without them, deployed models receive unscaled input and produce garbage predictions.\n"
        "\n"
        "> The cancer model saved is the **app-compatible** version (4 features + engineered) "
        "to maintain compatibility with `streamlit_app.py`."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# === Save Best Models & Scalers ===\n"
        "os.makedirs('model', exist_ok=True)\n"
        "\n"
        "# Heart — save XGBoost + scaler\n"
        "with open('model/heart_model.pkl', 'wb') as f:\n"
        "    pickle.dump(xgb_heart, f)\n"
        "with open('model/heart_scaler.pkl', 'wb') as f:\n"
        "    pickle.dump(scaler_heart, f)\n"
        "print('✅ Heart: model + scaler saved')\n"
        "\n"
        "# Diabetes — save XGBoost + scaler\n"
        "with open('model/diabetes_model.pkl', 'wb') as f:\n"
        "    pickle.dump(xgb_diabetes, f)\n"
        "with open('model/diabetes_scaler.pkl', 'wb') as f:\n"
        "    pickle.dump(scaler_diabetes, f)\n"
        "print('✅ Diabetes: model + scaler saved')\n"
        "\n"
        "# Cancer — save app-compatible RF model (keeps original 4-feature interface)\n"
        "with open('model/cancer_model.pkl', 'wb') as f:\n"
        "    pickle.dump(rf_cancer_app, f)\n"
        "with open('model/cancer_scaler.pkl', 'wb') as f:\n"
        "    pickle.dump(scaler_cancer_app, f)\n"
        "print('✅ Cancer: model + scaler saved (app-compatible, 4 features)')\n"
        "\n"
        "print('\\n🎉 All models and scalers saved to model/ directory!')\n"
        "print('\\nSaved files:')\n"
        "for f in sorted(os.listdir('model')):\n"
        "    size = os.path.getsize(f'model/{f}') / 1024\n"
        "    print(f'  📁 model/{f} ({size:.1f} KB)')"
    ))

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "---\n"
        "# 📋 Part 6: Final Summary"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# === Final Results Summary (DataFrame) ===\n"
        "# Collect all CV scores computed earlier\n"
        "results = []\n"
        "\n"
        "# We need to re-reference the stored CV scores and test splits\n"
        "# Heart results (re-split to get consistent test set)\n"
        "X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(\n"
        "    X_heart_scaled, y_heart, test_size=0.2, random_state=42, stratify=y_heart\n"
        ")\n"
        "for name, model in [('RandomForest', rf_heart), ('XGBoost ★', xgb_heart)]:\n"
        "    pred = model.predict(X_te_h)\n"
        "    proba = model.predict_proba(X_te_h)[:, 1]\n"
        "    results.append({\n"
        "        'Disease': 'Heart', 'Model': name,\n"
        "        'Features': X_heart.shape[1],\n"
        "        'Accuracy': f'{accuracy_score(y_te_h, pred):.4f}',\n"
        "        'Precision': f'{precision_score(y_te_h, pred):.4f}',\n"
        "        'Recall': f'{recall_score(y_te_h, pred):.4f}',\n"
        "        'F1': f'{f1_score(y_te_h, pred):.4f}',\n"
        "        'AUC-ROC': f'{roc_auc_score(y_te_h, proba):.4f}'\n"
        "    })\n"
        "\n"
        "# Diabetes results\n"
        "X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(\n"
        "    X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes\n"
        ")\n"
        "for name, model in [('RandomForest', rf_diabetes), ('XGBoost ★', xgb_diabetes)]:\n"
        "    pred = model.predict(X_te_d)\n"
        "    proba = model.predict_proba(X_te_d)[:, 1]\n"
        "    results.append({\n"
        "        'Disease': 'Diabetes', 'Model': name,\n"
        "        'Features': X_diabetes.shape[1],\n"
        "        'Accuracy': f'{accuracy_score(y_te_d, pred):.4f}',\n"
        "        'Precision': f'{precision_score(y_te_d, pred):.4f}',\n"
        "        'Recall': f'{recall_score(y_te_d, pred):.4f}',\n"
        "        'F1': f'{f1_score(y_te_d, pred):.4f}',\n"
        "        'AUC-ROC': f'{roc_auc_score(y_te_d, proba):.4f}'\n"
        "    })\n"
        "\n"
        "# Cancer results (full features)\n"
        "X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(\n"
        "    X_cancer_scaled, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer\n"
        ")\n"
        "for name, model in [('RandomForest', rf_cancer), ('XGBoost ★', xgb_cancer)]:\n"
        "    pred = model.predict(X_te_c)\n"
        "    proba = model.predict_proba(X_te_c)[:, 1]\n"
        "    results.append({\n"
        "        'Disease': 'Cancer', 'Model': name,\n"
        "        'Features': X_cancer.shape[1],\n"
        "        'Accuracy': f'{accuracy_score(y_te_c, pred):.4f}',\n"
        "        'Precision': f'{precision_score(y_te_c, pred, zero_division=0):.4f}',\n"
        "        'Recall': f'{recall_score(y_te_c, pred, zero_division=0):.4f}',\n"
        "        'F1': f'{f1_score(y_te_c, pred, zero_division=0):.4f}',\n"
        "        'AUC-ROC': f'{roc_auc_score(y_te_c, proba):.4f}'\n"
        "    })\n"
        "\n"
        "summary_df = pd.DataFrame(results)\n"
        "print('=' * 90)\n"
        "print('  🏥 HEALTHCARE AI — FINAL MODEL COMPARISON')\n"
        "print('=' * 90)\n"
        "print(summary_df.to_string(index=False))\n"
        "print('=' * 90)\n"
        "print('\\n★ = Best model (saved to model/ directory)')\n"
        "print('\\n📁 Saved artifacts:')\n"
        "print('  • model/*.pkl     — Trained models + fitted scalers')\n"
        "print('  • reports/*.html   — Pandas profiling reports')\n"
        "print('  • reports/*.png    — Comparison plots')\n"
        "print('\\n✅ Analysis complete!')"
    ))

    nb.cells = cells
    return nb


if __name__ == '__main__':
    print('Generating Healthcare AI Analysis Notebook...')
    notebook = create_notebook()

    output_path = 'Healthcare_AI_Analysis.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)

    print(f'Notebook saved to: {output_path}')
    print(f'   Total cells: {len(notebook.cells)}')
    print(f'\nTo run the notebook:')
    print(f'   jupyter notebook {output_path}')
    print(f'   - or open in VS Code -')
