import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data" / "diabetes.csv")

# Rename columns to match app input_fields
data.columns = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age', 'Outcome']

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='logloss', random_state=42
)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"Diabetes model accuracy: {acc:.4f}")

# Save model and scaler
model_dir = ROOT / "model"
model_dir.mkdir(exist_ok=True)
with open(model_dir / "diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(model_dir / "diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Diabetes model saved to model/diabetes_model.pkl")
print("Diabetes scaler saved to model/diabetes_scaler.pkl")
