import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Load full UCI Cleveland Heart Disease dataset (14 columns)
ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data" / "heart.csv")

# All features except target
feature_cols = [c for c in data.columns if c != 'target']
X = data[feature_cols]
y = data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42
)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"Heart model accuracy: {acc:.4f}")

# Save model and scaler
model_dir = ROOT / "model"
model_dir.mkdir(exist_ok=True)
with open(model_dir / "heart_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(model_dir / "heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"Heart model saved to model/heart_model.pkl")
print(f"Heart scaler saved to model/heart_scaler.pkl")
print(f"Features used ({len(feature_cols)}): {feature_cols}")

