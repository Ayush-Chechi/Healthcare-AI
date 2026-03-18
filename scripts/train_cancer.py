from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from pathlib import Path

data = load_breast_cancer()

# Use the 4 features for app compatibility
feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
X = pd.DataFrame(data.data, columns=data.feature_names)[feature_names]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"Cancer model accuracy: {acc:.4f}")

# Save model and scaler
ROOT = Path(__file__).resolve().parents[1]
model_dir = ROOT / "model"
model_dir.mkdir(exist_ok=True)
with open(model_dir / "cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(model_dir / "cancer_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Cancer model saved to model/cancer_model.pkl")
print("Cancer scaler saved to model/cancer_scaler.pkl")
