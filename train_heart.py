import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# ✅ WORKING URL from UCI repository mirror
url = "heart.csv"
data = pd.read_csv(url)

# Select only the columns we use in prediction
X = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
y = data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
os.makedirs("model", exist_ok=True)
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Heart model saved to model/heart_model.pkl")
