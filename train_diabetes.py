import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle, os

data = pd.read_csv("diabetes.csv")

# Rename columns to match app.py input_fields
data.columns = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age', 'Outcome']

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Diabetes model saved to model/diabetes_model.pkl")
