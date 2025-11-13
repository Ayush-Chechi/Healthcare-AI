from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

data = load_breast_cancer()

# Use the correct feature names from the dataset
feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
X = pd.DataFrame(data.data, columns=data.feature_names)[feature_names]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
with open("model/cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Cancer model saved to model/cancer_model.pkl")
