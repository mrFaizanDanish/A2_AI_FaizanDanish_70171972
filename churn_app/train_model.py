"""
train_model.py
--------------
This script repeats the Assignment 1 pipeline and saves the final model.
It uses the same steps as the notebook:
  - clean TotalCharges and drop customerID
  - make_features() for encoding
  - StandardScaler
  - Random Forest (200 trees, max_depth=8, class_weight="balanced")

It saves three files in the model/ folder:
  model/model.joblib     -> the trained Random Forest
  model/scaler.joblib    -> the fitted StandardScaler
  model/columns.joblib   -> the exact list and order of the 24 feature columns

Run it once:  python train_model.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import clean_data, make_features

# 1. Load and clean (same as Assignment 1)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = clean_data(df)

# 2. Encode (same make_features as Assignment 1)
X = make_features(df.drop("Churn", axis=1))
y = df["Churn"].map({"Yes": 1, "No": 0})
print("X shape:", X.shape)

# 3. Split and scale (same random_state and stratify as Assignment 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Train the best model from Assignment 1 (Random Forest)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
)
rf.fit(X_train_s, y_train)

# 5. Check that the scores match Assignment 1
pred = rf.predict(X_test_s)
print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall   :", round(recall_score(y_test, pred), 4))
print("F1-Score :", round(f1_score(y_test, pred), 4))

# 6. Save everything the website needs
joblib.dump(rf, "model/model.joblib")
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(list(X.columns), "model/columns.joblib")
print("Saved: model/model.joblib, model/scaler.joblib, model/columns.joblib")
print("Feature columns:", list(X.columns))
