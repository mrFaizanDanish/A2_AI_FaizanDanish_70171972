"""
preprocessing.py
----------------
The same cleaning and feature steps used in Assignment 1.
Both train_model.py and app.py import from here so the website
prepares a new customer exactly like the training data was prepared.
"""

import pandas as pd

# The eight service columns used to count num_services (same as Assignment 1)
SERVICES = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def clean_data(df):
    """Assignment 1 cleaning: fix TotalCharges, drop customerID."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    return df


def make_features(data):
    """
    Assignment 1 feature engineering:
      - "No internet service" / "No phone service" become "No"
      - num_services counts how many services are "Yes"
      - one-hot encoding with drop_first=True
    Used on the full dataset during training.
    """
    data = data.copy()
    data = data.replace({"No internet service": "No", "No phone service": "No"})
    data["num_services"] = (data[SERVICES] == "Yes").sum(axis=1)
    data = pd.get_dummies(data, drop_first=True, dtype=int)
    return data


def prepare_customer(customer, columns):
    """
    Turn ONE customer (a dict from the web form) into a row with the
    exact 24 columns, in the exact order, that the model was trained on.

    A single row cannot use drop_first (there is only one category present),
    so we one-hot encode without dropping and then reindex to the saved
    training columns. Any column that is missing becomes 0, which is the
    same value drop_first gives to the dropped category.
    """
    data = pd.DataFrame([customer])
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)
    data["tenure"] = data["tenure"].astype(int)
    data["MonthlyCharges"] = data["MonthlyCharges"].astype(float)
    data["TotalCharges"] = data["TotalCharges"].astype(float)
    data = data.replace({"No internet service": "No", "No phone service": "No"})
    data["num_services"] = (data[SERVICES] == "Yes").sum(axis=1)
    data = pd.get_dummies(data, dtype=int)
    data = data.reindex(columns=columns, fill_value=0)
    return data
