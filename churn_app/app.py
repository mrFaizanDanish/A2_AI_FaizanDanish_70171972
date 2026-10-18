import os

import joblib
from flask import Flask, render_template, request

from preprocessing import prepare_customer

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.joblib"))

OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

NUMBER_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 10000),
}


def read_form(form):
    """
    Read and check the submitted form.
    Returns (customer, error). If error is not None the input was bad
    and the model must not be called.
    """
    customer = {}

  
    for field, allowed in OPTIONS.items():
        value = form.get(field, "").strip()
        if value == "":
            return None, f"Please select a value for {field}."
        if value not in allowed:
            return None, f"Invalid value for {field}."
        customer[field] = value
    customer["SeniorCitizen"] = int(customer["SeniorCitizen"])

    for field, (low, high) in NUMBER_RANGES.items():
        value = form.get(field, "").strip()
        if value == "":
            return None, f"Please enter {field}."
        try:
            number = float(value)
        except ValueError:
            return None, f"{field} must be a number."
        if number < low or number > high:
            return None, f"{field} must be between {low} and {high}."
        customer[field] = number
    customer["tenure"] = int(customer["tenure"])

    return customer, None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        customer, error = read_form(request.form)

        if error is None:
  
            X = prepare_customer(customer, columns)
            X_scaled = scaler.transform(X)

            prediction = model.predict(X_scaled)[0]
            probability = round(model.predict_proba(X_scaled)[0][1] * 100, 2)
            result = "Churn" if prediction == 1 else "No Churn"

    return render_template("index.html", result=result, probability=probability, error=error)


if __name__ == "__main__":
    app.run(debug=True)
