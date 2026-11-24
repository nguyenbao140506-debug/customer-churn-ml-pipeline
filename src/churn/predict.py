"""Input validation and real prediction from the saved pipeline."""

import joblib
import numpy as np
import pandas as pd

from churn.data import CATEGORIES, FEATURES, NUMERIC, ROOT

DEFAULT_CUSTOMER = {
    "tenure": 12, "MonthlyCharges": 75.0, "TotalCharges": 900.0,
    **{column: values[0] for column, values in CATEGORIES.items()},
}
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]


def customer_frame(values):
    missing = set(FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing customer fields: {sorted(missing)}")
    record = {field: values[field] for field in FEATURES}
    for field in NUMERIC:
        record[field] = float(record[field])
        if not np.isfinite(record[field]) or record[field] < 0:
            raise ValueError(f"{field} must be a finite non-negative number.")
    if record["tenure"] > 72 or not record["tenure"].is_integer():
        raise ValueError("Tenure must be a whole number from 0 to 72 months.")
    for field, categories in CATEGORIES.items():
        record[field] = str(record[field])
        if record[field] not in categories:
            raise ValueError(f"Invalid category for {field}.")
    # Keep service-dependent fields consistent with the dataset.
    if record["PhoneService"] == "No":
        record["MultipleLines"] = "No phone service"
    elif record["MultipleLines"] == "No phone service":
        raise ValueError("Choose Yes or No for multiple lines when phone service is enabled.")
    if record["InternetService"] == "No":
        for field in ADDONS:
            record[field] = "No internet service"
    elif any(record[field] == "No internet service" for field in ADDONS):
        raise ValueError("Choose Yes or No for add-ons when internet service is enabled.")
    return pd.DataFrame([record], columns=FEATURES)


def load_model(path=ROOT / "models/churn_pipeline.joblib"):
    # Load only artifacts produced by this repository. Joblib files are executable.
    return joblib.load(path)


def predict_customer(model, values):
    frame = customer_frame(values)
    probability = float(model.predict_proba(frame)[0, 1])
    return {"churn": probability >= 0.5, "probability": probability}
