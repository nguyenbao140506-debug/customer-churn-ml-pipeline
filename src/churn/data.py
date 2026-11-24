"""Download and verify a fixed public IBM dataset; deterministic cleaning only."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/telco_churn.csv"
REVISION = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{REVISION}/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    **{name: ["No", "Yes", "No internet service"] for name in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}
FEATURES = NUMERIC + list(CATEGORIES)


def ensure_dataset(path=DATA_PATH):
    """Reuse verified bytes, or download them with normal TLS validation."""
    path = Path(path)
    content = path.read_bytes() if path.exists() else urlopen(SOURCE_URL, timeout=60).read()
    if hashlib.sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum mismatch; expected the pinned IBM CSV.")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def load_dataset(path=DATA_PATH):
    raw = pd.read_csv(ensure_dataset(path))
    required = set(FEATURES + ["customerID", "Churn"])
    if set(raw.columns) != required or raw.customerID.duplicated().any():
        raise ValueError("Unexpected dataset schema or duplicate customer IDs.")
    if raw.Churn.isna().any() or not set(raw.Churn.unique()) <= {"Yes", "No"}:
        raise ValueError("Target must contain only Yes and No.")
    return raw[FEATURES].copy(), raw.Churn.map({"No": 0, "Yes": 1}).astype(int), raw


def clean_features(frame):
    """Stateless conversion shared by training and the saved inference pipeline.

    No statistics are learned here. Imputation happens later, inside each fold.
    """
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing customer fields: {sorted(missing)}")
    result = frame[FEATURES].copy()
    for column in NUMERIC:
        values = result[column].astype("string").str.strip().replace("", pd.NA)
        result[column] = pd.to_numeric(values, errors="raise").astype(float)
        if np.isinf(result[column].to_numpy()).any():
            raise ValueError(f"{column} cannot be infinite.")
    for column in CATEGORIES:
        values = result[column].astype("string").str.strip().replace("", pd.NA)
        result[column] = values.astype(object).where(values.notna(), np.nan)
    return result


if __name__ == "__main__":
    print(f"Verified IBM Telco data: {ensure_dataset()}")
