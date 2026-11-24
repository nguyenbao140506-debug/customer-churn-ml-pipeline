"""Start with: python -m streamlit run app.py"""

import json
import streamlit as st

from churn.data import CATEGORIES, ROOT
from churn.predict import DEFAULT_CUSTOMER, load_model, predict_customer

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.caption("ML PORTFOLIO  /  IBM TELCO CUSTOMER CHURN")
st.title("Customer Churn Predictor")
st.write("Enter a customer profile to estimate the likelihood of leaving the service.")


@st.cache_resource
def get_model():
    return load_model()


if not (ROOT / "models/churn_pipeline.joblib").exists():
    st.error("Model not found. Run `python -m churn.train` from the repository first.")
    st.stop()

model = get_model()
metadata = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
st.caption(f"Model: {metadata['selected_model']} · Decision threshold: 50% · Educational demonstration")

labels = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
    "Dependents": "Dependents", "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
}

with st.form("customer"):
    st.subheader("Customer profile")
    left, middle, right = st.columns(3)
    with left:
        tenure = st.number_input("Tenure (months)", 0, 72, int(DEFAULT_CUSTOMER["tenure"]), key="tenure")
    with middle:
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 75.0, step=0.5, key="monthly")
    with right:
        total = st.number_input("Total charges ($)", 0.0, 20000.0, 900.0, step=10.0, key="total")
    values = {"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total}
    columns = st.columns(3)
    for index, (field, choices) in enumerate(CATEGORIES.items()):
        with columns[index % 3]:
            values[field] = st.selectbox(labels[field], choices, key=field,
                                        format_func=(lambda v: "Yes" if v == "1" else "No")
                                        if field == "SeniorCitizen" else str)
    st.caption("If phone or internet service is No, related service fields are set to 'No service' automatically.")
    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submitted:
    try:
        result = predict_customer(model, values)
        st.subheader("Prediction result")
        a, b = st.columns(2)
        a.metric("Predicted outcome", "Likely to churn" if result["churn"] else "Likely to stay")
        b.metric("Estimated churn probability", f"{result['probability']:.1%}")
        st.progress(result["probability"])
        st.caption("A model estimate, not a certainty. Probabilities have not been separately calibrated.")
    except ValueError as error:
        st.error(str(error))

st.divider()
st.caption("Trained on public IBM sample data. Results may not generalize to other businesses or current customers.")
