"""Small checks of data boundaries, saved inference, and the real app path."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from churn.data import DATA_SHA256, FEATURES, NUMERIC, ROOT, clean_features, ensure_dataset, load_dataset
from churn.model import build_pipeline
from churn.predict import DEFAULT_CUSTOMER, customer_frame, load_model, predict_customer
from churn.train import evaluate, split_data


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_verified_dataset_and_no_identifier_or_target_features(dataset):
    x, y, raw = dataset
    assert len(raw) == 7043
    assert len(FEATURES) == 19
    assert "customerID" not in x and "Churn" not in x
    assert set(y) == {0, 1}
    assert raw.Churn.eq("Yes").sum() == y.sum()
    assert clean_features(x).TotalCharges.isna().sum() == 11


def test_checksum_rejects_modified_bytes(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("altered content", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ensure_dataset(path)


def test_cleaning_is_stateless_and_does_not_mutate_input(dataset):
    x, _, _ = dataset
    sample = x.head(3).copy()
    sample["TotalCharges"] = ["  ", "12.5", " 20 "]
    cleaned = clean_features(sample)
    assert np.isnan(cleaned.TotalCharges.iloc[0])
    assert cleaned.TotalCharges.iloc[1:].tolist() == [12.5, 20.0]
    assert sample.TotalCharges.iloc[0] == "  "


def test_split_is_disjoint_deterministic_and_stratified(dataset):
    x, y, _ = dataset
    train_x, test_x, train_y, test_y = split_data(x, y)
    assert set(train_x.index).isdisjoint(test_x.index)
    assert len(train_x) == 5634 and len(test_x) == 1409
    assert train_x.index.equals(split_data(x, y)[0].index)
    assert abs(train_y.mean() - test_y.mean()) < 0.001


def test_preprocessing_learns_only_training_statistics_and_handles_unseen(dataset):
    x, y, _ = dataset
    train_x, test_x, train_y, _ = split_data(x, y)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(train_x, train_y)
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    np.testing.assert_allclose(numeric.named_steps["impute"].statistics_,
                               clean_features(train_x)[NUMERIC].median().to_numpy())
    sample = test_x.head(2).copy()
    sample["PaymentMethod"] = "Previously unseen category"
    sample["TotalCharges"] = " "
    probability = pipeline.predict_proba(sample)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1)


def test_saved_pipeline_reloads_equivalently(tmp_path):
    model = load_model()
    frame = customer_frame(DEFAULT_CUSTOMER)
    path = tmp_path / "roundtrip.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(frame), model.predict_proba(frame), atol=1e-12)
    result = predict_customer(model, DEFAULT_CUSTOMER)
    assert 0 <= result["probability"] <= 1
    assert result["churn"] == (result["probability"] >= 0.5)


@pytest.mark.parametrize("changes", [{"MonthlyCharges": -1}, {"tenure": 73},
                                      {"TotalCharges": float("nan")}, {"Contract": "Invalid"}])
def test_prediction_rejects_invalid_inputs(changes):
    with pytest.raises(ValueError):
        customer_frame({**DEFAULT_CUSTOMER, **changes})


def test_missing_field_and_service_dependencies():
    with pytest.raises(ValueError, match="Missing"):
        customer_frame({})
    frame = customer_frame({**DEFAULT_CUSTOMER, "PhoneService": "No", "InternetService": "No"})
    assert frame.MultipleLines.iloc[0] == "No phone service"
    assert frame.OnlineSecurity.iloc[0] == "No internet service"


def test_report_metrics_recompute_from_saved_probabilities():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "reports/test_predictions.csv")
    for name in metrics["cv"]:
        actual = evaluate(predictions.actual, predictions[name].to_numpy())
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert actual[metric] == pytest.approx(metrics["test"][name][metric], abs=1e-12)
    selected = max(metrics["cv"], key=lambda name: metrics["cv"][name]["mean"])
    assert selected == metrics["selected_model"]
    assert json.loads((ROOT / "models/metadata.json").read_text())["dataset_sha256"] == DATA_SHA256


def test_streamlit_submits_a_real_saved_model_prediction():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert not app.metric
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    expected = predict_customer(load_model(), DEFAULT_CUSTOMER)
    assert app.metric[1].value == f"{expected['probability']:.1%}"
    assert app.metric[0].value == ("Likely to churn" if expected["churn"] else "Likely to stay")
