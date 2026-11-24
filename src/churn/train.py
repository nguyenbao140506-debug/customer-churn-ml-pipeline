"""Run: python -m churn.train. Select on CV, evaluate once on held-out data."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn.data import DATA_SHA256, FEATURES, REVISION, ROOT, SOURCE_URL, clean_features, load_dataset
from churn.eda import save_eda
from churn.model import build_pipeline

SEED = 42
THRESHOLD = 0.5


def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, stratify=target, random_state=SEED)


def evaluate(target, probability):
    predicted = (probability >= THRESHOLD).astype(int)
    return {"accuracy": accuracy_score(target, predicted),
            "precision": precision_score(target, predicted, zero_division=0),
            "recall": recall_score(target, predicted, zero_division=0),
            "f1": f1_score(target, predicted, zero_division=0),
            "roc_auc": roc_auc_score(target, probability),
            "confusion_matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist()}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    features, target, raw = load_dataset()
    train_x, test_x, train_y, test_y = split_data(features, target)
    reports, figures, models = ROOT / "reports", ROOT / "reports/figures", ROOT / "models"
    for path in [reports, figures, models]:
        path.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "data/provenance.json", {
        "source_repository": "https://github.com/IBM/telco-customer-churn-on-icp4d",
        "revision": REVISION, "download_url": SOURCE_URL, "sha256": DATA_SHA256,
        "rows": len(raw), "columns": len(raw.columns),
        "note": "Public IBM sample data. Raw CSV is downloaded locally, not redistributed in this repository.",
    })
    write_json(reports / "data_audit.json", {
        "rows": len(raw), "columns": len(raw.columns),
        "raw_null_counts": raw.isna().sum().to_dict(),
        "cleaned_missing_counts": clean_features(features).isna().sum().to_dict(),
        "duplicate_customer_ids": int(raw.customerID.duplicated().sum()),
        "target_counts": raw.Churn.value_counts().to_dict(),
        "train_rows": len(train_y), "test_rows": len(test_y),
        "train_churn_rate": float(train_y.mean()), "test_churn_rate": float(test_y.mean()),
    })
    write_json(reports / "eda_summary.json", save_eda(train_x, train_y, figures))
    write_json(reports / "split.json", {"seed": SEED, "train_indices": train_x.index.tolist(),
                                        "test_indices": test_x.index.tolist()})

    candidates = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=1)),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, train_x, train_y, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_results[name] = {"fold_roc_auc": scores.tolist(), "mean": float(scores.mean()),
                            "std": float(scores.std())}
        print(f"{name} training CV ROC-AUC: {scores.mean():.6f}", flush=True)
    # Stable insertion order resolves exact ties in favor of Logistic Regression.
    selected = max(candidates, key=lambda name: cv_results[name]["mean"])
    print(f"Selected BEFORE test evaluation: {selected}", flush=True)
    results, probabilities = {}, {}
    for name, pipeline in candidates.items():
        pipeline.fit(train_x, train_y)
        probabilities[name] = pipeline.predict_proba(test_x)[:, 1]
        results[name] = evaluate(test_y, probabilities[name])
    dummy = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    results["Majority baseline"] = evaluate(test_y, dummy.predict_proba(test_x)[:, 1])

    model_path = models / "churn_pipeline.joblib"
    joblib.dump(candidates[selected], model_path, compress=3)
    restored = joblib.load(model_path)
    np.testing.assert_array_equal(restored.predict(test_x), candidates[selected].predict(test_x))
    np.testing.assert_allclose(restored.predict_proba(test_x), candidates[selected].predict_proba(test_x),
                               rtol=0, atol=1e-12)

    versions = {p: importlib.metadata.version(p) for p in
                ["pandas", "numpy", "scikit-learn", "matplotlib", "streamlit", "joblib"]}
    metadata = {"selected_model": selected, "selection_rule": "Highest mean training-only 3-fold CV ROC-AUC",
                "threshold": THRESHOLD, "seed": SEED, "features": FEATURES,
                "fitted_on": "5634 training rows only; test partition never fitted",
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "dataset_sha256": DATA_SHA256, "python": platform.python_version(), "versions": versions}
    write_json(models / "metadata.json", metadata)
    write_json(reports / "metrics.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "threshold": THRESHOLD, "train_rows": len(train_y), "test_rows": len(test_y),
        "positive_class": "Churn=Yes (1)", "selected_model": selected,
        "selection_rule": metadata["selection_rule"], "cv": cv_results, "test": results,
        "reload_prediction_equivalence": True, "versions": versions,
    })
    comparison = pd.DataFrame([
        {"model": name, "cv_roc_auc": cv_results.get(name, {}).get("mean", np.nan),
         **{k: v for k, v in row.items() if k != "confusion_matrix"}}
        for name, row in results.items()
    ])
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    pd.DataFrame({"row_index": test_x.index, "actual": test_y.to_numpy(),
                  **{name: p for name, p in probabilities.items()}}).to_csv(
        reports / "test_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, row) in zip(axes, list(results.items())[:2]):
        ConfusionMatrixDisplay(np.array(row["confusion_matrix"]),
                               display_labels=["Stayed", "Churned"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.suptitle("Held-out test set · threshold 0.50")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in probabilities.items():
        fpr, tpr, _ = roc_curve(test_y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance (AUC 0.500)")
    ax.set(title="ROC curves on held-out customers", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "roc_curves.png", dpi=160)
    plt.close(fig)
    print(comparison.to_string(index=False), flush=True)
    print("Saved pipeline reload: predictions and probabilities match.", flush=True)


if __name__ == "__main__":
    main()
