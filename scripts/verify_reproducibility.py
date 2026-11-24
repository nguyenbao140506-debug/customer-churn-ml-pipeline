"""Rerun training and compare metrics with the previous execution (atol=1e-12)."""

import json
import subprocess
import sys

import numpy as np
from churn.data import ROOT

path = ROOT / "reports/metrics.json"
before = json.loads(path.read_text())
subprocess.run([sys.executable, "-m", "churn.train"], cwd=ROOT, check=True)
after = json.loads(path.read_text())
assert before["selected_model"] == after["selected_model"]
for name in before["test"]:
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        np.testing.assert_allclose(before["test"][name][metric], after["test"][name][metric],
                                   rtol=0, atol=1e-12)
for name in before["cv"]:
    np.testing.assert_allclose(before["cv"][name]["fold_roc_auc"],
                               after["cv"][name]["fold_roc_auc"], rtol=0, atol=1e-12)
print("Reproducibility PASS: selection, all CV folds, and all test metrics match within 1e-12.")
