# Execution evidence

Date: November 24, 2026 (Asia/Bangkok). This records the verified local portfolio
build and its subsequent user-approved GitHub publication.

## Environment and data

- Separate Git root: `customer-churn-ml-pipeline`; branch `codex/customer-churn`.
- Python 3.13.12, Windows; a fresh project virtual environment was created.
- Runtime package versions appear in `models/metadata.json`; full installed
  dependency closure appears in `requirements-lock.txt`.
- Pinned public IBM CSV: 970,457 bytes, 7,043 rows, 21 columns.
- Checksum validated before training; the Python downloader also successfully
  downloaded a fresh copy into an ignored temporary tooling directory.
- 5,174 non-churn / 1,869 churn; no duplicated customer IDs; 11 blank
  TotalCharges values converted to missing and retained for fold-local imputation.

## Executed commands and outcomes

| Command | Exit | Evidence |
| --- | ---: | --- |
| `python -m pip install --no-deps -e .` | 0 | Editable installation in a fresh environment |
| `python -m churn.train` | 0 | `reports/verification/training.txt` |
| `python -m pytest -q` | 0 | 13 passed in 11.46s; `reports/verification/tests.txt` |
| `python scripts/verify_reproducibility.py` | 0 | All CV/test metrics and selection match within 1e-12; `reports/verification/reproducibility.txt` |
| `python -m pip check` | 0 | No broken requirements found |
| `python -m streamlit run app.py --server.port 8502` | running | Live browser screenshots and two verified submissions |

An initial editable installation attempt could not reach the build dependency
registry inside the restricted sandbox; it succeeded with approved network access.
Initial Streamlit/test startup processes in the restricted sandbox did not
respond and were stopped. The successful final test run and local server ran
with the applicable approval. No failed attempt is counted as successful evidence.

## Actual results

5,634 training customers; 1,409 untouched held-out test customers. Model selection
occurred before test evaluation, based solely on training-only CV ROC-AUC.

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.845806 | 0.805536 | 0.657233 | 0.558824 | 0.604046 | 0.841871 |
| Random Forest | 0.835426 | 0.798439 | 0.658451 | 0.500000 | 0.568389 | 0.834296 |
| Majority baseline | — | 0.734564 | 0 | 0 | 0 | 0.500000 |

Saved Logistic Regression pipeline reload produced equivalent labels and
probabilities on the entire test partition. A second complete training run
matched every CV fold and all test metrics within absolute tolerance 1e-12.
`reports/test_predictions.csv` enables independent recalculation of metrics.

## Live application verification

The live app was opened in the in-app browser at the local server URL. The
Predict churn button was clicked after entering valid profiles. Outputs:

1. Default customer: churn, probability **0.5956829165568849** (UI **59.6%**).
2. Default customer with tenure=60, MonthlyCharges=60, TotalCharges=3600,
   Contract=Two year: stay, probability **0.07750537738861611** (UI **7.8%**).

Both values were independently checked by calling the reloaded pipeline from
Python. These are manually entered demonstration profiles, not evaluation rows.

Unaltered browser screenshots:

- `docs/screenshots/app_input.jpg`
- `docs/screenshots/app_prediction.jpg`
- `docs/screenshots/app_prediction_stay.jpg`

All five Matplotlib output figures were visually inspected. Text and values
are readable; the confusion matrices match the saved metrics.

## Review and publication

The required ChatGPT PLAN was received after workspace identity and task-file
verification through the dedicated read-only connector. The independent final
review returned **STATE: DONE / REVIEW: PASS** for task `c2c_7c91`, iteration 1.
It inspected source files, staged diffs, README, metrics and execution outputs
through that connector and identified no blocking implementation defects.

Review boundary: the reviewer retrieved actual pytest output and saved training/
reproducibility records; it did not independently execute commands. It confirmed
the screenshot files via image metadata but could not inspect their pixels.
Codex separately inspected the actual browser captures and all five plot images.
The local secret-pattern scan passed; the intended remote was verified locally.
These local review checks are separate from the remote publication verification
recorded below.

Known limitations are documented in the README: historical public sample,
random rather than temporal validation, class imbalance, limited recall,
uncalibrated probabilities and no production/business-effect claims.

## Approved GitHub publication

The user explicitly approved creating the public repository and pushing the
completed project. GitHub authentication confirmed the account
`nguyenbao140506-debug` before publication.

- Repository: https://github.com/nguyenbao140506-debug/customer-churn-ml-pipeline
- Visibility: PUBLIC; default branch: `main`.
- Published implementation commit: `2812059c480e6833c35a6c227995f48ae7644175`.
- GitHub API verification confirmed that the remote implementation commit
  matched the local commit and that all 40 tracked files were present, including
  source, saved model, evaluation results, figures and three real screenshots.
- The virtual environment, local tooling, raw dataset and temporary logs remain
  excluded. This documentation update records publication after that verification.
