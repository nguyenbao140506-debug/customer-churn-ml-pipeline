# Project scope

Build a small, reproducible customer churn portfolio project using Python,
Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit and Joblib.

- Public IBM Telco data; record original source, immutable revision and SHA256.
- Missing-value audit, target balance, basic cleaning and three useful EDA plots.
- Drop customerID; encode Churn as 0/1. Convert blank TotalCharges to missing.
- Stratified 80/20 train/test split with seed 42. Fit imputers, encoders and
  scaling only within sklearn Pipelines on training folds.
- Compare Logistic Regression and Random Forest. Select by mean training-only
  three-fold stratified cross-validation ROC-AUC before test evaluation.
- Report held-out accuracy, precision, recall, F1 and ROC-AUC for both models,
  plus a majority baseline, confusion matrices and ROC curves.
- Save the selected fitted pipeline; verify equivalent predictions after reload.
- Simple Streamlit form for customer details, prediction and probability.
- Execute training and basic preprocessing/prediction/app tests. Capture genuine
  screenshots of a live app. No invented metrics or screenshots.
- English README with results, sources, charts, screenshots, setup, reproducible
  commands, limitations and three evidence-based CV bullets.
- Prepare an independent Git repository with remote
  https://github.com/nguyenbao140506-debug/customer-churn-ml-pipeline.git.
  User approval is required before creating a public repository or pushing.

Original dataset:
https://github.com/IBM/telco-customer-churn-on-icp4d

Pinned revision: d5371f5d83a446ad5673cbcca3b814b926491f8a

CSV SHA256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91

Keep scope simple: no Docker, extensive tuning, commercial deployment or MLOps.
