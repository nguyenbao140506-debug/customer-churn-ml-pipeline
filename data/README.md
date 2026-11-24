# Dataset provenance

This project uses the public **IBM Telco Customer Churn** sample CSV, published in
[IBM's example repository](https://github.com/IBM/telco-customer-churn-on-icp4d).
It contains 7,043 rows and 21 columns. `Churn=Yes` is the positive class.

Original download:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

For reproducibility, the downloader uses immutable revision
`d5371f5d83a446ad5673cbcca3b814b926491f8a` and validates SHA256
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
See `provenance.json` for the exact pinned URL.

Run `python -m churn.data` to download and verify `data/raw/telco_churn.csv`.
Training also downloads it automatically if it is absent. A modified existing
file is rejected. Raw data is excluded from Git; refer to IBM's source repository
for its license and dataset terms. The MIT license in this project covers our
source code, not ownership of the IBM dataset.
