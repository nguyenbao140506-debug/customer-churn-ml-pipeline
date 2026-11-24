"""Learn preprocessing within a Pipeline, including each validation fold."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import CATEGORIES, NUMERIC, clean_features


def build_pipeline(classifier):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean_features, validate=False)),
        ("preprocess", ColumnTransformer([
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, list(CATEGORIES)),
        ])),
        ("classifier", classifier),
    ])
