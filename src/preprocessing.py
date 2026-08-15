from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def build_preprocessor(categorical, numeric):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
