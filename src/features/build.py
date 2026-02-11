from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_training_pipeline(
    categorical_features: list[str],
    numerical_features: list[str],
    random_state: int,
    rf_params: dict,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = RandomForestClassifier(random_state=random_state, **rf_params)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
    return pipe


def infer_feature_types(
    df_columns: list[str], target_col: str = "income"
) -> tuple[list[str], list[str]]:
    cat = [
        c
        for c in df_columns
        if c
        in {
            "workclass",
            "education_level",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        }
    ]
    num = [c for c in df_columns if c not in cat and c != target_col]
    return cat, num


def get_transformed_feature_names(pipe: Pipeline) -> np.ndarray:
    preprocess = pipe.named_steps["preprocess"]
    return preprocess.get_feature_names_out()
