from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

REQUIRED_INFERENCE_COLUMNS = [
    "age",
    "workclass",
    "education_level",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


class InferenceSchemaError(ValueError):
    """Raised for invalid inference input shape."""


def validate_inference_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INFERENCE_COLUMNS if c not in df.columns]
    if missing:
        raise InferenceSchemaError(f"Missing required columns: {missing}")


def load_model_artifact(path: str | Path) -> dict:
    artifact = joblib.load(path)
    return artifact


def predict_scores(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    validate_inference_input(df)
    pipe = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    prob = pipe.predict_proba(df[REQUIRED_INFERENCE_COLUMNS])[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df.copy()
    out["score_proba_high_income"] = prob
    out["prediction_high_income"] = pred
    return out
