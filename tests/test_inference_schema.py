from __future__ import annotations

import pandas as pd
import pytest

from src.models.predict import InferenceSchemaError, validate_inference_input


def test_inference_schema_valid() -> None:
    df = pd.DataFrame(
        {
            "age": [39],
            "workclass": ["Private"],
            "education_level": ["Bachelors"],
            "education-num": [13],
            "marital-status": ["Never-married"],
            "occupation": ["Adm-clerical"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "capital-gain": [2174],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
    )
    validate_inference_input(df)


def test_inference_schema_missing_col() -> None:
    df = pd.DataFrame({"age": [39]})
    with pytest.raises(InferenceSchemaError):
        validate_inference_input(df)
