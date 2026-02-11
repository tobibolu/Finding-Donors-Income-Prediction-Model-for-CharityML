from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import DataValidationError, validate_dataset


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [39, 50],
            "workclass": ["Private", "Self-emp-not-inc"],
            "education_level": ["Bachelors", "Bachelors"],
            "education-num": [13, 13],
            "marital-status": ["Never-married", "Married-civ-spouse"],
            "occupation": ["Adm-clerical", "Exec-managerial"],
            "relationship": ["Not-in-family", "Husband"],
            "race": ["White", "White"],
            "sex": ["Male", "Male"],
            "capital-gain": [2174, 0],
            "capital-loss": [0, 0],
            "hours-per-week": [40, 13],
            "native-country": ["United-States", "United-States"],
            "income": ["<=50K", ">50K"],
        }
    )


def test_validate_dataset_passes(valid_df: pd.DataFrame) -> None:
    validate_dataset(valid_df, target_col="income")


def test_validate_dataset_raises_on_missing(valid_df: pd.DataFrame) -> None:
    bad = valid_df.drop(columns=["age"])
    with pytest.raises(DataValidationError):
        validate_dataset(bad, target_col="income")
