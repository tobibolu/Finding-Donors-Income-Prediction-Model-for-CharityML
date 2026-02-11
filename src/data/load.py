from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
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
    "income",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
