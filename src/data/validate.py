from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.data.load import EXPECTED_COLUMNS


class DataValidationError(ValueError):
    """Raised when input dataset fails validation checks."""


def _assert_columns(df: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise DataValidationError(f"Schema mismatch. missing={missing}, extra={extra}")


def _assert_non_negative(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if (df[col] < 0).any():
            raise DataValidationError(f"Column '{col}' has negative values.")


def _assert_label_values(df: pd.DataFrame, target_col: str) -> None:
    allowed = {">50K", "<=50K"}
    observed = set(df[target_col].dropna().unique())
    if not observed.issubset(allowed):
        raise DataValidationError(
            f"Target column '{target_col}' has unexpected values: {sorted(observed - allowed)}"
        )


def validate_dataset(df: pd.DataFrame, target_col: str = "income") -> None:
    _assert_columns(df, EXPECTED_COLUMNS)

    if df.duplicated().sum() > 0:
        raise DataValidationError("Dataset contains duplicated rows.")

    if df.isna().sum().sum() > 0:
        raise DataValidationError("Dataset contains missing values.")

    _assert_non_negative(
        df, ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    )

    _assert_label_values(df, target_col)
