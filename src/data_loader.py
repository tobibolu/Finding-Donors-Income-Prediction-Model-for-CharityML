"""Data loading, cleaning, and preprocessing for census income prediction."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_census(path: str = None) -> pd.DataFrame:
    """Load the census dataset.

    Args:
        path: Path to CSV. Defaults to data/census.csv.

    Returns:
        Raw census DataFrame with 14 features + income target.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "census.csv")
    return pd.read_csv(path)


def preprocess(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Full preprocessing pipeline: log transform, normalize, encode.

    Steps:
        1. Log-transform skewed features (capital-gain, capital-loss)
        2. Normalize numeric features to [0, 1]
        3. One-hot encode categorical features
        4. Binarize income target
        5. Engineer additional features

    Returns:
        Tuple of (features_df, target_series).
    """
    df = data.copy()

    # Feature engineering: additional derived features
    df["capital_net"] = df["capital-gain"] - df["capital-loss"]
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["young", "early_career", "mid_career", "senior", "retired"],
    )
    df["hours_category"] = pd.cut(
        df["hours-per-week"],
        bins=[0, 20, 40, 60, 100],
        labels=["part_time", "full_time", "overtime", "extreme"],
    )

    # Log-transform skewed features
    skewed = ["capital-gain", "capital-loss", "capital_net"]
    for col in skewed:
        # Handle negative values in capital_net
        if (df[col] < 0).any():
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        else:
            df[col] = np.log1p(df[col])

    # Normalize numeric features
    numeric = ["age", "education-num", "capital-gain", "capital-loss",
               "hours-per-week", "capital_net"]
    scaler = MinMaxScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    # Binarize target
    income = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["income"])

    # One-hot encode categoricals
    categorical = [
        "workclass", "education_level", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country", "age_bin", "hours_category",
    ]
    df = pd.get_dummies(df, columns=categorical)

    return df, income


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """Stratified train-test split."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)
