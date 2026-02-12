"""Tests for data loading and preprocessing."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_census, preprocess, split_data

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "census.csv")


@pytest.fixture(scope="module")
def raw_data():
    return load_census(DATA_PATH)


@pytest.fixture(scope="module")
def processed(raw_data):
    return preprocess(raw_data)


class TestLoadCensus:
    def test_loads_dataframe(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)

    def test_has_14_columns(self, raw_data):
        assert raw_data.shape[1] == 14

    def test_has_income_column(self, raw_data):
        assert "income" in raw_data.columns

    def test_no_missing_values(self, raw_data):
        assert raw_data.isnull().sum().sum() == 0

    def test_correct_row_count(self, raw_data):
        assert len(raw_data) > 40000


class TestPreprocess:
    def test_returns_tuple(self, processed):
        X, y = processed
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_binary(self, processed):
        _, y = processed
        assert set(y.unique()) == {0, 1}

    def test_no_income_in_features(self, processed):
        X, _ = processed
        assert "income" not in X.columns

    def test_features_have_engineered_columns(self, processed):
        X, _ = processed
        col_names = " ".join(X.columns)
        assert "capital_net" in col_names

    def test_numeric_features_normalized(self, processed):
        X, _ = processed
        assert X["age"].max() <= 1.0
        assert X["age"].min() >= 0.0


class TestSplitData:
    def test_split_sizes(self, processed):
        X, y = processed
        X_train, X_test, y_train, y_test = split_data(X, y)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert abs(len(X_test) / total - 0.2) < 0.01

    def test_stratified(self, processed):
        X, y = processed
        _, _, y_train, y_test = split_data(X, y)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.02
