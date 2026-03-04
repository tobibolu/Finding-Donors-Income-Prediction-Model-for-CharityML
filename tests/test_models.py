"""Tests for model training and evaluation."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_census, preprocess, split_data
from src.models import get_models, train_evaluate
from src.fairness import demographic_parity, disparate_impact_ratio

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    raw = load_census(DATA_PATH)
    X, y = preprocess(raw)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


class TestModels:
    def test_get_models_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)
        assert len(models) >= 4  # At least the base models

    def test_train_evaluate_returns_results(self, data):
        X_train, X_test, y_train, y_test = data
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        results = train_evaluate(model, X_train, y_train, X_test, y_test,
                                 sample_sizes=[0.01])
        assert len(results) == 1
        assert "test_accuracy" in results[0]
        assert results[0]["test_accuracy"] > 0.5


class TestFairness:
    def test_demographic_parity(self):
        y_pred = np.array([1, 0, 1, 0, 1, 1])
        import pandas as pd
        groups = pd.Series(["A", "A", "A", "B", "B", "B"])
        result = demographic_parity(y_pred, groups)
        assert len(result) == 2
        assert "positive_rate" in result.columns

    def test_disparate_impact(self):
        y_pred = np.array([1, 1, 1, 0, 0, 1])
        import pandas as pd
        groups = pd.Series(["M", "M", "M", "F", "F", "F"])
        ratio = disparate_impact_ratio(y_pred, groups, "M")
        assert 0 <= ratio <= 2.0
