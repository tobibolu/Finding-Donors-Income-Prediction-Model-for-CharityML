from __future__ import annotations

from src.features.build import infer_feature_types


def test_infer_feature_types() -> None:
    cols = [
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
    cat, num = infer_feature_types(cols, target_col="income")

    assert "workclass" in cat
    assert "native-country" in cat
    assert "age" in num
    assert "income" not in num
