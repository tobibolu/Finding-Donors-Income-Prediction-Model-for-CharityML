from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import load_config
from src.data.load import load_dataset
from src.data.validate import validate_dataset
from src.features.build import (
    build_training_pipeline,
    get_transformed_feature_names,
    infer_feature_types,
)
from src.models.evaluate import compute_metrics, optimize_threshold


def _write_model_card(
    output_path: Path,
    cv_mean: float,
    cv_std: float,
    threshold: float,
    metrics: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Charity Donor Model Card",
        "",
        "## Validation Summary",
        f"- CV ROC-AUC mean: {cv_mean:.4f}",
        f"- CV ROC-AUC std: {cv_std:.4f}",
        f"- Chosen probability threshold: {threshold:.4f}",
        "",
        "## Holdout Metrics",
    ]
    text.extend([f"- {k}: {v:.4f}" for k, v in metrics.items()])
    text.extend(
        [
            "",
            "## Caveats",
            "- Dataset is historical census proxy and may not fully represent current donor populations.",
            "- Fairness diagnostics should be reviewed before any production outreach decisions.",
        ]
    )
    output_path.write_text("\n".join(text), encoding="utf-8")


def run_training(config_path: str) -> None:
    cfg = load_config(config_path)

    df = load_dataset(cfg.input_data_path)
    validate_dataset(df, target_col=cfg.target_col)

    X = df.drop(columns=[cfg.target_col])
    y = (df[cfg.target_col] == cfg.positive_label).astype(int)

    cat_features, num_features = infer_feature_types(list(X.columns), target_col=cfg.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipe = build_training_pipeline(
        categorical_features=cat_features,
        numerical_features=num_features,
        random_state=cfg.random_state,
        rf_params=cfg.rf_params,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    threshold, _ = optimize_threshold(y_test.to_numpy(), y_prob, beta=cfg.threshold_beta)
    metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold)

    cfg.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold}, cfg.model_output_path)

    feature_names = get_transformed_feature_names(pipe)
    cfg.features_output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.features_output_path.write_text("\n".join(feature_names), encoding="utf-8")

    _write_model_card(
        output_path=cfg.report_output_path,
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        threshold=threshold,
        metrics=metrics,
    )

    print("Training complete")
    print(f"Saved pipeline artifact to: {cfg.model_output_path}")
    print(f"Saved transformed features to: {cfg.features_output_path}")
    print(f"Saved model card to: {cfg.report_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CharityML donor model pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
