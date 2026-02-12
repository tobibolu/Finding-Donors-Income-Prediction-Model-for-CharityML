"""Model training, evaluation, and comparison for income prediction.

Includes baseline models, modern gradient boosting (XGBoost, LightGBM),
and a proper sklearn Pipeline for production readiness.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, fbeta_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_models() -> dict:
    """Return a dictionary of candidate models.

    Includes both classic models and modern gradient boosting for a
    comprehensive comparison.
    """
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            random_state=42, n_estimators=100, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            random_state=42, n_estimators=100, verbose=-1,
        )
    except ImportError:
        pass

    return models


def train_evaluate(model, X_train, y_train, X_test, y_test,
                   sample_sizes: list[float] = None) -> list[dict]:
    """Train a model at various sample sizes and evaluate performance.

    Args:
        model: sklearn-compatible classifier.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        sample_sizes: Fractions of training data to use (default: [0.01, 0.1, 1.0]).

    Returns:
        List of result dicts with metrics for each sample size.
    """
    if sample_sizes is None:
        sample_sizes = [0.01, 0.1, 1.0]

    results = []
    for frac in sample_sizes:
        n = max(1, int(len(X_train) * frac))
        X_sub, y_sub = X_train.iloc[:n], y_train.iloc[:n]

        start = time.time()
        model.fit(X_sub, y_sub)
        train_time = time.time() - start

        start = time.time()
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_sub[:min(300, n)])
        pred_time = time.time() - start

        results.append({
            "sample_frac": frac,
            "sample_size": n,
            "train_time": round(train_time, 3),
            "pred_time": round(pred_time, 3),
            "train_accuracy": round(accuracy_score(y_sub[:min(300, n)], y_pred_train), 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred_test), 4),
            "train_fbeta": round(fbeta_score(y_sub[:min(300, n)], y_pred_train, beta=0.5), 4),
            "test_fbeta": round(fbeta_score(y_test, y_pred_test, beta=0.5), 4),
        })
    return results


def compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Compare all candidate models and return a summary DataFrame."""
    all_results = []
    models = get_models()

    for name, model in models.items():
        results = train_evaluate(model, X_train, y_train, X_test, y_test)
        for r in results:
            r["model"] = name
        all_results.extend(results)

    return pd.DataFrame(all_results)


def optimize_model(X_train, y_train, model_class=None, param_grid=None) -> tuple:
    """Optimize a model using GridSearchCV.

    Returns:
        Tuple of (best_model, grid_search_object).
    """
    if model_class is None:
        model_class = GradientBoostingClassifier(random_state=42)
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.05, 0.1],
            "min_samples_split": [2, 5],
        }

    grid = GridSearchCV(model_class, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid


def get_classification_report(model, X_test, y_test) -> dict:
    """Generate comprehensive classification metrics."""
    y_pred = model.predict(X_test)

    report = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "fbeta_0.5": round(fbeta_score(y_test, y_pred, beta=0.5), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        report["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
        report["avg_precision"] = round(average_precision_score(y_test, y_proba), 4)

    return report
