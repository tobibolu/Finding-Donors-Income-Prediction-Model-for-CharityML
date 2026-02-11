"""Visualization utilities for the donor prediction project."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_income_distribution(data: pd.DataFrame) -> None:
    """Plot income distribution and age distribution by income level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    income_dist = data["income"].value_counts()
    sns.barplot(x=income_dist.index, y=income_dist.values, ax=ax1)
    ax1.set_title("Income Distribution")
    ax1.set_ylabel("Count")

    sns.kdeplot(data=data, x="age", hue="income", common_norm=False, ax=ax2)
    ax2.set_title("Age Distribution by Income Level")

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(data: pd.DataFrame, features: list) -> None:
    """Plot distributions of numeric features by income level."""
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.histplot(data=data, x=feat, hue="income", bins=30,
                     multiple="stack", ax=axes[i])
        axes[i].set_title(f"Distribution of {feat}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Plot test accuracy and F-beta comparison across models."""
    full = results_df[results_df["sample_frac"] == 1.0].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=full, x="model", y="test_accuracy", ax=ax1)
    ax1.set_title("Test Accuracy by Model")
    ax1.set_ylim(0.7, 0.9)
    ax1.tick_params(axis="x", rotation=45)

    sns.barplot(data=full, x="model", y="test_fbeta", ax=ax2)
    ax2.set_title("Test F-beta (0.5) by Model")
    ax2.set_ylim(0.5, 0.8)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(model, X_test, y_test) -> None:
    """Plot ROC curve and Precision-Recall curve side by side."""
    y_proba = model.predict_proba(X_test)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = np.trapezoid(precision, recall)
    ax2.plot(recall, precision, lw=2, label=f"PR (AP = {ap:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    """Plot top feature importances."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance.head(top_n), x="importance", y="feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_fairness_results(fairness_data: dict) -> None:
    """Visualize fairness metrics across demographic groups."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, attr in enumerate(["sex", "race"]):
        key = f"{attr}_equalized_odds"
        if key not in fairness_data:
            continue
        df = fairness_data[key]
        x = np.arange(len(df))
        width = 0.25

        axes[idx].bar(x - width, df["accuracy"], width, label="Accuracy")
        axes[idx].bar(x, df["tpr"], width, label="TPR")
        axes[idx].bar(x + width, df["fpr"], width, label="FPR")
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(df["group"], rotation=45, ha="right")
        axes[idx].set_title(f"Fairness Metrics by {attr.title()}")
        axes[idx].legend()
        axes[idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
