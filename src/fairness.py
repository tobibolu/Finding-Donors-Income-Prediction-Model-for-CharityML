"""Fairness and bias analysis for income prediction.

Evaluates model predictions across protected demographic groups (race, sex)
to identify potential biases and disparate impact.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix


def demographic_parity(y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Calculate prediction rates across demographic groups.

    Demographic parity requires that the probability of a positive prediction
    is the same across all groups.

    Returns:
        DataFrame with positive prediction rate per group.
    """
    results = []
    for group in sorted(groups.unique()):
        mask = groups == group
        rate = y_pred[mask].mean()
        results.append({
            "group": group,
            "count": int(mask.sum()),
            "positive_rate": round(rate, 4),
        })

    df = pd.DataFrame(results)
    overall_rate = y_pred.mean()
    df["disparity_ratio"] = round(df["positive_rate"] / overall_rate, 4)
    return df


def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray,
                   groups: pd.Series) -> pd.DataFrame:
    """Calculate true positive rate and false positive rate by group.

    Equalized odds requires that TPR and FPR are equal across groups.
    """
    results = []
    for group in sorted(groups.unique()):
        mask = groups == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = accuracy_score(y_t, y_p)

        results.append({
            "group": group,
            "count": int(mask.sum()),
            "accuracy": round(acc, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "fbeta_0.5": round(fbeta_score(y_t, y_p, beta=0.5, zero_division=0), 4),
        })

    return pd.DataFrame(results)


def disparate_impact_ratio(y_pred: np.ndarray,
                           protected: pd.Series,
                           privileged_value) -> float:
    """Calculate the disparate impact ratio.

    The 4/5ths rule: if the selection rate for the unprivileged group is
    less than 80% of the privileged group's rate, there may be adverse impact.

    Returns:
        Minimum ratio across all unprivileged groups vs privileged group.
    """
    priv_mask = protected == privileged_value
    priv_rate = y_pred[priv_mask].mean()

    if priv_rate == 0:
        return float("inf")

    ratios = []
    for group in protected.unique():
        if group == privileged_value:
            continue
        group_mask = protected == group
        group_rate = y_pred[group_mask].mean()
        ratios.append(group_rate / priv_rate)

    return round(min(ratios), 4) if ratios else 1.0


def fairness_summary(y_true: np.ndarray, y_pred: np.ndarray,
                     data: pd.DataFrame) -> dict:
    """Run full fairness analysis across sex and race.

    Returns:
        Dictionary with demographic parity, equalized odds, and
        disparate impact results for both protected attributes.
    """
    results = {}

    for attr, privileged in [("sex", "Male"), ("race", "White")]:
        if attr not in data.columns:
            continue
        groups = data[attr]
        results[f"{attr}_demographic_parity"] = demographic_parity(y_pred, groups)
        results[f"{attr}_equalized_odds"] = equalized_odds(y_true, y_pred, groups)
        results[f"{attr}_disparate_impact"] = disparate_impact_ratio(
            y_pred, groups, privileged
        )

    return results
