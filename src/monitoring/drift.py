from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected, quantiles))
    if len(cut_points) < 3:
        return 0.0

    exp_bin = pd.cut(expected, bins=cut_points, include_lowest=True)
    act_bin = pd.cut(actual, bins=cut_points, include_lowest=True)

    exp_pct = exp_bin.value_counts(normalize=True).sort_index() + eps
    act_pct = act_bin.value_counts(normalize=True).sort_index() + eps

    psi = ((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum()
    return float(psi)
