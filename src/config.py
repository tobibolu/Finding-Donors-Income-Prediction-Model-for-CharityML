from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    random_state: int
    input_data_path: Path
    model_output_path: Path
    features_output_path: Path
    report_output_path: Path
    target_col: str
    test_size: float
    positive_label: str
    threshold_beta: float
    rf_params: dict[str, Any]


def load_config(config_path: str | Path) -> TrainConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return TrainConfig(
        random_state=int(data["random_state"]),
        input_data_path=Path(data["input_data_path"]),
        model_output_path=Path(data["model_output_path"]),
        features_output_path=Path(data["features_output_path"]),
        report_output_path=Path(data["report_output_path"]),
        target_col=str(data["target_col"]),
        test_size=float(data["test_size"]),
        positive_label=str(data["positive_label"]),
        threshold_beta=float(data["threshold_beta"]),
        rf_params=dict(data["rf_params"]),
    )
