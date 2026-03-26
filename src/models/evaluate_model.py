"""Evaluation helpers for Phase 3 baseline regression models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for core regression metrics used in baseline evaluation."""

    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        """Convert the metric object into a JSON-serializable dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
        }


class RegressionMetricCalculator:
    """Computes regression metrics in original target units."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
        """Calculate RMSE, MAE, and R^2 for a regression output."""
        return RegressionMetrics(
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
        )
