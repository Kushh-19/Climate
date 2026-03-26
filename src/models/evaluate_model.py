"""Shared regression and extreme-event evaluation helpers for Phases 3 to 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score


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


@dataclass(frozen=True)
class ClassificationMetrics:
    """Container for classification metrics used in extreme-event evaluation."""

    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert the classification object into a JSON-serializable dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }


class ExtremeEventMetricCalculator:
    """Computes threshold-based classification metrics for extreme events."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        """Calculate precision, recall, F1, and confusion counts."""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
        false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
        false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
        true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))

        return ClassificationMetrics(
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
        )
