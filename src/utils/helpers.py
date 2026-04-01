"""Helpers for loading saved project artifacts into local demos and reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectSnapshot:
    """Compact summary of the saved project outputs used in the dashboard."""

    cleaned_rows: int
    modeled_rows: int
    feature_count: int
    target_count: int
    train_rows: int
    validation_rows: int
    test_rows: int
    date_start: str
    date_end: str
    temperature_threshold: float | None
    precipitation_threshold: float | None
    best_regression_models: list[dict]
    best_extreme_models: list[dict]
    mean_regression_metrics: dict[str, dict[str, float]]
    mean_extreme_metrics: dict[str, dict[str, float]]


def resolve_project_path(path_value: str | Path) -> Path:
    """Convert a relative project path into an absolute path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path_value: str | Path) -> dict:
    """Load a JSON file from the project workspace."""
    resolved_path = resolve_project_path(path_value)
    with resolved_path.open("r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


def read_csv(path_value: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file from the project workspace."""
    resolved_path = resolve_project_path(path_value)
    return pd.read_csv(resolved_path, **kwargs)


def display_target_name(target_variable: str) -> str:
    """Convert an internal target name into a reader-friendly label."""
    labels = {
        "temperature_2m": "Temperature",
        "precipitation": "Precipitation",
    }
    return labels.get(target_variable, target_variable.replace("_", " ").title())


def display_model_family(model_family: str) -> str:
    """Convert an internal model family into a reader-friendly label."""
    labels = {
        "baseline": "Baseline",
        "deep_learning": "Deep Learning",
    }
    return labels.get(model_family, model_family.replace("_", " ").title())


def load_project_snapshot() -> ProjectSnapshot:
    """Load the main saved summaries used by the local presentation dashboard."""
    eda_summary = read_json("reports/phase1/eda_summary.json")
    feature_summary = read_json("reports/phase2/feature_engineering_summary.json")
    evaluation_summary = read_json("reports/phase5/academic_evaluation_summary.json")

    thresholds_by_target: dict[str, float] = {}
    for threshold_record in evaluation_summary["extreme_thresholds"]:
        target_variable = threshold_record["target_variable"]
        if target_variable not in thresholds_by_target:
            thresholds_by_target[target_variable] = float(threshold_record["extreme_threshold"])

    return ProjectSnapshot(
        cleaned_rows=int(eda_summary["row_count"]),
        modeled_rows=int(feature_summary["supervised_row_count"]),
        feature_count=int(feature_summary["feature_count"]),
        target_count=int(feature_summary["target_count"]),
        train_rows=int(feature_summary["splits"]["train"]["row_count"]),
        validation_rows=int(feature_summary["splits"]["validation"]["row_count"]),
        test_rows=int(feature_summary["splits"]["test"]["row_count"]),
        date_start=str(eda_summary["date_start"]),
        date_end=str(eda_summary["date_end"]),
        temperature_threshold=thresholds_by_target.get("temperature_2m"),
        precipitation_threshold=thresholds_by_target.get("precipitation"),
        best_regression_models=evaluation_summary["best_regression_model_by_target"],
        best_extreme_models=evaluation_summary["best_extreme_model_by_target"],
        mean_regression_metrics=evaluation_summary["mean_regression_metrics_by_model_family"],
        mean_extreme_metrics=evaluation_summary["mean_extreme_metrics_by_model_family"],
    )
