"""Phase 3 pipeline for baseline forecasting models.

This module trains and compares classical machine-learning baselines for
multi-horizon temperature and precipitation forecasting. It follows a
time-series-safe protocol:
1. Load the engineered Phase 2 dataset.
2. Train candidate baselines on the training split.
3. Select the best model per target family using validation RMSE.
4. Refit the selected model on train plus validation.
5. Evaluate once on the held-out test split.
6. Save metrics, predictions, and model artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    from src.models.evaluate_model import RegressionMetricCalculator
except ModuleNotFoundError:
    from evaluate_model import RegressionMetricCalculator


@dataclass(frozen=True)
class PhaseThreeConfig:
    """Configuration for Phase 3 baseline model training."""

    dataset_path: Path = Path("data/processed/phase2/baroda_supervised_unscaled.csv.gz")
    phase_two_artifact_path: Path = Path("models/artifacts/phase2_feature_scaler.pkl")
    model_dir: Path = Path("models/phase3/baselines")
    validation_metrics_path: Path = Path("reports/phase3/baseline_validation_metrics.csv")
    test_metrics_path: Path = Path("reports/phase3/baseline_test_metrics.csv")
    summary_path: Path = Path("reports/phase3/baseline_selection_summary.json")
    predictions_path: Path = Path("data/processed/phase3/baseline_test_predictions.csv.gz")
    random_state: int = 42


class PersistenceRegressor:
    """Naive time-series baseline that repeats the current observed value forward."""

    def __init__(self, source_column: str, output_count: int, uses_log_transform: bool) -> None:
        self.source_column = source_column
        self.output_count = output_count
        self.uses_log_transform = uses_log_transform

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "PersistenceRegressor":
        """Store feature names for consistency with the scikit-learn interface."""
        self.feature_names_in_ = features.columns.to_list()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Replicate the current source variable across all forecast horizons."""
        base_values = features[self.source_column].to_numpy(dtype=np.float32)
        if self.uses_log_transform:
            base_values = np.log1p(base_values)

        return np.repeat(base_values[:, None], repeats=self.output_count, axis=1)


class ClimateBaselineTrainer:
    """Train, select, and persist baseline regressors for each target horizon."""

    TARGET_PATTERN = re.compile(r"(?P<variable>.+)_target_t_plus_(?P<horizon>\d+)h")
    ROLLING_PATTERN = re.compile(
        r"(?P<base>.+)_rolling_(?P<stat>mean|std|sum)_(?P<window>\d+)h"
    )

    def __init__(self, config: PhaseThreeConfig) -> None:
        self.config = config

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Execute the full baseline-model training workflow."""
        self._ensure_output_directories()

        dataframe, feature_columns, target_columns, split_indices = self.load_phase_two_artifacts()
        candidate_builders = self.get_candidate_model_builders()
        target_groups = self.build_target_groups(target_columns)
        baseline_feature_columns = self.select_baseline_feature_columns(feature_columns)

        train_frame = dataframe.iloc[split_indices["train_start"] : split_indices["train_end"]]
        validation_frame = dataframe.iloc[
            split_indices["validation_start"] : split_indices["validation_end"]
        ]
        train_validation_frame = dataframe.iloc[
            split_indices["train_start"] : split_indices["validation_end"]
        ]
        test_frame = dataframe.iloc[split_indices["test_start"] : split_indices["test_end"]]

        x_train = train_frame[baseline_feature_columns]
        x_validation = validation_frame[baseline_feature_columns]
        x_train_validation = train_validation_frame[baseline_feature_columns]
        x_test = test_frame[baseline_feature_columns]

        validation_records: list[dict] = []
        test_records: list[dict] = []
        prediction_records: list[dict] = []
        selected_models: list[dict] = []

        for target_variable, grouped_target_columns in target_groups.items():
            uses_log_transform = target_variable == "precipitation"

            y_train = train_frame[grouped_target_columns].to_numpy()
            y_validation = validation_frame[grouped_target_columns].to_numpy()

            best_model_name: str | None = None
            best_validation_rmse_mean = float("inf")
            candidate_validation_scores: dict[str, float] = {}

            for model_name, model_builder in candidate_builders.items():
                model = model_builder(
                    target_variable=target_variable,
                    output_count=len(grouped_target_columns),
                    uses_log_transform=uses_log_transform,
                )
                self.fit_model(
                    model=model,
                    features=x_train,
                    target=y_train,
                    uses_log_transform=uses_log_transform,
                )

                validation_predictions = self.predict_with_model(
                    model=model,
                    features=x_validation,
                    uses_log_transform=uses_log_transform,
                )

                horizon_rmses: list[float] = []
                for target_index, target_column in enumerate(grouped_target_columns):
                    target_metadata = self.parse_target_column(target_column)
                    validation_metrics = RegressionMetricCalculator.compute(
                        y_true=y_validation[:, target_index],
                        y_pred=validation_predictions[:, target_index],
                    )
                    validation_records.append(
                        {
                            "target_column": target_column,
                            "target_variable": target_variable,
                            "horizon_hours": target_metadata["horizon_hours"],
                            "model_name": model_name,
                            "target_transform": "log1p" if uses_log_transform else "none",
                            **validation_metrics.to_dict(),
                        }
                    )
                    horizon_rmses.append(validation_metrics.rmse)

                candidate_validation_scores[model_name] = float(np.mean(horizon_rmses))

                if candidate_validation_scores[model_name] < best_validation_rmse_mean:
                    best_validation_rmse_mean = candidate_validation_scores[model_name]
                    best_model_name = model_name

            if best_model_name is None:
                raise RuntimeError(f"Failed to select a model for target group {target_variable}.")

            final_model = candidate_builders[best_model_name](
                target_variable=target_variable,
                output_count=len(grouped_target_columns),
                uses_log_transform=uses_log_transform,
            )
            y_train_validation = train_validation_frame[grouped_target_columns].to_numpy()
            y_test = test_frame[grouped_target_columns].to_numpy()

            self.fit_model(
                model=final_model,
                features=x_train_validation,
                target=y_train_validation,
                uses_log_transform=uses_log_transform,
            )

            test_predictions = self.predict_with_model(
                model=final_model,
                features=x_test,
                uses_log_transform=uses_log_transform,
            )

            model_path = self.save_model_artifact(
                model=final_model,
                model_name=best_model_name,
                target_variable=target_variable,
                target_columns=grouped_target_columns,
                feature_columns=baseline_feature_columns,
                uses_log_transform=uses_log_transform,
            )

            selected_models.append(
                {
                    "target_variable": target_variable,
                    "target_columns": grouped_target_columns,
                    "selected_model_name": best_model_name,
                    "validation_rmse_mean": best_validation_rmse_mean,
                    "candidate_validation_rmse_mean": candidate_validation_scores,
                    "target_transform": "log1p" if uses_log_transform else "none",
                    "model_path": str(model_path),
                }
            )

            for target_index, target_column in enumerate(grouped_target_columns):
                target_metadata = self.parse_target_column(target_column)
                target_test_metrics = RegressionMetricCalculator.compute(
                    y_true=y_test[:, target_index],
                    y_pred=test_predictions[:, target_index],
                )

                test_records.append(
                    {
                        "target_column": target_column,
                        "target_variable": target_variable,
                        "horizon_hours": target_metadata["horizon_hours"],
                        "selected_model_name": best_model_name,
                        "target_transform": "log1p" if uses_log_transform else "none",
                        "model_path": str(model_path),
                        **target_test_metrics.to_dict(),
                    }
                )

                for timestamp, actual_value, predicted_value in zip(
                    test_frame.index,
                    y_test[:, target_index],
                    test_predictions[:, target_index],
                ):
                    prediction_records.append(
                        {
                            "date": timestamp,
                            "target_column": target_column,
                            "target_variable": target_variable,
                            "horizon_hours": target_metadata["horizon_hours"],
                            "selected_model_name": best_model_name,
                            "actual": float(actual_value),
                            "prediction": float(predicted_value),
                            "residual": float(actual_value - predicted_value),
                        }
                    )

        validation_metrics_frame = pd.DataFrame(validation_records).sort_values(
            by=["target_variable", "horizon_hours", "rmse"]
        )
        test_metrics_frame = pd.DataFrame(test_records).sort_values(
            by=["target_variable", "horizon_hours"]
        )
        prediction_frame = pd.DataFrame(prediction_records)

        summary = self.build_summary(
            feature_columns=baseline_feature_columns,
            all_engineered_feature_count=len(feature_columns),
            target_columns=target_columns,
            candidate_model_names=list(candidate_builders.keys()),
            selected_models=selected_models,
            split_indices=split_indices,
            dataframe=dataframe,
        )

        self.save_outputs(
            validation_metrics_frame=validation_metrics_frame,
            test_metrics_frame=test_metrics_frame,
            prediction_frame=prediction_frame,
            summary=summary,
        )

        return validation_metrics_frame, test_metrics_frame, summary

    def _ensure_output_directories(self) -> None:
        """Create folders needed for model artifacts and reports."""
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.validation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.predictions_path.parent.mkdir(parents=True, exist_ok=True)

    def load_phase_two_artifacts(self) -> tuple[pd.DataFrame, list[str], list[str], dict]:
        """Load the Phase 2 dataset along with exact feature metadata and split indices."""
        dataframe = pd.read_csv(self.config.dataset_path, parse_dates=["date"])
        dataframe = dataframe.sort_values("date").set_index("date")

        with self.config.phase_two_artifact_path.open("rb") as file_pointer:
            artifact = pickle.load(file_pointer)

        feature_columns = artifact["feature_columns"]
        target_columns = artifact["target_columns"]
        split_indices = artifact["split_indices"]

        required_columns = set(feature_columns + target_columns)
        missing_columns = sorted(required_columns.difference(dataframe.columns))
        if missing_columns:
            raise ValueError(f"Phase 2 dataset is missing required columns: {missing_columns}")

        return dataframe, feature_columns, target_columns, split_indices

    def build_target_groups(self, target_columns: list[str]) -> dict[str, list[str]]:
        """Group direct forecast targets by their underlying meteorological variable."""
        grouped_columns: dict[str, list[tuple[int, str]]] = {}

        for target_column in target_columns:
            target_metadata = self.parse_target_column(target_column)
            grouped_columns.setdefault(target_metadata["target_variable"], []).append(
                (target_metadata["horizon_hours"], target_column)
            )

        return {
            target_variable: [
                target_column for _, target_column in sorted(group_entries, key=lambda item: item[0])
            ]
            for target_variable, group_entries in grouped_columns.items()
        }

    def select_baseline_feature_columns(self, feature_columns: list[str]) -> list[str]:
        """Keep a compact, expert-informed feature subset for classical baselines."""
        direct_features = {
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_gusts_10m",
            "wind_direction_10m_sin",
            "wind_direction_10m_cos",
            "wind_direction_100m_sin",
            "wind_direction_100m_cos",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "is_weekend",
        }
        lag_bases = {
            "temperature_2m",
            "precipitation",
            "relative_humidity_2m",
            "pressure_msl",
            "cloud_cover",
            "wind_speed_10m",
        }
        lag_hours = {"1h", "6h", "24h", "72h"}
        rolling_bases = {
            "temperature_2m",
            "precipitation",
            "relative_humidity_2m",
            "pressure_msl",
            "cloud_cover",
            "wind_speed_10m",
        }
        rolling_windows = {"24", "72"}

        selected_columns: list[str] = []
        for column in feature_columns:
            if column in direct_features:
                selected_columns.append(column)
                continue

            if "_lag_" in column:
                base_name, lag_hour = column.rsplit("_lag_", maxsplit=1)
                if base_name in lag_bases and lag_hour in lag_hours:
                    selected_columns.append(column)
                continue

            rolling_match = self.ROLLING_PATTERN.fullmatch(column)
            if rolling_match is not None:
                if (
                    rolling_match.group("base") in rolling_bases
                    and rolling_match.group("window") in rolling_windows
                ):
                    selected_columns.append(column)

        return selected_columns

    def get_candidate_model_builders(self) -> dict[str, callable]:
        """Define candidate baseline regressors used for model selection."""
        return {
            "persistence": (
                lambda target_variable, output_count, uses_log_transform: PersistenceRegressor(
                    source_column=target_variable,
                    output_count=output_count,
                    uses_log_transform=uses_log_transform,
                )
            ),
            "random_forest": (
                lambda target_variable, output_count, uses_log_transform: RandomForestRegressor(
                    n_estimators=40,
                    max_depth=12,
                    min_samples_leaf=4,
                    max_features=0.5,
                    max_samples=0.7,
                    n_jobs=1,
                    random_state=self.config.random_state,
                )
            ),
        }

    @classmethod
    def parse_target_column(cls, target_column: str) -> dict[str, str | int]:
        """Extract the base variable and forecast horizon from the target column name."""
        match = cls.TARGET_PATTERN.fullmatch(target_column)
        if match is None:
            raise ValueError(f"Unrecognized target column format: {target_column}")

        return {
            "target_variable": match.group("variable"),
            "horizon_hours": int(match.group("horizon")),
        }

    @staticmethod
    def fit_model(
        model,
        features: pd.DataFrame,
        target: np.ndarray,
        uses_log_transform: bool,
    ) -> None:
        """Fit a regressor, optionally using a log-transform for precipitation targets."""
        if uses_log_transform:
            model.fit(features, np.log1p(target))
            return

        model.fit(features, target)

    @staticmethod
    def predict_with_model(
        model,
        features: pd.DataFrame,
        uses_log_transform: bool,
    ) -> np.ndarray:
        """Generate predictions and invert the precipitation transform when needed."""
        predictions = model.predict(features)

        if uses_log_transform:
            predictions = np.expm1(predictions)
            predictions = np.clip(predictions, a_min=0.0, a_max=None)

        return predictions.astype(np.float32)

    def save_model_artifact(
        self,
        model,
        model_name: str,
        target_variable: str,
        target_columns: list[str],
        feature_columns: list[str],
        uses_log_transform: bool,
    ) -> Path:
        """Persist the final selected model and its metadata."""
        model_path = self.config.model_dir / f"{model_name}__{target_variable}_multi_horizon.joblib"

        artifact = {
            "model": model,
            "model_name": model_name,
            "target_variable": target_variable,
            "target_columns": target_columns,
            "feature_columns": feature_columns,
            "target_transform": "log1p" if uses_log_transform else "none",
        }
        joblib.dump(artifact, model_path)
        return model_path

    def build_summary(
        self,
        feature_columns: list[str],
        all_engineered_feature_count: int,
        target_columns: list[str],
        candidate_model_names: list[str],
        selected_models: list[dict],
        split_indices: dict,
        dataframe: pd.DataFrame,
    ) -> dict:
        """Create a JSON summary describing the baseline modeling experiment."""
        train_frame = dataframe.iloc[split_indices["train_start"] : split_indices["train_end"]]
        validation_frame = dataframe.iloc[
            split_indices["validation_start"] : split_indices["validation_end"]
        ]
        test_frame = dataframe.iloc[split_indices["test_start"] : split_indices["test_end"]]

        return {
            "input_dataset": str(self.config.dataset_path),
            "phase_two_artifact": str(self.config.phase_two_artifact_path),
            "candidate_models": candidate_model_names,
            "selection_rule": (
                "Best model per target variable is chosen by the lowest mean validation RMSE "
                "across the 24h, 48h, and 72h direct forecasts."
            ),
            "all_engineered_feature_count": int(all_engineered_feature_count),
            "baseline_feature_count": int(len(feature_columns)),
            "baseline_feature_columns": feature_columns,
            "target_count": int(len(target_columns)),
            "splits": {
                "train": {
                    "row_count": int(len(train_frame)),
                    "start": str(train_frame.index.min()),
                    "end": str(train_frame.index.max()),
                },
                "validation": {
                    "row_count": int(len(validation_frame)),
                    "start": str(validation_frame.index.min()),
                    "end": str(validation_frame.index.max()),
                },
                "test": {
                    "row_count": int(len(test_frame)),
                    "start": str(test_frame.index.min()),
                    "end": str(test_frame.index.max()),
                },
            },
            "selected_models": selected_models,
        }

    def save_outputs(
        self,
        validation_metrics_frame: pd.DataFrame,
        test_metrics_frame: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        summary: dict,
    ) -> None:
        """Save metrics, predictions, and the experiment summary."""
        validation_metrics_frame.to_csv(self.config.validation_metrics_path, index=False)
        test_metrics_frame.to_csv(self.config.test_metrics_path, index=False)
        prediction_frame.to_csv(self.config.predictions_path, index=False, compression="gzip")

        with self.config.summary_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(summary, file_pointer, indent=4)


def main() -> None:
    """Run Phase 3 baseline training using the project defaults."""
    config = PhaseThreeConfig()
    trainer = ClimateBaselineTrainer(config)
    validation_metrics_frame, test_metrics_frame, summary = trainer.run()

    print("Phase 3 completed successfully.")
    print(f"Validation metrics saved to: {config.validation_metrics_path}")
    print(f"Test metrics saved to: {config.test_metrics_path}")
    print(f"Test predictions saved to: {config.predictions_path}")
    print(f"Model selection summary saved to: {config.summary_path}")
    print(f"Models saved under: {config.model_dir}")
    print(f"Validation experiments: {len(validation_metrics_frame)}")
    print(f"Final test models: {len(test_metrics_frame)}")
    print(f"Selected baseline models: {len(summary['selected_models'])}")


if __name__ == "__main__":
    main()
