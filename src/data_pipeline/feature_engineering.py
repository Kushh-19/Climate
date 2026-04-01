"""Phase 2 pipeline for time-series feature engineering.

This module converts the cleaned hourly weather data into a supervised
learning dataset suitable for forecasting 24, 48, and 72 hours ahead.
It creates:
1. Calendar-based cyclical features.
2. Circular encodings for wind direction.
3. Lagged and rolling-window time-series predictors.
4. Multi-horizon forecasting targets.
5. Chronological train/validation/test splits.
6. A feature scaler fit only on the training partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PhaseTwoConfig:
    """Configuration for the Phase 2 feature engineering pipeline."""

    input_data_path: Path = Path("data/interim/Baroda_phase1_clean.csv")
    output_dir: Path = Path("data/processed/phase2")
    unscaled_output_path: Path = Path("data/processed/phase2/baroda_supervised_unscaled.csv.gz")
    scaled_output_path: Path = Path("data/processed/phase2/baroda_supervised_scaled.csv.gz")
    report_path: Path = Path("reports/phase2/feature_engineering_summary.json")
    scaler_artifact_path: Path = Path("models/artifacts/phase2_feature_scaler.pkl")
    target_columns: tuple[str, ...] = ("temperature_2m", "precipitation")
    horizons: tuple[int, ...] = (24, 48, 72)
    lag_steps: tuple[int, ...] = (1, 3, 6, 12, 24, 48, 72)
    rolling_windows: tuple[int, ...] = (6, 24, 72)
    wind_direction_columns: tuple[str, ...] = ("wind_direction_10m", "wind_direction_100m")
    rolling_stat_columns: tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "cloud_cover",
        "wind_speed_10m",
    )
    rolling_sum_columns: tuple[str, ...] = ("precipitation", "rain")
    train_fraction: float = 0.70
    validation_fraction: float = 0.15


class ClimatePhaseTwoFeatureEngineer:
    """Builds leakage-safe forecasting features from the cleaned climate data."""

    def __init__(self, config: PhaseTwoConfig) -> None:
        self.config = config

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Execute the complete Phase 2 feature engineering workflow."""
        self._ensure_output_directories()

        dataframe = self.load_phase_one_dataset()
        dataframe = self.encode_wind_direction(dataframe)

        dynamic_feature_columns = dataframe.columns.tolist()
        calendar_features = self.create_calendar_features(dataframe.index)
        lag_features = self.create_lag_features(dataframe[dynamic_feature_columns])
        rolling_features = self.create_rolling_features(dataframe)
        future_targets = self.create_future_targets(dataframe)

        supervised_dataframe = pd.concat(
            [dataframe, calendar_features, lag_features, rolling_features, future_targets],
            axis=1,
        )

        target_columns = future_targets.columns.tolist()
        feature_columns = [
            column for column in supervised_dataframe.columns if column not in target_columns
        ]

        rows_before_dropna = len(supervised_dataframe)
        supervised_dataframe = supervised_dataframe.dropna(subset=feature_columns + target_columns)
        supervised_dataframe = supervised_dataframe.astype(np.float32)
        rows_removed = rows_before_dropna - len(supervised_dataframe)

        split_indices = self.create_chronological_splits(supervised_dataframe)
        scaled_dataframe, scaler = self.scale_features(
            supervised_dataframe=supervised_dataframe,
            feature_columns=feature_columns,
            split_indices=split_indices,
        )

        report = self.build_feature_report(
            original_rows=len(dataframe),
            supervised_dataframe=supervised_dataframe,
            feature_columns=feature_columns,
            target_columns=target_columns,
            split_indices=split_indices,
            rows_removed=rows_removed,
        )

        self.save_datasets(supervised_dataframe, scaled_dataframe)
        self.save_scaler_artifact(
            scaler=scaler,
            feature_columns=feature_columns,
            target_columns=target_columns,
            split_indices=split_indices,
        )
        self.save_report(report)

        return supervised_dataframe, scaled_dataframe, report

    def _ensure_output_directories(self) -> None:
        """Create output folders required by the Phase 2 pipeline."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.scaler_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    def load_phase_one_dataset(self) -> pd.DataFrame:
        """Load the cleaned Phase 1 data and validate its hourly structure."""
        dataframe = pd.read_csv(self.config.input_data_path, parse_dates=["date"])
        dataframe = dataframe.sort_values("date").set_index("date")

        if dataframe.index.has_duplicates:
            duplicate_count = int(dataframe.index.duplicated().sum())
            raise ValueError(f"Found {duplicate_count} duplicate timestamps in Phase 1 data.")

        timestamp_deltas = dataframe.index.to_series().diff().dropna()
        expected_delta = pd.Timedelta(hours=1)
        if not (timestamp_deltas == expected_delta).all():
            raise ValueError("Phase 1 dataset is not strictly hourly. Re-run Phase 1 before Phase 2.")

        return dataframe

    def encode_wind_direction(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert angular wind directions into sine and cosine components."""
        transformed_dataframe = dataframe.copy()

        for column in self.config.wind_direction_columns:
            radians = np.deg2rad(transformed_dataframe[column])
            transformed_dataframe[f"{column}_sin"] = np.sin(radians)
            transformed_dataframe[f"{column}_cos"] = np.cos(radians)

        transformed_dataframe = transformed_dataframe.drop(
            columns=list(self.config.wind_direction_columns)
        )
        return transformed_dataframe

    @staticmethod
    def create_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create cyclical time features from the hourly datetime index."""
        calendar_frame = pd.DataFrame(index=index)

        hour = index.hour
        day_of_week = index.dayofweek
        month = index.month - 1
        day_of_year = index.dayofyear - 1

        calendar_frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        calendar_frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        calendar_frame["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        calendar_frame["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        calendar_frame["month_sin"] = np.sin(2 * np.pi * month / 12)
        calendar_frame["month_cos"] = np.cos(2 * np.pi * month / 12)
        calendar_frame["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        calendar_frame["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        calendar_frame["is_weekend"] = index.dayofweek.isin([5, 6]).astype(np.int8)

        return calendar_frame

    def create_lag_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create autoregressive lag features from past observations only."""
        lagged_features: dict[str, pd.Series] = {}

        for column in dataframe.columns:
            for lag_step in self.config.lag_steps:
                feature_name = f"{column}_lag_{lag_step}h"
                lagged_features[feature_name] = dataframe[column].shift(lag_step)

        return pd.DataFrame(lagged_features, index=dataframe.index)

    def create_rolling_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create leakage-safe rolling statistics using only historical observations."""
        rolling_features: dict[str, pd.Series] = {}

        for column in self.config.rolling_stat_columns:
            shifted_series = dataframe[column].shift(1)
            for window in self.config.rolling_windows:
                rolling_features[f"{column}_rolling_mean_{window}h"] = shifted_series.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
                rolling_features[f"{column}_rolling_std_{window}h"] = shifted_series.rolling(
                    window=window,
                    min_periods=window,
                ).std()

        for column in self.config.rolling_sum_columns:
            shifted_series = dataframe[column].shift(1)
            for window in self.config.rolling_windows:
                rolling_features[f"{column}_rolling_sum_{window}h"] = shifted_series.rolling(
                    window=window,
                    min_periods=window,
                ).sum()

        return pd.DataFrame(rolling_features, index=dataframe.index)

    def create_future_targets(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create direct multi-horizon targets for 24, 48, and 72 hour forecasting."""
        target_frame: dict[str, pd.Series] = {}

        for target_column in self.config.target_columns:
            for horizon in self.config.horizons:
                target_name = f"{target_column}_target_t_plus_{horizon}h"
                target_frame[target_name] = dataframe[target_column].shift(-horizon)

        return pd.DataFrame(target_frame, index=dataframe.index)

    def create_chronological_splits(self, dataframe: pd.DataFrame) -> dict:
        """Define train, validation, and test partitions without temporal leakage."""
        total_rows = len(dataframe)
        train_end = int(total_rows * self.config.train_fraction)
        validation_end = train_end + int(total_rows * self.config.validation_fraction)

        return {
            "train_start": 0,
            "train_end": train_end,
            "validation_start": train_end,
            "validation_end": validation_end,
            "test_start": validation_end,
            "test_end": total_rows,
        }

    @staticmethod
    def scale_features(
        supervised_dataframe: pd.DataFrame,
        feature_columns: list[str],
        split_indices: dict,
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """Scale features using only the training partition, then transform all rows."""
        scaler = StandardScaler()

        train_slice = supervised_dataframe.iloc[
            split_indices["train_start"] : split_indices["train_end"]
        ]
        scaler.fit(train_slice[feature_columns])

        scaled_dataframe = supervised_dataframe.copy()
        scaled_values = scaler.transform(supervised_dataframe[feature_columns])
        scaled_dataframe.loc[:, feature_columns] = scaled_values.astype(np.float32)

        return scaled_dataframe, scaler

    def build_feature_report(
        self,
        original_rows: int,
        supervised_dataframe: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
        split_indices: dict,
        rows_removed: int,
    ) -> dict:
        """Build a JSON summary describing the engineered dataset."""
        train_frame = supervised_dataframe.iloc[
            split_indices["train_start"] : split_indices["train_end"]
        ]
        validation_frame = supervised_dataframe.iloc[
            split_indices["validation_start"] : split_indices["validation_end"]
        ]
        test_frame = supervised_dataframe.iloc[
            split_indices["test_start"] : split_indices["test_end"]
        ]

        report = {
            "input_dataset": str(self.config.input_data_path),
            "original_row_count": int(original_rows),
            "supervised_row_count": int(len(supervised_dataframe)),
            "rows_removed_due_to_lags_and_horizons": int(rows_removed),
            "feature_count": int(len(feature_columns)),
            "target_count": int(len(target_columns)),
            "forecast_horizons_hours": list(self.config.horizons),
            "lag_steps_hours": list(self.config.lag_steps),
            "rolling_windows_hours": list(self.config.rolling_windows),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "date_range": {
                "start": str(supervised_dataframe.index.min()),
                "end": str(supervised_dataframe.index.max()),
            },
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
            "artifacts": {
                "unscaled_dataset": str(self.config.unscaled_output_path),
                "scaled_dataset": str(self.config.scaled_output_path),
                "scaler_artifact": str(self.config.scaler_artifact_path),
            },
        }
        return report

    def save_datasets(
        self,
        supervised_dataframe: pd.DataFrame,
        scaled_dataframe: pd.DataFrame,
    ) -> None:
        """Persist both the unscaled and scaled supervised datasets."""
        supervised_dataframe.reset_index().to_csv(
            self.config.unscaled_output_path,
            index=False,
            compression="gzip",
        )
        scaled_dataframe.reset_index().to_csv(
            self.config.scaled_output_path,
            index=False,
            compression="gzip",
        )

    def save_scaler_artifact(
        self,
        scaler: StandardScaler,
        feature_columns: list[str],
        target_columns: list[str],
        split_indices: dict,
    ) -> None:
        """Save the fitted scaler and column metadata for later model training."""
        artifact = {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "split_indices": split_indices,
        }
        with self.config.scaler_artifact_path.open("wb") as file_pointer:
            pickle.dump(artifact, file_pointer)

    def save_report(self, report: dict) -> None:
        """Persist the Phase 2 metadata report as formatted JSON."""
        with self.config.report_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(report, file_pointer, indent=4)


def main() -> None:
    """Run the Phase 2 feature engineering workflow using project defaults."""
    config = PhaseTwoConfig()
    feature_engineer = ClimatePhaseTwoFeatureEngineer(config)
    supervised_dataframe, scaled_dataframe, report = feature_engineer.run()

    print("Feature engineering completed successfully.")
    print(f"Unscaled supervised dataset saved to: {config.unscaled_output_path}")
    print(f"Scaled supervised dataset saved to: {config.scaled_output_path}")
    print(f"Scaler artifact saved to: {config.scaler_artifact_path}")
    print(f"Report saved to: {config.report_path}")
    print(f"Final engineered shape: {supervised_dataframe.shape}")
    print(f"Feature count: {report['feature_count']}")
    print(f"Target count: {report['target_count']}")
    print(
        "Train/Validation/Test rows: "
        f"{report['splits']['train']['row_count']}/"
        f"{report['splits']['validation']['row_count']}/"
        f"{report['splits']['test']['row_count']}"
    )
    print(f"Scaled dataframe shape: {scaled_dataframe.shape}")


if __name__ == "__main__":
    main()
