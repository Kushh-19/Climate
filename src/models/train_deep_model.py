"""Phase 4 pipeline for PyTorch-based LSTM and GRU forecasting models.

This module trains recurrent neural networks for direct multi-horizon weather
forecasting. The implementation is intentionally limited to Phase 4:
1. Build sequence datasets from the Phase 2 scaled features.
2. Train candidate LSTM and GRU architectures.
3. Select the best architecture per target variable using validation RMSE.
4. Retrain the selected architecture on train plus validation.
5. Evaluate once on the held-out test partition.
6. Save checkpoints, predictions, and experiment summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from src.models.evaluate_model import RegressionMetricCalculator
    from src.models.predict import RecurrentForecaster
except ModuleNotFoundError:
    from evaluate_model import RegressionMetricCalculator
    from predict import RecurrentForecaster


@dataclass(frozen=True)
class PhaseFourConfig:
    """Configuration for Phase 4 recurrent model training."""

    dataset_path: Path = Path("data/processed/phase2/baroda_supervised_scaled.csv.gz")
    phase_two_summary_path: Path = Path("reports/phase2/feature_engineering_summary.json")
    baseline_metrics_path: Path = Path("reports/phase3/baseline_test_metrics.csv")
    model_dir: Path = Path("models/phase4/deep_learning")
    validation_metrics_path: Path = Path("reports/phase4/deep_validation_metrics.csv")
    test_metrics_path: Path = Path("reports/phase4/deep_test_metrics.csv")
    summary_path: Path = Path("reports/phase4/deep_selection_summary.json")
    predictions_path: Path = Path("data/processed/phase4/deep_test_predictions.csv.gz")
    sequence_length: int = 72
    batch_size: int = 1024
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 6
    patience: int = 2
    wet_event_threshold_mm: float = 0.1
    wet_event_weight: float = 3.0
    random_state: int = 42


class TargetTransform:
    """Handles target transforms and precipitation-aware loss weighting."""

    def __init__(
        self,
        target_variable: str,
        wet_event_threshold_mm: float,
        wet_event_weight: float,
    ) -> None:
        self.target_variable = target_variable
        self.wet_event_threshold_mm = wet_event_threshold_mm
        self.wet_event_weight = wet_event_weight

    @property
    def transform_name(self) -> str:
        """Return the target transform name for reporting."""
        return "log1p" if self.target_variable == "precipitation" else "none"

    def forward_tensor(self, target_tensor: torch.Tensor) -> torch.Tensor:
        """Transform the training target into a numerically stable space."""
        if self.target_variable == "precipitation":
            return torch.log1p(target_tensor)
        return target_tensor

    def inverse_tensor(self, prediction_tensor: torch.Tensor) -> torch.Tensor:
        """Map predictions back into the original physical units."""
        if self.target_variable == "precipitation":
            return torch.clamp(torch.expm1(prediction_tensor), min=0.0)
        return prediction_tensor

    def loss_weights(self, raw_target_tensor: torch.Tensor) -> torch.Tensor:
        """Increase the loss contribution of rainy samples to address imbalance."""
        if self.target_variable != "precipitation":
            return torch.ones_like(raw_target_tensor)

        wet_event_mask = raw_target_tensor > self.wet_event_threshold_mm
        return torch.where(
            wet_event_mask,
            torch.full_like(raw_target_tensor, self.wet_event_weight),
            torch.ones_like(raw_target_tensor),
        )


class WeightedMSELoss(nn.Module):
    """Weighted mean squared error used for multi-horizon regression."""

    def forward(
        self,
        prediction_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a weighted squared-error loss across all horizons."""
        return torch.mean(weight_tensor * torch.square(prediction_tensor - target_tensor))


class TimeSeriesWindowDataset(Dataset):
    """Creates fixed-length input windows ending at specified forecast issue times."""

    def __init__(
        self,
        feature_array: np.ndarray,
        target_array: np.ndarray,
        sequence_end_indices: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
        self.target_tensor = torch.tensor(target_array, dtype=torch.float32)
        self.sequence_end_indices = sequence_end_indices.astype(np.int64)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """Return the number of valid sequence endpoints."""
        return len(self.sequence_end_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sequence window and its multi-horizon regression target."""
        end_index = int(self.sequence_end_indices[index])
        start_index = end_index - self.sequence_length + 1
        sequence_tensor = self.feature_tensor[start_index : end_index + 1]
        target_tensor = self.target_tensor[end_index]
        return sequence_tensor, target_tensor


class ClimateDeepLearningTrainer:
    """Train and select PyTorch recurrent models for climate forecasting."""

    TARGET_PATTERN = re.compile(r"(?P<variable>.+)_target_t_plus_(?P<horizon>\d+)h")

    def __init__(self, config: PhaseFourConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = WeightedMSELoss()
        self._set_random_seed()
        torch.set_num_threads(4)

    def _set_random_seed(self) -> None:
        """Seed NumPy and PyTorch for reproducible experiments."""
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Execute the full Phase 4 recurrent-model experiment."""
        self._ensure_output_directories()

        dataframe, feature_columns, target_columns, split_indices = self.load_phase_two_scaled_data()
        sequence_feature_columns = self.select_sequence_feature_columns(feature_columns)
        target_groups = self.build_target_groups(target_columns)
        split_end_indices = self.build_sequence_end_indices(split_indices)

        feature_array = dataframe[sequence_feature_columns].to_numpy(dtype=np.float32)

        validation_records: list[dict] = []
        test_records: list[dict] = []
        prediction_records: list[dict] = []
        selected_models: list[dict] = []

        for target_variable, grouped_target_columns in target_groups.items():
            target_array = dataframe[grouped_target_columns].to_numpy(dtype=np.float32)
            target_transform = TargetTransform(
                target_variable=target_variable,
                wet_event_threshold_mm=self.config.wet_event_threshold_mm,
                wet_event_weight=self.config.wet_event_weight,
            )

            dataloaders = self.create_dataloaders(
                feature_array=feature_array,
                target_array=target_array,
                split_end_indices=split_end_indices,
            )

            candidate_results: dict[str, dict] = {}
            for recurrent_type in ("lstm", "gru"):
                candidate_result = self.train_candidate_model(
                    recurrent_type=recurrent_type,
                    train_loader=dataloaders["train"],
                    validation_loader=dataloaders["validation"],
                    target_transform=target_transform,
                    input_size=len(sequence_feature_columns),
                    output_size=len(grouped_target_columns),
                )

                validation_predictions, validation_targets = self.collect_predictions(
                    model=candidate_result["model"],
                    data_loader=dataloaders["validation"],
                    target_transform=target_transform,
                )
                candidate_validation_records, validation_rmse_mean = self.compute_metric_records(
                    target_variable=target_variable,
                    target_columns=grouped_target_columns,
                    actual_array=validation_targets,
                    prediction_array=validation_predictions,
                    model_name=recurrent_type,
                    split_name="validation",
                    checkpoint_path=None,
                    target_transform_name=target_transform.transform_name,
                )

                validation_records.extend(candidate_validation_records)
                candidate_result["validation_rmse_mean"] = validation_rmse_mean
                candidate_results[recurrent_type] = candidate_result

            selected_recurrent_type = min(
                candidate_results,
                key=lambda recurrent_name: candidate_results[recurrent_name]["validation_rmse_mean"],
            )
            selected_candidate = candidate_results[selected_recurrent_type]

            final_model = self.fit_final_model(
                recurrent_type=selected_recurrent_type,
                train_validation_loader=dataloaders["train_validation"],
                target_transform=target_transform,
                input_size=len(sequence_feature_columns),
                output_size=len(grouped_target_columns),
                epochs=selected_candidate["best_epoch"],
            )

            model_path = self.save_model_checkpoint(
                model=final_model,
                recurrent_type=selected_recurrent_type,
                target_variable=target_variable,
                target_columns=grouped_target_columns,
                feature_columns=sequence_feature_columns,
                target_transform=target_transform,
            )

            test_predictions, test_targets = self.collect_predictions(
                model=final_model,
                data_loader=dataloaders["test"],
                target_transform=target_transform,
            )
            target_test_records, test_rmse_mean = self.compute_metric_records(
                target_variable=target_variable,
                target_columns=grouped_target_columns,
                actual_array=test_targets,
                prediction_array=test_predictions,
                model_name=selected_recurrent_type,
                split_name="test",
                checkpoint_path=model_path,
                target_transform_name=target_transform.transform_name,
            )
            test_records.extend(target_test_records)

            selected_models.append(
                {
                    "target_variable": target_variable,
                    "target_columns": grouped_target_columns,
                    "selected_model_name": selected_recurrent_type,
                    "best_epoch": int(selected_candidate["best_epoch"]),
                    "validation_rmse_mean": float(selected_candidate["validation_rmse_mean"]),
                    "candidate_validation_rmse_mean": {
                        recurrent_name: float(result["validation_rmse_mean"])
                        for recurrent_name, result in candidate_results.items()
                    },
                    "test_rmse_mean": float(test_rmse_mean),
                    "target_transform": target_transform.transform_name,
                    "checkpoint_path": str(model_path),
                }
            )

            prediction_end_indices = split_end_indices["test"]
            prediction_dates = dataframe.index[prediction_end_indices]
            for target_index, target_column in enumerate(grouped_target_columns):
                target_metadata = self.parse_target_column(target_column)
                for timestamp, actual_value, predicted_value in zip(
                    prediction_dates,
                    test_targets[:, target_index],
                    test_predictions[:, target_index],
                ):
                    prediction_records.append(
                        {
                            "date": timestamp,
                            "target_column": target_column,
                            "target_variable": target_variable,
                            "horizon_hours": target_metadata["horizon_hours"],
                            "selected_model_name": selected_recurrent_type,
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
            sequence_feature_columns=sequence_feature_columns,
            target_columns=target_columns,
            split_indices=split_indices,
            selected_models=selected_models,
            dataframe=dataframe,
        )
        self.save_outputs(validation_metrics_frame, test_metrics_frame, prediction_frame, summary)

        return validation_metrics_frame, test_metrics_frame, summary

    def _ensure_output_directories(self) -> None:
        """Create folders needed for checkpoints, reports, and prediction files."""
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.validation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.predictions_path.parent.mkdir(parents=True, exist_ok=True)

    def load_phase_two_scaled_data(self) -> tuple[pd.DataFrame, list[str], list[str], dict]:
        """Load the Phase 2 scaled dataset and reconstruct its split metadata from JSON."""
        dataframe = pd.read_csv(self.config.dataset_path, parse_dates=["date"])
        dataframe = dataframe.sort_values("date").set_index("date")

        with self.config.phase_two_summary_path.open("r", encoding="utf-8") as file_pointer:
            phase_two_summary = json.load(file_pointer)

        train_row_count = int(phase_two_summary["splits"]["train"]["row_count"])
        validation_row_count = int(phase_two_summary["splits"]["validation"]["row_count"])
        test_row_count = int(phase_two_summary["splits"]["test"]["row_count"])
        split_indices = {
            "train_start": 0,
            "train_end": train_row_count,
            "validation_start": train_row_count,
            "validation_end": train_row_count + validation_row_count,
            "test_start": train_row_count + validation_row_count,
            "test_end": train_row_count + validation_row_count + test_row_count,
        }

        return (
            dataframe,
            phase_two_summary["feature_columns"],
            phase_two_summary["target_columns"],
            split_indices,
        )

    @staticmethod
    def select_sequence_feature_columns(feature_columns: list[str]) -> list[str]:
        """Keep only contemporaneous meteorological and cyclical inputs for sequences."""
        return [
            column
            for column in feature_columns
            if "_lag_" not in column and "_rolling_" not in column
        ]

    def build_target_groups(self, target_columns: list[str]) -> dict[str, list[str]]:
        """Group multi-horizon targets by variable and sort them by forecast horizon."""
        grouped_targets: dict[str, list[tuple[int, str]]] = {}
        for target_column in target_columns:
            target_metadata = self.parse_target_column(target_column)
            grouped_targets.setdefault(target_metadata["target_variable"], []).append(
                (target_metadata["horizon_hours"], target_column)
            )

        return {
            target_variable: [
                target_column for _, target_column in sorted(group_entries, key=lambda item: item[0])
            ]
            for target_variable, group_entries in grouped_targets.items()
        }

    @classmethod
    def parse_target_column(cls, target_column: str) -> dict[str, str | int]:
        """Extract the target variable name and the horizon encoded in the column."""
        match = cls.TARGET_PATTERN.fullmatch(target_column)
        if match is None:
            raise ValueError(f"Unrecognized target column format: {target_column}")

        return {
            "target_variable": match.group("variable"),
            "horizon_hours": int(match.group("horizon")),
        }

    def build_sequence_end_indices(self, split_indices: dict) -> dict[str, np.ndarray]:
        """Create valid sequence endpoints for each chronological partition."""
        sequence_length = self.config.sequence_length
        return {
            "train": np.arange(sequence_length - 1, split_indices["train_end"]),
            "validation": np.arange(split_indices["validation_start"], split_indices["validation_end"]),
            "train_validation": np.arange(sequence_length - 1, split_indices["validation_end"]),
            "test": np.arange(split_indices["test_start"], split_indices["test_end"]),
        }

    def create_dataloaders(
        self,
        feature_array: np.ndarray,
        target_array: np.ndarray,
        split_end_indices: dict[str, np.ndarray],
    ) -> dict[str, DataLoader]:
        """Create PyTorch data loaders for training, selection, and testing."""
        datasets = {
            split_name: TimeSeriesWindowDataset(
                feature_array=feature_array,
                target_array=target_array,
                sequence_end_indices=end_indices,
                sequence_length=self.config.sequence_length,
            )
            for split_name, end_indices in split_end_indices.items()
        }

        return {
            "train": DataLoader(
                datasets["train"],
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
            ),
            "validation": DataLoader(
                datasets["validation"],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            ),
            "train_validation": DataLoader(
                datasets["train_validation"],
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            ),
        }

    def train_candidate_model(
        self,
        recurrent_type: str,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        target_transform: TargetTransform,
        input_size: int,
        output_size: int,
    ) -> dict:
        """Train one candidate recurrent architecture with early stopping."""
        model = RecurrentForecaster(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=output_size,
            recurrent_type=recurrent_type,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_validation_rmse = float("inf")
        best_epoch = 1
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, self.config.max_epochs + 1):
            self.train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                target_transform=target_transform,
            )

            validation_predictions, validation_targets = self.collect_predictions(
                model=model,
                data_loader=validation_loader,
                target_transform=target_transform,
            )
            validation_rmse_mean = self.compute_mean_rmse(
                actual_array=validation_targets,
                prediction_array=validation_predictions,
            )

            if validation_rmse_mean < best_validation_rmse:
                best_validation_rmse = validation_rmse_mean
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        model.load_state_dict(best_state_dict)
        return {
            "model": model,
            "best_epoch": best_epoch,
            "validation_rmse_mean": best_validation_rmse,
        }

    def train_one_epoch(
        self,
        model: RecurrentForecaster,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        target_transform: TargetTransform,
    ) -> None:
        """Run one optimization epoch for a recurrent model."""
        model.train()

        for feature_batch, raw_target_batch in data_loader:
            feature_batch = feature_batch.to(self.device)
            raw_target_batch = raw_target_batch.to(self.device)
            transformed_target_batch = target_transform.forward_tensor(raw_target_batch)
            weight_batch = target_transform.loss_weights(raw_target_batch)

            optimizer.zero_grad(set_to_none=True)
            prediction_batch = model(feature_batch)
            loss = self.loss_function(
                prediction_tensor=prediction_batch,
                target_tensor=transformed_target_batch,
                weight_tensor=weight_batch,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    def fit_final_model(
        self,
        recurrent_type: str,
        train_validation_loader: DataLoader,
        target_transform: TargetTransform,
        input_size: int,
        output_size: int,
        epochs: int,
    ) -> RecurrentForecaster:
        """Retrain the selected architecture on train plus validation for the chosen epoch count."""
        model = RecurrentForecaster(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=output_size,
            recurrent_type=recurrent_type,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        for _ in range(epochs):
            self.train_one_epoch(
                model=model,
                data_loader=train_validation_loader,
                optimizer=optimizer,
                target_transform=target_transform,
            )

        return model

    def collect_predictions(
        self,
        model: RecurrentForecaster,
        data_loader: DataLoader,
        target_transform: TargetTransform,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference and return predictions and ground truth in original units."""
        model.eval()
        prediction_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []

        with torch.no_grad():
            for feature_batch, raw_target_batch in data_loader:
                feature_batch = feature_batch.to(self.device)
                prediction_batch = model(feature_batch)
                prediction_batch = target_transform.inverse_tensor(prediction_batch)
                prediction_batches.append(prediction_batch.cpu().numpy())
                target_batches.append(raw_target_batch.numpy())

        return np.vstack(prediction_batches), np.vstack(target_batches)

    @staticmethod
    def compute_mean_rmse(actual_array: np.ndarray, prediction_array: np.ndarray) -> float:
        """Compute the mean RMSE across all forecast horizons."""
        rmse_values = []
        for target_index in range(actual_array.shape[1]):
            metrics = RegressionMetricCalculator.compute(
                y_true=actual_array[:, target_index],
                y_pred=prediction_array[:, target_index],
            )
            rmse_values.append(metrics.rmse)

        return float(np.mean(rmse_values))

    def compute_metric_records(
        self,
        target_variable: str,
        target_columns: list[str],
        actual_array: np.ndarray,
        prediction_array: np.ndarray,
        model_name: str,
        split_name: str,
        checkpoint_path: Path | None,
        target_transform_name: str,
    ) -> tuple[list[dict], float]:
        """Create per-horizon metric records and the mean RMSE summary."""
        metric_records: list[dict] = []
        rmse_values: list[float] = []

        for target_index, target_column in enumerate(target_columns):
            target_metadata = self.parse_target_column(target_column)
            metrics = RegressionMetricCalculator.compute(
                y_true=actual_array[:, target_index],
                y_pred=prediction_array[:, target_index],
            )
            metric_records.append(
                {
                    "target_column": target_column,
                    "target_variable": target_variable,
                    "horizon_hours": target_metadata["horizon_hours"],
                    "model_name": model_name,
                    "split_name": split_name,
                    "target_transform": target_transform_name,
                    "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
                    **metrics.to_dict(),
                }
            )
            rmse_values.append(metrics.rmse)

        return metric_records, float(np.mean(rmse_values))

    def save_model_checkpoint(
        self,
        model: RecurrentForecaster,
        recurrent_type: str,
        target_variable: str,
        target_columns: list[str],
        feature_columns: list[str],
        target_transform: TargetTransform,
    ) -> Path:
        """Persist a trained recurrent model along with its reconstruction metadata."""
        checkpoint_path = self.config.model_dir / f"{recurrent_type}__{target_variable}_multi_horizon.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "recurrent_type": recurrent_type,
                "input_size": len(feature_columns),
                "hidden_size": self.config.hidden_size,
                "output_size": len(target_columns),
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "sequence_length": self.config.sequence_length,
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "target_variable": target_variable,
                "target_transform": target_transform.transform_name,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def build_summary(
        self,
        sequence_feature_columns: list[str],
        target_columns: list[str],
        split_indices: dict,
        selected_models: list[dict],
        dataframe: pd.DataFrame,
    ) -> dict:
        """Build a JSON summary describing the Phase 4 deep-learning experiment."""
        baseline_reference = None
        if self.config.baseline_metrics_path.exists():
            baseline_reference = str(self.config.baseline_metrics_path)

        train_frame = dataframe.iloc[split_indices["train_start"] : split_indices["train_end"]]
        validation_frame = dataframe.iloc[
            split_indices["validation_start"] : split_indices["validation_end"]
        ]
        test_frame = dataframe.iloc[split_indices["test_start"] : split_indices["test_end"]]

        return {
            "input_dataset": str(self.config.dataset_path),
            "phase_two_summary": str(self.config.phase_two_summary_path),
            "baseline_reference_metrics": baseline_reference,
            "candidate_models": ["lstm", "gru"],
            "selection_rule": (
                "Best recurrent architecture per target variable is chosen by the lowest mean "
                "validation RMSE across the 24h, 48h, and 72h direct forecasts."
            ),
            "device": str(self.device),
            "sequence_length": int(self.config.sequence_length),
            "sequence_feature_count": int(len(sequence_feature_columns)),
            "sequence_feature_columns": sequence_feature_columns,
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
        """Save deep-learning metrics, predictions, and summary metadata."""
        validation_metrics_frame.to_csv(self.config.validation_metrics_path, index=False)
        test_metrics_frame.to_csv(self.config.test_metrics_path, index=False)
        prediction_frame.to_csv(self.config.predictions_path, index=False, compression="gzip")

        with self.config.summary_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(summary, file_pointer, indent=4)


def main() -> None:
    """Run Phase 4 recurrent-model training with the default project configuration."""
    config = PhaseFourConfig()
    trainer = ClimateDeepLearningTrainer(config)
    validation_metrics_frame, test_metrics_frame, summary = trainer.run()

    print("Recurrent-model training completed successfully.")
    print(f"Validation metrics saved to: {config.validation_metrics_path}")
    print(f"Test metrics saved to: {config.test_metrics_path}")
    print(f"Test predictions saved to: {config.predictions_path}")
    print(f"Selection summary saved to: {config.summary_path}")
    print(f"Model checkpoints saved under: {config.model_dir}")
    print(f"Validation experiments: {len(validation_metrics_frame)}")
    print(f"Final test results: {len(test_metrics_frame)}")
    print(f"Selected recurrent models: {len(summary['selected_models'])}")


if __name__ == "__main__":
    main()
