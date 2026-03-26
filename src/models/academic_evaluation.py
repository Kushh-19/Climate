"""Phase 5 pipeline for academic evaluation and visualization.

This module compares the classical baseline models from Phase 3 against the
PyTorch recurrent models from Phase 4. It focuses on:
1. Regression metrics in physical units.
2. Extreme-event detection via thresholded F1-score.
3. Comparative visualizations for thesis-quality reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from src.models.evaluate_model import ExtremeEventMetricCalculator, RegressionMetricCalculator
except ModuleNotFoundError:
    from evaluate_model import ExtremeEventMetricCalculator, RegressionMetricCalculator


@dataclass(frozen=True)
class PhaseFiveConfig:
    """Configuration for Phase 5 academic evaluation."""

    phase_two_dataset_path: Path = Path("data/processed/phase2/baroda_supervised_unscaled.csv.gz")
    phase_two_summary_path: Path = Path("reports/phase2/feature_engineering_summary.json")
    baseline_predictions_path: Path = Path("data/processed/phase3/baseline_test_predictions.csv.gz")
    deep_predictions_path: Path = Path("data/processed/phase4/deep_test_predictions.csv.gz")
    regression_output_path: Path = Path("reports/phase5/regression_comparison.csv")
    extreme_output_path: Path = Path("reports/phase5/extreme_event_metrics.csv")
    thresholds_output_path: Path = Path("reports/phase5/extreme_thresholds.csv")
    summary_output_path: Path = Path("reports/phase5/academic_evaluation_summary.json")
    figures_dir: Path = Path("reports/phase5/figures")
    temperature_extreme_quantile: float = 0.95
    precipitation_positive_extreme_quantile: float = 0.95
    snapshot_horizon_hours: int = 24
    snapshot_steps: int = 7 * 24


class AcademicClimateEvaluator:
    """Runs the final academic comparison between baseline and deep models."""

    TARGET_PATTERN = re.compile(r"(?P<variable>.+)_target_t_plus_(?P<horizon>\d+)h")

    def __init__(self, config: PhaseFiveConfig) -> None:
        self.config = config
        sns.set_theme(style="whitegrid", context="talk")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Execute the full Phase 5 evaluation workflow."""
        self._ensure_output_directories()

        thresholds_frame = self.compute_extreme_thresholds()
        prediction_frame = self.load_prediction_artifacts()
        regression_frame = self.compute_regression_metrics(prediction_frame)
        extreme_frame = self.compute_extreme_metrics(prediction_frame, thresholds_frame)
        summary = self.build_summary(regression_frame, extreme_frame, thresholds_frame)

        self.save_outputs(regression_frame, extreme_frame, thresholds_frame, summary)
        self.generate_visualizations(prediction_frame, regression_frame, extreme_frame)

        return regression_frame, extreme_frame, summary

    def _ensure_output_directories(self) -> None:
        """Create folders used for Phase 5 reports and visualizations."""
        self.config.regression_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)

    def compute_extreme_thresholds(self) -> pd.DataFrame:
        """Derive leakage-safe extreme thresholds from the training partition only."""
        with self.config.phase_two_summary_path.open("r", encoding="utf-8") as file_pointer:
            phase_two_summary = json.load(file_pointer)

        target_columns = phase_two_summary["target_columns"]
        train_row_count = int(phase_two_summary["splits"]["train"]["row_count"])
        training_targets = pd.read_csv(
            self.config.phase_two_dataset_path,
            usecols=target_columns,
        ).iloc[:train_row_count]

        threshold_records: list[dict] = []
        for target_column in target_columns:
            target_metadata = self.parse_target_column(target_column)
            series = training_targets[target_column]

            if target_metadata["target_variable"] == "precipitation":
                positive_series = series[series > 0]
                if positive_series.empty:
                    threshold = float(series.quantile(self.config.precipitation_positive_extreme_quantile))
                    threshold_strategy = (
                        "Fallback to the full training distribution because no positive precipitation values were present."
                    )
                else:
                    threshold = float(
                        positive_series.quantile(self.config.precipitation_positive_extreme_quantile)
                    )
                    threshold_strategy = (
                        "95th percentile of positive training precipitation to avoid zero-inflation bias."
                    )
            else:
                threshold = float(series.quantile(self.config.temperature_extreme_quantile))
                threshold_strategy = "95th percentile of the full training temperature distribution."

            threshold_records.append(
                {
                    "target_column": target_column,
                    "target_variable": target_metadata["target_variable"],
                    "horizon_hours": target_metadata["horizon_hours"],
                    "extreme_threshold": threshold,
                    "threshold_strategy": threshold_strategy,
                    "training_extreme_support": int((series >= threshold).sum()),
                    "training_total_samples": int(len(series)),
                }
            )

        return pd.DataFrame(threshold_records).sort_values(
            by=["target_variable", "horizon_hours"]
        )

    def load_prediction_artifacts(self) -> pd.DataFrame:
        """Load baseline and deep-learning test predictions into a common schema."""
        baseline_predictions = pd.read_csv(
            self.config.baseline_predictions_path,
            parse_dates=["date"],
        )
        baseline_predictions["model_family"] = "baseline"
        baseline_predictions["model_name"] = baseline_predictions["selected_model_name"]

        deep_predictions = pd.read_csv(
            self.config.deep_predictions_path,
            parse_dates=["date"],
        )
        deep_predictions["model_family"] = "deep_learning"
        deep_predictions["model_name"] = deep_predictions["selected_model_name"]

        prediction_frame = pd.concat(
            [baseline_predictions, deep_predictions],
            ignore_index=True,
        )
        return prediction_frame.sort_values(
            by=["target_variable", "horizon_hours", "model_family", "date"]
        )

    def compute_regression_metrics(self, prediction_frame: pd.DataFrame) -> pd.DataFrame:
        """Compute regression metrics directly from the saved test predictions."""
        metric_records: list[dict] = []

        grouping_columns = [
            "model_family",
            "model_name",
            "target_column",
            "target_variable",
            "horizon_hours",
        ]
        for group_key, group_frame in prediction_frame.groupby(grouping_columns, sort=True):
            metrics = RegressionMetricCalculator.compute(
                y_true=group_frame["actual"].to_numpy(),
                y_pred=group_frame["prediction"].to_numpy(),
            )
            metric_records.append(
                {
                    "model_family": group_key[0],
                    "model_name": group_key[1],
                    "target_column": group_key[2],
                    "target_variable": group_key[3],
                    "horizon_hours": group_key[4],
                    "sample_count": int(len(group_frame)),
                    **metrics.to_dict(),
                }
            )

        return pd.DataFrame(metric_records).sort_values(
            by=["target_variable", "horizon_hours", "rmse"]
        )

    def compute_extreme_metrics(
        self,
        prediction_frame: pd.DataFrame,
        thresholds_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluate each model as an extreme-event detector via thresholded predictions."""
        merged_frame = prediction_frame.merge(
            thresholds_frame,
            on=["target_column", "target_variable", "horizon_hours"],
            how="left",
        )

        metric_records: list[dict] = []
        grouping_columns = [
            "model_family",
            "model_name",
            "target_column",
            "target_variable",
            "horizon_hours",
        ]
        for group_key, group_frame in merged_frame.groupby(grouping_columns, sort=True):
            threshold = float(group_frame["extreme_threshold"].iloc[0])
            actual_extreme = (group_frame["actual"] >= threshold).astype(int).to_numpy()
            predicted_extreme = (group_frame["prediction"] >= threshold).astype(int).to_numpy()

            metrics = ExtremeEventMetricCalculator.compute(
                y_true=actual_extreme,
                y_pred=predicted_extreme,
            )
            metric_records.append(
                {
                    "model_family": group_key[0],
                    "model_name": group_key[1],
                    "target_column": group_key[2],
                    "target_variable": group_key[3],
                    "horizon_hours": group_key[4],
                    "extreme_threshold": threshold,
                    "threshold_strategy": group_frame["threshold_strategy"].iloc[0],
                    "test_extreme_support": int(actual_extreme.sum()),
                    "test_predicted_extreme_support": int(predicted_extreme.sum()),
                    "extreme_prevalence": float(actual_extreme.mean()),
                    **metrics.to_dict(),
                }
            )

        return pd.DataFrame(metric_records).sort_values(
            by=["target_variable", "horizon_hours", "f1"],
            ascending=[True, True, False],
        )

    def build_summary(
        self,
        regression_frame: pd.DataFrame,
        extreme_frame: pd.DataFrame,
        thresholds_frame: pd.DataFrame,
    ) -> dict:
        """Create a concise academic summary of model ranking and extreme performance."""
        rmse_pivot = self._build_metric_pivot(regression_frame, metric_name="rmse")
        f1_pivot = self._build_metric_pivot(extreme_frame, metric_name="f1")

        best_regression_models = (
            regression_frame.sort_values(by=["target_column", "rmse"])
            .groupby("target_column", as_index=False)
            .first()
        )
        best_extreme_models = (
            extreme_frame.sort_values(by=["target_column", "f1"], ascending=[True, False])
            .groupby("target_column", as_index=False)
            .first()
        )

        summary = {
            "phase_two_summary": str(self.config.phase_two_summary_path),
            "baseline_predictions": str(self.config.baseline_predictions_path),
            "deep_predictions": str(self.config.deep_predictions_path),
            "extreme_thresholds": thresholds_frame.to_dict(orient="records"),
            "mean_regression_metrics_by_model_family": (
                regression_frame.groupby("model_family")[["rmse", "mae", "r2"]]
                .mean()
                .round(6)
                .to_dict(orient="index")
            ),
            "mean_extreme_metrics_by_model_family": (
                extreme_frame.groupby("model_family")[["precision", "recall", "f1"]]
                .mean()
                .round(6)
                .to_dict(orient="index")
            ),
            "rmse_comparison": rmse_pivot.to_dict(orient="records"),
            "f1_comparison": f1_pivot.to_dict(orient="records"),
            "best_regression_model_by_target": best_regression_models.to_dict(orient="records"),
            "best_extreme_model_by_target": best_extreme_models.to_dict(orient="records"),
        }
        return summary

    @staticmethod
    def _build_metric_pivot(metric_frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """Create a baseline-vs-deep comparison table for one metric."""
        pivot_frame = (
            metric_frame.pivot_table(
                index=["target_column", "target_variable", "horizon_hours"],
                columns="model_family",
                values=metric_name,
            )
            .reset_index()
            .rename_axis(columns=None)
        )

        if {"baseline", "deep_learning"}.issubset(pivot_frame.columns):
            pivot_frame[f"deep_minus_baseline_{metric_name}"] = (
                pivot_frame["deep_learning"] - pivot_frame["baseline"]
            )

        return pivot_frame.sort_values(by=["target_variable", "horizon_hours"])

    def save_outputs(
        self,
        regression_frame: pd.DataFrame,
        extreme_frame: pd.DataFrame,
        thresholds_frame: pd.DataFrame,
        summary: dict,
    ) -> None:
        """Persist Phase 5 tables and JSON summary."""
        regression_frame.to_csv(self.config.regression_output_path, index=False)
        extreme_frame.to_csv(self.config.extreme_output_path, index=False)
        thresholds_frame.to_csv(self.config.thresholds_output_path, index=False)

        with self.config.summary_output_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(summary, file_pointer, indent=4)

    def generate_visualizations(
        self,
        prediction_frame: pd.DataFrame,
        regression_frame: pd.DataFrame,
        extreme_frame: pd.DataFrame,
    ) -> None:
        """Create compact comparative figures for the final report."""
        self._plot_rmse_comparison(regression_frame)
        self._plot_extreme_f1_comparison(extreme_frame)
        self._plot_forecast_snapshot(prediction_frame, target_variable="temperature_2m")
        self._plot_forecast_snapshot(prediction_frame, target_variable="precipitation")

    def _plot_rmse_comparison(self, regression_frame: pd.DataFrame) -> None:
        """Plot RMSE comparison across forecast horizons for both model families."""
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        target_titles = {
            "temperature_2m": "Temperature RMSE Comparison",
            "precipitation": "Precipitation RMSE Comparison",
        }

        for axis, target_variable in zip(axes, ["temperature_2m", "precipitation"]):
            target_frame = regression_frame[regression_frame["target_variable"] == target_variable]
            sns.lineplot(
                data=target_frame,
                x="horizon_hours",
                y="rmse",
                hue="model_family",
                style="model_family",
                markers=True,
                dashes=False,
                palette={"baseline": "#264653", "deep_learning": "#e76f51"},
                ax=axis,
            )
            axis.set_title(target_titles[target_variable])
            axis.set_xlabel("Forecast Horizon (hours)")
            axis.set_ylabel("RMSE")

        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "rmse_comparison.png", dpi=300)
        plt.close(figure)

    def _plot_extreme_f1_comparison(self, extreme_frame: pd.DataFrame) -> None:
        """Plot F1-score comparison for thresholded extreme-event detection."""
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        target_titles = {
            "temperature_2m": "Extreme Heat F1 Comparison",
            "precipitation": "Extreme Precipitation F1 Comparison",
        }

        for axis, target_variable in zip(axes, ["temperature_2m", "precipitation"]):
            target_frame = extreme_frame[extreme_frame["target_variable"] == target_variable]
            sns.lineplot(
                data=target_frame,
                x="horizon_hours",
                y="f1",
                hue="model_family",
                style="model_family",
                markers=True,
                dashes=False,
                palette={"baseline": "#264653", "deep_learning": "#e76f51"},
                ax=axis,
            )
            axis.set_title(target_titles[target_variable])
            axis.set_xlabel("Forecast Horizon (hours)")
            axis.set_ylabel("F1-score")
            axis.set_ylim(0.0, 1.0)

        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "extreme_f1_comparison.png", dpi=300)
        plt.close(figure)

    def _plot_forecast_snapshot(self, prediction_frame: pd.DataFrame, target_variable: str) -> None:
        """Plot a short test-window forecast snapshot for the 24-hour horizon."""
        subset_frame = prediction_frame[
            (prediction_frame["target_variable"] == target_variable)
            & (prediction_frame["horizon_hours"] == self.config.snapshot_horizon_hours)
        ].copy()

        actual_frame = (
            subset_frame[["date", "actual"]]
            .drop_duplicates(subset="date")
            .sort_values("date")
            .tail(self.config.snapshot_steps)
        )
        baseline_frame = (
            subset_frame[subset_frame["model_family"] == "baseline"][["date", "prediction"]]
            .rename(columns={"prediction": "baseline_prediction"})
            .sort_values("date")
            .tail(self.config.snapshot_steps)
        )
        deep_frame = (
            subset_frame[subset_frame["model_family"] == "deep_learning"][["date", "prediction"]]
            .rename(columns={"prediction": "deep_prediction"})
            .sort_values("date")
            .tail(self.config.snapshot_steps)
        )

        plot_frame = actual_frame.merge(baseline_frame, on="date").merge(deep_frame, on="date")
        figure, axis = plt.subplots(figsize=(16, 6))

        if target_variable == "precipitation":
            axis.bar(
                plot_frame["date"],
                plot_frame["actual"],
                color="#8ecae6",
                alpha=0.65,
                width=0.03,
                label="Actual",
            )
            axis.plot(
                plot_frame["date"],
                plot_frame["baseline_prediction"],
                color="#264653",
                linewidth=1.8,
                label="Baseline",
            )
            axis.plot(
                plot_frame["date"],
                plot_frame["deep_prediction"],
                color="#e76f51",
                linewidth=1.8,
                label="Deep Learning",
            )
            axis.set_ylabel("Precipitation (mm)")
            axis.set_title("24-Hour Precipitation Forecast Snapshot on Test Set")
        else:
            axis.plot(
                plot_frame["date"],
                plot_frame["actual"],
                color="#023047",
                linewidth=2.0,
                label="Actual",
            )
            axis.plot(
                plot_frame["date"],
                plot_frame["baseline_prediction"],
                color="#264653",
                linewidth=1.6,
                label="Baseline",
            )
            axis.plot(
                plot_frame["date"],
                plot_frame["deep_prediction"],
                color="#e76f51",
                linewidth=1.6,
                label="Deep Learning",
            )
            axis.set_ylabel("Temperature (deg C)")
            axis.set_title("24-Hour Temperature Forecast Snapshot on Test Set")

        axis.set_xlabel("Date")
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(
            self.config.figures_dir / f"{target_variable}_24h_snapshot.png",
            dpi=300,
        )
        plt.close(figure)

    @classmethod
    def parse_target_column(cls, target_column: str) -> dict[str, str | int]:
        """Parse a target column into its variable name and forecast horizon."""
        match = cls.TARGET_PATTERN.fullmatch(target_column)
        if match is None:
            raise ValueError(f"Unrecognized target column format: {target_column}")

        return {
            "target_variable": match.group("variable"),
            "horizon_hours": int(match.group("horizon")),
        }


def main() -> None:
    """Run the final academic evaluation pipeline."""
    config = PhaseFiveConfig()
    evaluator = AcademicClimateEvaluator(config)
    regression_frame, extreme_frame, summary = evaluator.run()

    print("Phase 5 completed successfully.")
    print(f"Regression comparison saved to: {config.regression_output_path}")
    print(f"Extreme-event metrics saved to: {config.extreme_output_path}")
    print(f"Extreme thresholds saved to: {config.thresholds_output_path}")
    print(f"Academic summary saved to: {config.summary_output_path}")
    print(f"Figures saved to: {config.figures_dir}")
    print(f"Regression rows: {len(regression_frame)}")
    print(f"Extreme-event rows: {len(extreme_frame)}")
    print(f"Summary sections: {len(summary)}")


if __name__ == "__main__":
    main()
