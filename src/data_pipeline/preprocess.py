"""Phase 1 pipeline for exploratory data analysis and structural cleaning.

1. Load the raw Baroda weather dataset.
2. Apply conservative structural cleaning.
3. Quantify data quality and descriptive statistics.
4. Generate visual diagnostics for academic reporting.
5. Save a cleaned dataset for the later modeling phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class PhaseOneConfig:
    """Configuration container for the Phase 1 preprocessing pipeline."""

    raw_data_path: Path = Path("data/raw/Baroda.csv")
    cleaned_data_path: Path = Path("data/interim/Baroda_phase1_clean.csv")
    report_path: Path = Path("reports/phase1/eda_summary.json")
    figures_dir: Path = Path("reports/phase1/figures")
    local_timezone: str = "Asia/Kolkata"
    expected_frequency: str = "h"
    target_columns: tuple[str, ...] = ("temperature_2m", "precipitation")


class ClimatePhaseOnePipeline:
    """Runs structural cleaning and EDA for the localized climate dataset."""

    def __init__(self, config: PhaseOneConfig) -> None:
        self.config = config
        self.time_index_summary: dict[str, int | str] = {}
        sns.set_theme(style="whitegrid", context="talk")

    def run(self) -> tuple[pd.DataFrame, dict]:
        """Execute the full Phase 1 workflow."""
        self._ensure_output_directories()
        dataframe = self.load_and_clean_data()
        report = self.build_quality_report(dataframe)
        self.save_cleaned_dataset(dataframe)
        self.save_report(report)
        self.generate_visualizations(dataframe)
        return dataframe, report

    def _ensure_output_directories(self) -> None:
        """Create output folders required by the pipeline."""
        self.config.cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_and_clean_data(self) -> pd.DataFrame:
        """Load the raw CSV and apply conservative structural cleaning."""
        dataframe = pd.read_csv(self.config.raw_data_path)

        unnamed_columns = [
            column for column in dataframe.columns if column.lower().startswith("unnamed")
        ]
        dataframe = dataframe.drop(columns=unnamed_columns, errors="ignore")

        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        if dataframe["date"].isna().any():
            invalid_count = int(dataframe["date"].isna().sum())
            raise ValueError(f"Found {invalid_count} invalid timestamps in the raw dataset.")

        dataframe["date"] = (
            dataframe["date"]
            .dt.tz_convert(self.config.local_timezone)
            .dt.tz_localize(None)
        )

        rows_before_deduplication = len(dataframe)
        dataframe = dataframe.sort_values("date").drop_duplicates(subset="date", keep="last")
        duplicate_timestamps_removed = rows_before_deduplication - len(dataframe)

        dataframe = dataframe.set_index("date")
        dataframe = self._reindex_to_expected_frequency(dataframe)
        dataframe = dataframe.sort_index()

        self.time_index_summary["duplicate_timestamps_removed"] = duplicate_timestamps_removed
        self.time_index_summary["missing_timestamp_rows_after_reindex"] = int(
            dataframe.isna().all(axis=1).sum()
        )

        return dataframe

    def _reindex_to_expected_frequency(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Align the dataset to a complete hourly grid without imputing missing values."""
        full_index = pd.date_range(
            start=dataframe.index.min(),
            end=dataframe.index.max(),
            freq=self.config.expected_frequency,
        )
        reindexed_dataframe = dataframe.reindex(full_index)
        reindexed_dataframe.index.name = "date"
        self.time_index_summary["expected_rows"] = len(full_index)
        self.time_index_summary["observed_rows_before_reindex"] = len(dataframe)
        return reindexed_dataframe

    def build_quality_report(self, dataframe: pd.DataFrame) -> dict:
        """Assemble a JSON-serializable EDA report for academic documentation."""
        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        summary_statistics = dataframe[numeric_columns].describe().round(3).to_dict()
        missing_values = dataframe.isna().sum().sort_values(ascending=False).to_dict()
        outlier_candidates = self._compute_iqr_outlier_report(dataframe[numeric_columns])

        report = {
            "dataset_name": self.config.raw_data_path.name,
            "row_count": int(len(dataframe)),
            "column_count": int(dataframe.shape[1]),
            "date_start": str(dataframe.index.min()),
            "date_end": str(dataframe.index.max()),
            "timezone_after_cleaning": self.config.local_timezone,
            "expected_frequency": self.config.expected_frequency,
            "time_index_summary": self.time_index_summary,
            "missing_values_by_column": {key: int(value) for key, value in missing_values.items()},
            "summary_statistics": summary_statistics,
            "iqr_outlier_candidates": outlier_candidates,
            "target_correlation_matrix": dataframe[list(self.config.target_columns)]
            .corr()
            .round(4)
            .to_dict(),
        }
        return report

    @staticmethod
    def _compute_iqr_outlier_report(dataframe: pd.DataFrame) -> dict:
        """Flag statistical outlier candidates without deleting them automatically."""
        outlier_report: dict[str, dict[str, float | int]] = {}

        for column in dataframe.columns:
            series = dataframe[column].dropna()
            if series.empty:
                outlier_report[column] = {"candidate_count": 0}
                continue

            first_quartile = float(series.quantile(0.25))
            third_quartile = float(series.quantile(0.75))
            interquartile_range = third_quartile - first_quartile
            lower_bound = first_quartile - (1.5 * interquartile_range)
            upper_bound = third_quartile + (1.5 * interquartile_range)
            candidate_count = int(((series < lower_bound) | (series > upper_bound)).sum())

            outlier_report[column] = {
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "candidate_count": candidate_count,
            }

        return outlier_report

    def save_cleaned_dataset(self, dataframe: pd.DataFrame) -> None:
        """Persist the cleaned dataset for subsequent phases."""
        dataframe.reset_index().to_csv(self.config.cleaned_data_path, index=False)

    def save_report(self, report: dict) -> None:
        """Save the EDA report as human-readable JSON."""
        with self.config.report_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(report, file_pointer, indent=4)

    def generate_visualizations(self, dataframe: pd.DataFrame) -> None:
        """Create EDA figures used to understand trends and data quality."""
        self._plot_missing_values(dataframe)
        self._plot_target_overview(dataframe)
        self._plot_target_distributions(dataframe)
        self._plot_monthly_patterns(dataframe)
        self._plot_correlation_heatmap(dataframe)

    def _plot_missing_values(self, dataframe: pd.DataFrame) -> None:
        """Visualize missing values per column."""
        missing_counts = dataframe.isna().sum().sort_values(ascending=False)
        figure, axis = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x=missing_counts.index,
            y=missing_counts.values,
            hue=missing_counts.index,
            palette="crest",
            legend=False,
            ax=axis,
        )
        axis.set_title("Missing Values per Variable")
        axis.set_xlabel("Variables")
        axis.set_ylabel("Missing Count")
        axis.tick_params(axis="x", rotation=75)
        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "missing_values.png", dpi=300)
        plt.close(figure)

    def _plot_target_overview(self, dataframe: pd.DataFrame) -> None:
        """Plot long-term target behaviour after daily aggregation."""
        daily_targets = dataframe.resample("D").agg(
            {
                "temperature_2m": "mean",
                "precipitation": "sum",
            }
        )

        figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        daily_targets["temperature_2m"].plot(
            ax=axes[0],
            color="#d55e00",
            linewidth=1.0,
            title="Daily Mean Temperature",
        )
        axes[0].set_ylabel("Temperature (deg C)")

        daily_targets["precipitation"].plot(
            ax=axes[1],
            color="#0072b2",
            linewidth=1.0,
            title="Daily Total Precipitation",
        )
        axes[1].set_ylabel("Precipitation (mm)")
        axes[1].set_xlabel("Date")

        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "target_overview.png", dpi=300)
        plt.close(figure)

    def _plot_target_distributions(self, dataframe: pd.DataFrame) -> None:
        """Visualize marginal distributions of the two forecasting targets."""
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(
            dataframe["temperature_2m"].dropna(),
            kde=True,
            bins=40,
            color="#d55e00",
            ax=axes[0],
        )
        axes[0].set_title("Temperature Distribution")
        axes[0].set_xlabel("Temperature (deg C)")

        sns.histplot(
            dataframe["precipitation"].dropna(),
            kde=False,
            bins=40,
            color="#0072b2",
            ax=axes[1],
        )
        axes[1].set_title("Precipitation Distribution")
        axes[1].set_xlabel("Precipitation (mm)")

        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "target_distributions.png", dpi=300)
        plt.close(figure)

    def _plot_monthly_patterns(self, dataframe: pd.DataFrame) -> None:
        """Plot month-wise target variability to expose seasonality."""
        monthly_frame = dataframe.reset_index().copy()
        monthly_frame["month_name"] = monthly_frame["date"].dt.strftime("%b")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        sns.boxplot(
            data=monthly_frame,
            x="month_name",
            y="temperature_2m",
            order=month_order,
            color="#f4a261",
            ax=axes[0],
        )
        axes[0].set_title("Monthly Temperature Variability")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Temperature (deg C)")

        sns.boxplot(
            data=monthly_frame,
            x="month_name",
            y="precipitation",
            order=month_order,
            color="#90caf9",
            ax=axes[1],
        )
        axes[1].set_title("Monthly Precipitation Variability")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Precipitation (mm)")

        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "monthly_patterns.png", dpi=300)
        plt.close(figure)

    def _plot_correlation_heatmap(self, dataframe: pd.DataFrame) -> None:
        """Plot a correlation matrix for all numeric variables."""
        correlation_matrix = dataframe.select_dtypes(include="number").corr()
        figure, axis = plt.subplots(figsize=(16, 12))
        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            center=0.0,
            linewidths=0.5,
            square=False,
            cbar_kws={"shrink": 0.8},
            ax=axis,
        )
        axis.set_title("Correlation Heatmap of Numeric Weather Variables")
        figure.tight_layout()
        figure.savefig(self.config.figures_dir / "correlation_heatmap.png", dpi=300)
        plt.close(figure)


def main() -> None:
    """Run the Phase 1 pipeline using default project paths."""
    config = PhaseOneConfig()
    pipeline = ClimatePhaseOnePipeline(config)
    dataframe, report = pipeline.run()

    print("Phase 1 completed successfully.")
    print(f"Cleaned dataset saved to: {config.cleaned_data_path}")
    print(f"EDA report saved to: {config.report_path}")
    print(f"Figures saved to: {config.figures_dir}")
    print(
        "Date range after cleaning: "
        f"{report['date_start']} to {report['date_end']}"
    )
    print(
        "Missing timestamps introduced by hourly reindexing: "
        f"{report['time_index_summary']['missing_timestamp_rows_after_reindex']}"
    )
    print(f"Final dataframe shape: {dataframe.shape}")


if __name__ == "__main__":
    main()
