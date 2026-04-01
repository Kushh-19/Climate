from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))

from src.utils.helpers import (
    PROJECT_ROOT,
    display_model_family,
    display_target_name,
    load_project_snapshot,
    read_csv,
)


st.set_page_config(
    page_title="Climate Forecasting Dashboard",
    layout="wide",
)


REQUIRED_FILES = [
    PROJECT_ROOT / "reports/phase1/eda_summary.json",
    PROJECT_ROOT / "reports/phase2/feature_engineering_summary.json",
    PROJECT_ROOT / "reports/phase5/academic_evaluation_summary.json",
    PROJECT_ROOT / "reports/phase5/regression_comparison.csv",
    PROJECT_ROOT / "reports/phase5/extreme_event_metrics.csv",
]

GALLERY_IMAGES = {
    "Target overview": PROJECT_ROOT / "reports/phase1/figures/target_overview.png",
    "Target distributions": PROJECT_ROOT / "reports/phase1/figures/target_distributions.png",
    "Monthly patterns": PROJECT_ROOT / "reports/phase1/figures/monthly_patterns.png",
    "RMSE comparison": PROJECT_ROOT / "reports/phase5/figures/rmse_comparison.png",
    "Extreme-event F1 comparison": PROJECT_ROOT / "reports/phase5/figures/extreme_f1_comparison.png",
    "Temperature forecast snapshot": (
        PROJECT_ROOT / "reports/phase5/figures/temperature_2m_24h_snapshot.png"
    ),
    "Precipitation forecast snapshot": (
        PROJECT_ROOT / "reports/phase5/figures/precipitation_24h_snapshot.png"
    ),
}


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple:
    """Load the saved report tables needed by the dashboard."""
    snapshot = load_project_snapshot()
    regression_frame = read_csv("reports/phase5/regression_comparison.csv")
    extreme_frame = read_csv("reports/phase5/extreme_event_metrics.csv")
    return snapshot, regression_frame, extreme_frame


def validate_required_files() -> list[Path]:
    """Return any expected dashboard inputs that are currently missing."""
    return [path for path in REQUIRED_FILES if not path.exists()]


def build_metric_table(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Shape a comparison table for display and charting."""
    display_frame = frame.copy()
    display_frame["Target"] = display_frame["target_variable"].map(display_target_name)
    display_frame["Model family"] = display_frame["model_family"].map(display_model_family)
    display_frame["Horizon"] = display_frame["horizon_hours"].astype(str) + "h"
    metric_label = metric_name.upper() if metric_name != "f1" else "F1"

    return (
        display_frame[["Target", "Horizon", "Model family", metric_name, "model_name"]]
        .rename(columns={metric_name: metric_label, "model_name": "Selected model"})
        .sort_values(by=["Target", "Horizon", metric_label])
    )


def add_custom_styles() -> None:
    """Inject higher-contrast styling so the dashboard is easier to read."""
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 175, 118, 0.18), transparent 26%),
                    radial-gradient(circle at top right, rgba(82, 170, 212, 0.18), transparent 24%),
                    linear-gradient(180deg, #eef4f8 0%, #f7fafc 35%, #ffffff 100%);
                color: #15242d;
            }
            .hero {
                padding: 2rem 2.2rem;
                border-radius: 28px;
                background: linear-gradient(135deg, #0d2734 0%, #1a4456 55%, #9b5b38 100%);
                box-shadow: 0 20px 48px rgba(13, 39, 52, 0.25);
                margin-bottom: 1.2rem;
            }
            .hero h1 {
                font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
                font-size: 2.7rem;
                margin-bottom: 0.4rem;
                color: #ffffff;
            }
            .hero p {
                font-size: 1.05rem;
                color: rgba(255, 255, 255, 0.92);
                margin-bottom: 1.1rem;
            }
            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
            }
            .hero-badge {
                padding: 0.65rem 0.9rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.22);
                color: #ffffff;
                font-weight: 600;
                font-size: 0.95rem;
            }
            .insight-card {
                min-height: 148px;
                padding: 1.05rem 1.15rem;
                border-radius: 20px;
                background: #ffffff;
                border: 1px solid #d9e3ea;
                box-shadow: 0 12px 30px rgba(36, 58, 70, 0.08);
                margin-bottom: 0.5rem;
            }
            .insight-card h3 {
                margin: 0 0 0.45rem 0;
                color: #17313b;
                font-size: 1.08rem;
            }
            .insight-card p {
                margin: 0;
                color: #3a5562;
                line-height: 1.55;
            }
            .panel-note {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: #f7fbfd;
                border: 1px solid #d4e3ea;
                color: #213944;
            }
            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #d9e3ea;
                padding: 1rem 1rem 0.85rem 1rem;
                border-radius: 18px;
                box-shadow: 0 10px 24px rgba(36, 58, 70, 0.08);
            }
            [data-testid="stMetricLabel"] {
                color: #486473;
            }
            [data-testid="stMetricValue"] {
                color: #10242c;
            }
            .stTabs [role="tablist"] {
                gap: 0.5rem;
            }
            .stTabs [role="tab"] {
                border-radius: 999px;
                padding: 0.55rem 1rem;
                background: #e9f0f4;
                color: #19313b;
                border: 1px solid #d0dde5;
                font-weight: 600;
            }
            .stTabs [role="tab"][aria-selected="true"] {
                background: #17313b;
                color: #ffffff;
                border-color: #17313b;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #17313b 0%, #224655 100%);
            }
            [data-testid="stSidebar"] * {
                color: #f5fbff;
            }
            .stDataFrame, div[data-testid="stImage"] {
                background: #ffffff;
                border-radius: 18px;
                padding: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(snapshot) -> None:
    """Render the compact sidebar summary and quick commands."""
    st.sidebar.title("Project Notes")
    st.sidebar.write(
        "This dashboard is for local presentation and semester-project demonstration."
    )
    st.sidebar.markdown("**Quick run commands**")
    st.sidebar.code("python main.py all", language="powershell")
    st.sidebar.code("streamlit run dashboard/streamlit_app.py", language="powershell")
    st.sidebar.markdown("**Forecast targets**")
    st.sidebar.write("Temperature: 24h, 48h, 72h")
    st.sidebar.write("Precipitation: 24h, 48h, 72h")
    st.sidebar.markdown("**Date range**")
    st.sidebar.write(f"{snapshot.date_start} to {snapshot.date_end}")


def render_hero(snapshot) -> None:
    """Render the headline banner."""
    st.markdown(
        f"""
        <section class="hero">
            <h1>Localized Climate Forecasting for Baroda</h1>
            <p>
                A clearer local dashboard for presenting the semester project results. It compares
                baseline machine learning with recurrent neural networks for temperature and
                precipitation forecasting.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">{snapshot.cleaned_rows:,} cleaned rows</span>
                <span class="hero-badge">{snapshot.feature_count:,} engineered features</span>
                <span class="hero-badge">{snapshot.target_count} forecast targets</span>
                <span class="hero-badge">24h, 48h, 72h horizons</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_insight_cards() -> None:
    """Render short, high-visibility project findings."""
    insight_columns = st.columns(3)
    cards = [
        (
            "Temperature Forecasting",
            "Random Forest stays very strong at 24 hours, while GRU performs better as the horizon becomes longer.",
        ),
        (
            "Precipitation Forecasting",
            "The deep model gives only small RMSE gains, which shows rainfall is still a difficult target.",
        ),
        (
            "Extreme Events",
            "Heat-event detection is reasonably strong, but heavy-rain detection remains weak for both model families.",
        ),
    ]
    for column, (title, description) in zip(insight_columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def plot_metric_chart(metric_frame: pd.DataFrame, metric_name: str, title: str, y_label: str) -> None:
    """Render a higher-contrast comparison plot with explicit colors."""
    display_frame = metric_frame.copy()
    display_frame["Model family"] = display_frame["model_family"].map(display_model_family)
    figure, axis = plt.subplots(figsize=(7.2, 4.3))

    colors = {
        "Baseline": "#1d5c7a",
        "Deep Learning": "#d95d39",
    }

    for family_name, family_frame in display_frame.groupby("Model family", sort=True):
        ordered_frame = family_frame.sort_values("horizon_hours")
        axis.plot(
            ordered_frame["horizon_hours"],
            ordered_frame[metric_name],
            marker="o",
            linewidth=2.8,
            markersize=8,
            color=colors.get(family_name, "#3f5c68"),
            label=family_name,
        )
        for _, row in ordered_frame.iterrows():
            axis.annotate(
                f"{row[metric_name]:.2f}",
                (row["horizon_hours"], row[metric_name]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="#17313b",
            )

    axis.set_title(title, fontsize=13, color="#17313b", pad=12)
    axis.set_xlabel("Forecast Horizon (hours)", color="#28424d")
    axis.set_ylabel(y_label, color="#28424d")
    axis.set_xticks([24, 48, 72])
    axis.grid(alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)

    if metric_name == "f1":
        axis.set_ylim(0.0, 1.0)

    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_key_visuals() -> None:
    """Render the strongest pre-generated visuals near the top of the dashboard."""
    visual_columns = st.columns(2)
    lead_visuals = [
        ("RMSE Comparison", GALLERY_IMAGES["RMSE comparison"]),
        ("Extreme-event F1 Comparison", GALLERY_IMAGES["Extreme-event F1 comparison"]),
    ]
    for column, (title, image_path) in zip(visual_columns, lead_visuals):
        with column:
            st.subheader(title)
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)


def render_overview_tab(snapshot, regression_frame: pd.DataFrame, extreme_frame: pd.DataFrame) -> None:
    """Render the project snapshot tab."""
    overview_regression_frame = (
        regression_frame.groupby(["model_family", "horizon_hours"], as_index=False)["rmse"].mean()
    )
    overview_extreme_frame = (
        extreme_frame.groupby(["model_family", "horizon_hours"], as_index=False)["f1"].mean()
    )

    metrics = st.columns(4)
    metrics[0].metric("Cleaned rows", f"{snapshot.cleaned_rows:,}")
    metrics[1].metric("Modeling rows", f"{snapshot.modeled_rows:,}")
    metrics[2].metric("Engineered features", f"{snapshot.feature_count:,}")
    metrics[3].metric("Forecast targets", f"{snapshot.target_count:,}")

    split_columns = st.columns(3)
    split_columns[0].metric("Train rows", f"{snapshot.train_rows:,}")
    split_columns[1].metric("Validation rows", f"{snapshot.validation_rows:,}")
    split_columns[2].metric("Test rows", f"{snapshot.test_rows:,}")

    threshold_columns = st.columns(2)
    temperature_threshold = (
        "N/A" if snapshot.temperature_threshold is None else f"{snapshot.temperature_threshold:.3f} deg C"
    )
    precipitation_threshold = (
        "N/A" if snapshot.precipitation_threshold is None else f"{snapshot.precipitation_threshold:.3f} mm"
    )
    threshold_columns[0].metric("Extreme temperature threshold", temperature_threshold)
    threshold_columns[1].metric("Extreme precipitation threshold", precipitation_threshold)

    render_insight_cards()
    render_key_visuals()

    chart_columns = st.columns(2)
    with chart_columns[0]:
        plot_metric_chart(
            overview_regression_frame,
            metric_name="rmse",
            title="Overall RMSE Comparison",
            y_label="RMSE",
        )
    with chart_columns[1]:
        plot_metric_chart(
            overview_extreme_frame,
            metric_name="f1",
            title="Overall Extreme-event F1 Comparison",
            y_label="F1 Score",
        )

    overview_columns = st.columns(2)
    with overview_columns[0]:
        st.subheader("Best regression results")
        best_regression_frame = pd.DataFrame(snapshot.best_regression_models)
        best_regression_frame["Target"] = best_regression_frame["target_variable"].map(
            display_target_name
        )
        best_regression_frame["Model family"] = best_regression_frame["model_family"].map(
            display_model_family
        )
        best_regression_frame["Horizon"] = best_regression_frame["horizon_hours"].astype(str) + "h"
        st.dataframe(
            best_regression_frame[
                ["Target", "Horizon", "Model family", "model_name", "rmse", "mae", "r2"]
            ].rename(
                columns={
                    "model_name": "Selected model",
                    "rmse": "RMSE",
                    "mae": "MAE",
                    "r2": "R2",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with overview_columns[1]:
        st.subheader("Best extreme-event detection")
        best_extreme_frame = pd.DataFrame(snapshot.best_extreme_models)
        best_extreme_frame["Target"] = best_extreme_frame["target_variable"].map(
            display_target_name
        )
        best_extreme_frame["Model family"] = best_extreme_frame["model_family"].map(
            display_model_family
        )
        best_extreme_frame["Horizon"] = best_extreme_frame["horizon_hours"].astype(str) + "h"
        st.dataframe(
            best_extreme_frame[
                ["Target", "Horizon", "Model family", "model_name", "precision", "recall", "f1"]
            ].rename(
                columns={
                    "model_name": "Selected model",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Average model-family performance")
    family_columns = st.columns(2)
    with family_columns[0]:
        family_regression_frame = (
            pd.DataFrame(snapshot.mean_regression_metrics)
            .T.rename_axis("Model family")
            .reset_index()
        )
        family_regression_frame["Model family"] = family_regression_frame["Model family"].map(
            display_model_family
        )
        st.dataframe(
            family_regression_frame.rename(
                columns={"rmse": "RMSE", "mae": "MAE", "r2": "R2"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with family_columns[1]:
        family_extreme_frame = (
            pd.DataFrame(snapshot.mean_extreme_metrics)
            .T.rename_axis("Model family")
            .reset_index()
        )
        family_extreme_frame["Model family"] = family_extreme_frame["Model family"].map(
            display_model_family
        )
        st.dataframe(
            family_extreme_frame.rename(
                columns={"precision": "Precision", "recall": "Recall", "f1": "F1"}
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_comparison_tab(regression_frame: pd.DataFrame, extreme_frame: pd.DataFrame) -> None:
    """Render the interactive comparison tab."""
    target_option = st.selectbox(
        "Choose a target variable",
        options=["temperature_2m", "precipitation"],
        format_func=display_target_name,
    )

    target_regression_frame = regression_frame[
        regression_frame["target_variable"] == target_option
    ].copy()
    target_extreme_frame = extreme_frame[extreme_frame["target_variable"] == target_option].copy()

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("RMSE by horizon")
        plot_metric_chart(
            target_regression_frame,
            metric_name="rmse",
            title=f"{display_target_name(target_option)} RMSE",
            y_label="RMSE",
        )
        st.dataframe(
            build_metric_table(target_regression_frame, metric_name="rmse"),
            use_container_width=True,
            hide_index=True,
        )

    with chart_columns[1]:
        st.subheader("Extreme-event F1 by horizon")
        plot_metric_chart(
            target_extreme_frame,
            metric_name="f1",
            title=f"{display_target_name(target_option)} Extreme-event F1",
            y_label="F1 Score",
        )
        st.dataframe(
            build_metric_table(target_extreme_frame, metric_name="f1"),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <div class="panel-note">
            <strong>Reading tip:</strong> lower RMSE is better for regression, while higher F1 is
            better for extreme-event detection. This gives a fuller picture than reporting only one
            metric family.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gallery_tab() -> None:
    """Render the saved figure gallery."""
    st.write("Saved figures from the project reports are shown below for presentation use.")
    gallery_items = [(title, path) for title, path in GALLERY_IMAGES.items() if path.exists()]
    for index in range(0, len(gallery_items), 2):
        pair = gallery_items[index : index + 2]
        gallery_columns = st.columns(2)
        for column, (title, image_path) in zip(gallery_columns, pair):
            with column:
                st.subheader(title)
                st.image(str(image_path), use_container_width=True)


def main() -> None:
    """Run the local Streamlit dashboard."""
    missing_files = validate_required_files()
    if missing_files:
        st.error("Some saved outputs are missing, so the dashboard cannot be rendered yet.")
        st.write("Run the full workflow first:")
        st.code("python main.py all", language="powershell")
        st.write("Missing files:")
        for missing_file in missing_files:
            st.write(f"- {missing_file.relative_to(PROJECT_ROOT)}")
        st.stop()

    snapshot, regression_frame, extreme_frame = load_dashboard_data()

    add_custom_styles()
    render_sidebar(snapshot)
    render_hero(snapshot)

    overview_tab, comparison_tab, gallery_tab = st.tabs(
        ["Project Overview", "Model Comparison", "Visual Gallery"]
    )

    with overview_tab:
        render_overview_tab(snapshot, regression_frame, extreme_frame)

    with comparison_tab:
        render_comparison_tab(regression_frame, extreme_frame)

    with gallery_tab:
        render_gallery_tab()


if __name__ == "__main__":
    main()
