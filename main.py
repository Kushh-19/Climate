"""Simple command runner for the climate forecasting semester project.

The public CLI is intentionally small so the project is easier to understand:
1. Prepare the data.
2. Train a baseline model.
3. Train the recurrent model.
4. Compare the final results.

Legacy phase aliases are still accepted so older notes and screenshots keep
working, but the main workflow no longer depends on phase-first terminology.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable


def run_data_cleaning() -> None:
    """Clean the raw dataset and generate EDA outputs."""
    from src.data_pipeline.preprocess import main as pipeline_main

    pipeline_main()


def run_feature_engineering() -> None:
    """Build the supervised forecasting dataset."""
    from src.data_pipeline.feature_engineering import main as pipeline_main

    pipeline_main()


def run_baseline_training() -> None:
    """Train and evaluate the baseline machine-learning models."""
    from src.models.train_model import main as pipeline_main

    pipeline_main()


def run_recurrent_training() -> None:
    """Train and evaluate the recurrent deep-learning models."""
    from src.models.train_deep_model import main as pipeline_main

    pipeline_main()


def run_final_evaluation() -> None:
    """Compare the saved model outputs and create final visuals."""
    from src.models.academic_evaluation import main as pipeline_main

    pipeline_main()


STEP_RUNNERS: dict[str, Callable[[], None]] = {
    "clean-data": run_data_cleaning,
    "build-features": run_feature_engineering,
    "train-baseline": run_baseline_training,
    "train-rnn": run_recurrent_training,
    "evaluate": run_final_evaluation,
}

STEP_LABELS: dict[str, str] = {
    "clean-data": "data cleaning",
    "build-features": "feature engineering",
    "train-baseline": "baseline training",
    "train-rnn": "recurrent-model training",
    "evaluate": "final evaluation",
}

COMMAND_GROUPS: dict[str, list[str]] = {
    "clean-data": ["clean-data"],
    "build-features": ["build-features"],
    "prepare-data": ["clean-data", "build-features"],
    "train-baseline": ["train-baseline"],
    "train-rnn": ["train-rnn"],
    "evaluate": ["evaluate"],
    "all": [
        "clean-data",
        "build-features",
        "train-baseline",
        "train-rnn",
        "evaluate",
    ],
}

LEGACY_ALIASES: dict[str, str] = {
    "phase1": "clean-data",
    "phase2": "build-features",
    "phase3": "train-baseline",
    "phase4": "train-rnn",
    "phase5": "evaluate",
}

COMMAND_HELP = """Available commands:
  clean-data      Clean the raw hourly dataset and create EDA outputs.
  build-features  Create lag, rolling, and target features from the cleaned data.
  prepare-data    Run both data-preparation steps in sequence.
  train-baseline  Train the Random Forest and persistence baselines.
  train-rnn       Train the LSTM and GRU sequence models.
  evaluate        Build comparison tables and final result figures.
  all             Run the complete project workflow from start to finish.
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the semester-project CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run one workflow command for the climate forecasting semester project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=COMMAND_HELP,
    )
    parser.add_argument(
        "command",
        help="Workflow command to run. Use one of the commands listed below.",
    )
    return parser


def normalize_command(command: str) -> str:
    """Resolve user input into a supported public command name."""
    normalized_command = LEGACY_ALIASES.get(command.strip().lower(), command.strip().lower())
    if normalized_command not in COMMAND_GROUPS:
        valid_commands = ", ".join(COMMAND_GROUPS)
        raise ValueError(
            f"Unknown command '{command}'. Choose one of: {valid_commands}."
        )
    return normalized_command


def iter_selected_steps(command: str) -> list[str]:
    """Return the workflow steps that should run for a given command."""
    normalized_command = normalize_command(command)
    return COMMAND_GROUPS[normalized_command]


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested workflow command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        selected_steps = iter_selected_steps(arguments.command)
    except ValueError as error:
        parser.error(str(error))

    for step_name in selected_steps:
        print(f"[semester-project] Starting {STEP_LABELS[step_name]}...")
        STEP_RUNNERS[step_name]()
        print(f"[semester-project] Finished {STEP_LABELS[step_name]}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
