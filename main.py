"""Top-level runner for the climate forecasting project phases.

This entry point centralizes execution of the five academic phases so the
project can be reproduced from one command surface. The phase functions are
imported lazily to avoid paying the import cost of heavy dependencies such as
PyTorch when they are not required.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable


def run_phase_one() -> None:
    """Execute Phase 1: EDA and structural data cleaning."""
    from src.data_pipeline.preprocess import main as phase_main

    phase_main()


def run_phase_two() -> None:
    """Execute Phase 2: feature engineering for supervised forecasting."""
    from src.data_pipeline.feature_engineering import main as phase_main

    phase_main()


def run_phase_three() -> None:
    """Execute Phase 3: baseline model training and evaluation."""
    from src.models.train_model import main as phase_main

    phase_main()


def run_phase_four() -> None:
    """Execute Phase 4: PyTorch recurrent-model training and evaluation."""
    from src.models.train_deep_model import main as phase_main

    phase_main()


def run_phase_five() -> None:
    """Execute Phase 5: final academic evaluation and visualization."""
    from src.models.academic_evaluation import main as phase_main

    phase_main()


PHASE_RUNNERS: dict[str, Callable[[], None]] = {
    "phase1": run_phase_one,
    "phase2": run_phase_two,
    "phase3": run_phase_three,
    "phase4": run_phase_four,
    "phase5": run_phase_five,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the project CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one phase of the climate forecasting graduation project or "
            "execute the full pipeline sequentially."
        )
    )
    parser.add_argument(
        "phase",
        choices=[*PHASE_RUNNERS.keys(), "all"],
        help="Phase identifier to run, or 'all' to execute the full pipeline.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested phase or phases."""
    arguments = build_parser().parse_args(argv)
    selected_phases = (
        list(PHASE_RUNNERS.items())
        if arguments.phase == "all"
        else [(arguments.phase, PHASE_RUNNERS[arguments.phase])]
    )

    for phase_name, runner in selected_phases:
        print(f"[project-runner] Starting {phase_name}...")
        runner()
        print(f"[project-runner] Finished {phase_name}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
