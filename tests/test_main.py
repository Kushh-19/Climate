"""Smoke tests for the simplified CLI command surface."""

from __future__ import annotations

import unittest

import main


class MainCommandTests(unittest.TestCase):
    """Verify the simplified command mapping stays stable."""

    def test_prepare_data_runs_two_steps(self) -> None:
        self.assertEqual(
            main.iter_selected_steps("prepare-data"),
            ["clean-data", "build-features"],
        )

    def test_phase_aliases_still_work(self) -> None:
        self.assertEqual(main.normalize_command("phase3"), "train-baseline")
        self.assertEqual(main.normalize_command("phase5"), "evaluate")

    def test_all_command_expands_full_workflow(self) -> None:
        self.assertEqual(
            main.iter_selected_steps("all"),
            [
                "clean-data",
                "build-features",
                "train-baseline",
                "train-rnn",
                "evaluate",
            ],
        )

    def test_invalid_command_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            main.normalize_command("not-a-real-command")


if __name__ == "__main__":
    unittest.main()
