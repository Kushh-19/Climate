"""Shared recurrent-model and inference utilities for Phase 4."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    """Sequence-to-multi-horizon regressor built on top of LSTM or GRU blocks."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        recurrent_type: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if recurrent_type not in {"lstm", "gru"}:
            raise ValueError("recurrent_type must be either 'lstm' or 'gru'.")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        recurrent_class = nn.LSTM if recurrent_type == "lstm" else nn.GRU
        self.recurrent_type = recurrent_type
        self.recurrent_layer = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, sequence_batch: torch.Tensor) -> torch.Tensor:
        """Predict all forecast horizons from the final recurrent hidden state."""
        recurrent_output, _ = self.recurrent_layer(sequence_batch)
        final_hidden_state = recurrent_output[:, -1, :]
        return self.regression_head(final_hidden_state)


class DeepForecastPredictor:
    """Loads a saved Phase 4 checkpoint and produces batched predictions."""

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path | str,
        device: torch.device,
    ) -> tuple[RecurrentForecaster, dict]:
        """Reconstruct a trained recurrent model from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        metadata = checkpoint["metadata"]

        model = RecurrentForecaster(
            input_size=metadata["input_size"],
            hidden_size=metadata["hidden_size"],
            output_size=metadata["output_size"],
            recurrent_type=metadata["recurrent_type"],
            num_layers=metadata["num_layers"],
            dropout=metadata["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return model, metadata

    @staticmethod
    def predict(model: RecurrentForecaster, sequence_batch: torch.Tensor) -> torch.Tensor:
        """Run inference without gradient tracking."""
        model.eval()
        with torch.no_grad():
            return model(sequence_batch)
