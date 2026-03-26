# Climate Change Adaptation: Micro-Climate Forecasting

This repository implements a five-phase academic pipeline for localized weather forecasting in Baroda. The project predicts `temperature_2m` and `precipitation` at 24, 48, and 72 hour horizons using both classical machine learning and PyTorch recurrent neural networks.

## Project Phases

1. `Phase 1`: Exploratory Data Analysis (EDA) and conservative structural cleaning.
2. `Phase 2`: Time-series feature engineering with lags, rolling statistics, cyclical calendar encodings, and scaling.
3. `Phase 3`: Baseline modeling with persistence and Random Forest regressors.
4. `Phase 4`: Deep learning with PyTorch LSTM and GRU models.
5. `Phase 5`: Academic evaluation with RMSE, MAE, R^2, extreme-event F1, and visualization.

## Repository Layout

- `data/raw`: Original input dataset.
- `data/interim`: Cleaned Phase 1 dataset.
- `data/processed`: Engineered features and saved prediction outputs.
- `models/artifacts`: Shared preprocessing artifacts.
- `models/phase3`: Saved baseline model artifacts.
- `models/phase4`: Saved deep-learning checkpoints.
- `reports/phase1` to `reports/phase5`: JSON reports, CSV tables, and thesis-ready figures.
- `src/data_pipeline`: Phase 1 and Phase 2 code.
- `src/models`: Phase 3 to Phase 5 code and shared evaluation helpers.
- `main.py`: Top-level CLI for executing phases.

## Environment Setup

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` includes the CPU PyTorch index so the deep-learning phase remains reproducible on machines without CUDA.

## How To Run

Run individual phases:

```powershell
python main.py phase1
python main.py phase2
python main.py phase3
python main.py phase4
python main.py phase5
```

Run the full pipeline sequentially:

```powershell
python main.py all
```

## Core Outputs

- Phase 1 cleaned data: `data/interim/Baroda_phase1_clean.csv`
- Phase 2 supervised datasets: `data/processed/phase2/`
- Phase 3 reports: `reports/phase3/`
- Phase 4 reports: `reports/phase4/`
- Phase 5 final evaluation: `reports/phase5/`

## Current Academic Findings

- Random Forest is a strong baseline for short-horizon `temperature_2m` forecasting.
- GRU performs slightly better than Random Forest at longer temperature horizons.
- Precipitation remains the hardest target because of zero inflation and rare extremes.
- Under a strict heavy-rain threshold, both baseline and deep models miss the rarest extreme precipitation events, which is an important thesis result rather than a failure of the evaluation.
