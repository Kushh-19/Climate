# Results

This page summarizes the most important findings from the final comparison.

## Short Summary

- Random Forest is the strongest short-horizon model for temperature.
- GRU performs better for temperature at longer horizons.
- Precipitation remains much harder than temperature because rainfall is sparse and highly imbalanced.
- Extreme rainfall detection is the weakest part of the project for both model families.

## Best Regression Results

| Target | Horizon | Best model | RMSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| `precipitation` | `24h` | GRU | `0.6242` | `0.1579` | `0.0707` |
| `precipitation` | `48h` | GRU | `0.6333` | `0.1767` | `0.0433` |
| `precipitation` | `72h` | Random Forest | `0.6353` | `0.1309` | `0.0337` |
| `temperature_2m` | `24h` | Random Forest | `1.5443` | `1.1323` | `0.9609` |
| `temperature_2m` | `48h` | GRU | `1.8183` | `1.3670` | `0.9457` |
| `temperature_2m` | `72h` | GRU | `2.0437` | `1.5426` | `0.9313` |

## Best Extreme-Event Results

| Target | Horizon | Best model | F1 | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: |
| `precipitation` | `24h` | Tie | `0.0000` | `0.0000` | `0.0000` |
| `precipitation` | `48h` | Tie | `0.0000` | `0.0000` | `0.0000` |
| `precipitation` | `72h` | Tie | `0.0000` | `0.0000` | `0.0000` |
| `temperature_2m` | `24h` | Random Forest | `0.8500` | `0.8863` | `0.8166` |
| `temperature_2m` | `48h` | GRU | `0.7987` | `0.8257` | `0.7735` |
| `temperature_2m` | `72h` | GRU | `0.7652` | `0.7857` | `0.7458` |

## Interpretation

### Temperature

Temperature prediction is the stronger task in this project. Both model families perform well, but the GRU shows clearer benefits once the forecast horizon becomes longer.

### Precipitation

Precipitation is much harder because many hours are dry and extreme rainfall is rare. The regression metrics show only small differences between models, and the event-detection results make the limitation clear: rare heavy-rain cases are still missed.

### Academic Takeaway

The project gives a balanced result:

- classical baselines stay strong
- deep learning helps selectively instead of universally
- reporting both regression and event-based metrics gives a more honest conclusion

## Visuals

### Data Overview

![Target overview](../reports/phase1/figures/target_overview.png)

![Target distributions](../reports/phase1/figures/target_distributions.png)

![Monthly patterns](../reports/phase1/figures/monthly_patterns.png)

![Correlation heatmap](../reports/phase1/figures/correlation_heatmap.png)

### Model Comparison

![RMSE comparison](../reports/phase5/figures/rmse_comparison.png)

![Extreme F1 comparison](../reports/phase5/figures/extreme_f1_comparison.png)

![Temperature forecast snapshot](../reports/phase5/figures/temperature_2m_24h_snapshot.png)

![Precipitation forecast snapshot](../reports/phase5/figures/precipitation_24h_snapshot.png)

## Final Conclusion

For a semester project, the strongest story is not that deep learning wins everywhere. The stronger conclusion is that:

- Random Forest is already a very good benchmark
- GRU becomes useful for longer-horizon temperature forecasting
- precipitation forecasting needs more specialized imbalance-aware work in the future
