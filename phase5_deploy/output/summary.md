# Basketball-God — Final Build Summary

## Model Performance (Test: 2015-2025 NCAA Tournament)

| Metric | Model | Seed Baseline |
|--------|-------|---------------|
| Accuracy | 72.5% | 28.8% |
| Accuracy lift | +43.6% | — |
| 95% CI on lift | [36.3%, 50.2%] | — |
| Avg ESPN pts | 126.2 | 51.1 |
| Log-loss | 0.5129 | — |
| Brier score | 0.1739 | — |

## Calibration

- ECE: 0.0317
- MCE: 0.0718
- Status: well-calibrated

## Top 10 Features

1. `diff_conf_win_pct`
2. `diff_sos`
3. `diff_avg_margin`
4. `t1_seed`
5. `diff_massey_avg_rank`
6. `t2_seed`
7. `diff_win_pct`
8. `t2_avg_margin`
9. `t2_win_pct`
10. `t1_avg_margin`

## Worst 3 Seasons

- **2018**: 65.1% accuracy
- **2022**: 66.7% accuracy
- **2016**: 68.3% accuracy

## Architecture

- **Phase 1**: 202,529 games ingested (1985-2026), SQLite DB
- **Phase 2**: 3-tier feature engineering (31 diff features + 12 absolute)
- **Phase 3**: 4 era-aware approaches trained, transfer learning best for tourney
- **Phase 4**: Round calibration + path features + Monte Carlo bracket simulation
- **Phase 5**: Bootstrap CI, SHAP analysis, production predictor API

## Usage

```python
from phase5_deploy.robustness import BasketballGodPredictor
predictor = BasketballGodPredictor.load()
result = predictor.predict('Duke', 'North Carolina', season=2025,
                           round_name='Round of 64')
print(result)
```