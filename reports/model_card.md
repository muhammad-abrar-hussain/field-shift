# Field Shift — model card

## Purpose
Predict peak-season winter wheat NDVI (crop health proxy) and compare three USDA CDL-labelled rotations for Finney County, Kansas dryland fields.

## Training data
1,490 high-purity wheat field-years from 1,000 sample points (2013–2025), MODIS MOD13Q1 NDVI, NASA POWER weather, USDA CDL, SSURGO soil attributes at each point.

## Final model (shipped)
Empirical Bayes rotation + year effects with Monte Carlo weather draws (`models/final_model.json`). Conformal adjustment on p10–p90 intervals.

## Validation (held-out years)
| Metric | Value |
|--------|-------|
| Leave-one-year-out MAE (MC p50) | 0.118 NDVI |
| Fair baseline MAE (no weather) | 0.129 NDVI |
| p10–p90 coverage (after conformal) | 80% |
| County NDVI vs NASS wheat yield | r = 0.86 (n = 6 years; sanity check only) |

## Limitations
- Dryland wheat focus; irrigated pivots excluded from training.
- Rotation C (forage) has very few pure fields — low confidence.
- NDVI is greenness, not yield or profit.
- NASS validation is county-level with small n.

## References
See `README.md` and `CURSOR_CONTEXT.md` for data sources and methodology.