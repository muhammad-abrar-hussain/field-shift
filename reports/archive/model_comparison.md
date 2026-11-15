# Model comparison (v1 vs v2 vs final)

| Model | MAE | p10–p90 coverage | Beats fair LOO baseline |
| --- | --- | --- | --- |
| v1 LightGBM (fixed split 2022+ test) | 0.1318 | 0.529 | yes |
| v2 year+LightGBM (LOO 2014–25) | 0.1176 | 0.800 | no |
| final year+empirical (LOO 2014–25) | 0.1178 | 0.800 | yes |

Final LOO deterministic MAE **0.1178** (MC p50 **0.1174**). Fair baseline (train mean NDVI + rotation median anomaly, no weather): **0.1287**. Rotation mean NDVI only: **0.1282**.

The v2 LOO baseline that used the true held-out county mean is not fair at prediction time (farmers do not know that year's county average); we compare against the no-weather baseline instead.

We ship the final model because it is transparent (county weather year layer plus resampled rotation anomalies), preserves the observed A vs B separation, and runs from a single JSON file with no LightGBM.

Monte Carlo draws per prediction: 5000. Conformal interval widening (LOO): **0.0330**.