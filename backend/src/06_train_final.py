# Final shipped model: year regression + empirical rotation anomalies (Monte Carlo).

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_V2 = PROJECT_ROOT / "data" / "processed" / "model_table_v2.parquet"
TABLE_V1 = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
FINAL_JSON = PROJECT_ROOT / "models" / "final_model.json"
FEATURE_STORE = PROJECT_ROOT / "data" / "processed" / "feature_store.parquet"
COMPARISON_MD = PROJECT_ROOT / "reports" / "model_comparison.md"
ROTATION_PNG = PROJECT_ROOT / "reports" / "rotation_effect.png"
METRICS_V1 = PROJECT_ROOT / "reports" / "model_metrics.json"
METRICS_V2 = PROJECT_ROOT / "reports" / "model_metrics_v2.json"

YEAR_FEATURES = ["season_rain_mm", "heat_days"]
TARGET = "peak_ndvi"
LOO_YEAR_MIN = 2013
LOO_YEAR_MAX = 2025
MC_DRAWS = 5000
RNG = np.random.default_rng(42)
YEAR_NDVI_CLIP = (0.4, 0.8)
SHRINK_K = 100
MIN_DRY_N = 100

ROTATIONS = ("A", "B", "C", "other")
ROTATION_CYCLE_YEARS = {"A": 2, "B": 3, "C": 2, "other": 2}


def _purity_mask(df: pd.DataFrame) -> pd.Series:
    thr = np.where(df["rotation_context"] == "C", 0.5, 0.7)
    return df["purity"].to_numpy() >= thr


def _yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year", as_index=False)
        .agg(
            county_mean_ndvi=(TARGET, "mean"),
            season_rain_mm=("season_rain_mm", "first"),
            heat_days=("heat_days", "first"),
        )
        .sort_values("year")
    )


def _fit_year_pipe(yearly: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())])
    pipe.fit(yearly[YEAR_FEATURES], yearly["county_mean_ndvi"])
    return pipe


def _year_params(pipe: Pipeline) -> dict:
    scale: StandardScaler = pipe.named_steps["scale"]
    reg: LinearRegression = pipe.named_steps["reg"]
    return {
        "features": YEAR_FEATURES,
        "intercept": float(reg.intercept_),
        "coefficients": {f: float(c) for f, c in zip(YEAR_FEATURES, reg.coef_)},
        "scaler_mean": {f: float(m) for f, m in zip(YEAR_FEATURES, scale.mean_)},
        "scaler_std": {f: float(s) for f, s in zip(YEAR_FEATURES, scale.scale_)},
    }


def _predict_year_mu(params: dict, season_rain_mm: float, heat_days: float, *, clip: bool = True) -> float:
    vals = {"season_rain_mm": float(season_rain_mm), "heat_days": float(heat_days)}
    mu = float(params["intercept"])
    for f in YEAR_FEATURES:
        z = (vals[f] - params["scaler_mean"][f]) / params["scaler_std"][f]
        mu += params["coefficients"][f] * z
    if not clip:
        return mu
    lo, hi = YEAR_NDVI_CLIP
    return float(np.clip(mu, lo, hi))


def _year_loo_county_pred(yearly: pd.DataFrame, hold_year: int, *, clip: bool = False) -> float:
    """Predict county-mean NDVI for hold_year using all other years (NDVI units)."""
    train = yearly[yearly["year"] != hold_year]
    if len(train) < 3:
        return float("nan")
    pipe = _fit_year_pipe(train)
    row = yearly.loc[yearly["year"] == hold_year, YEAR_FEATURES]
    pred = float(pipe.predict(row)[0])
    if clip:
        lo, hi = YEAR_NDVI_CLIP
        pred = float(np.clip(pred, lo, hi))
    return pred


def _conformal_delta(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: float = 0.80) -> tuple[float, float, float]:
    need = np.maximum(lo - y, y - hi)
    need = np.maximum(need, 0.0)
    n = len(y)
    if n == 0:
        return 0.0, 0.0, 0.0
    order = np.sort(need)
    idx = int(np.ceil(target * n)) - 1
    idx = min(max(idx, 0), n - 1)
    delta = float(order[idx])
    cov_b = _interval_coverage(y, lo, hi)
    cov_a = _interval_coverage(y, lo - delta, hi + delta)
    return delta, cov_b, cov_a


def _loo_year_residuals(yearly: pd.DataFrame) -> tuple[list[float], float, list[dict]]:
    residuals: list[float] = []
    folds: list[dict] = []
    for hold in yearly["year"]:
        train = yearly[yearly["year"] != hold]
        if len(train) < 3:
            continue
        pipe = _fit_year_pipe(train)
        row = yearly.loc[yearly["year"] == hold, YEAR_FEATURES]
        pred = float(pipe.predict(row)[0])
        obs = float(yearly.loc[yearly["year"] == hold, "county_mean_ndvi"].iloc[0])
        res = obs - pred
        residuals.append(res)
        folds.append({"year": int(hold), "obs": obs, "pred": pred, "residual": res})
    mae = float(np.mean([abs(f["obs"] - f["pred"]) for f in folds])) if folds else float("nan")
    return residuals, mae, folds


def _confidence_label(n: int) -> str:
    if n >= 300:
        return "high"
    if n >= 100:
        return "medium"
    return "low"


def _dry_harvest_years() -> set[int]:
    if not WEATHER_CSV.is_file():
        return set()
    w = pd.read_csv(WEATHER_CSV).rename(columns={"harvest_year": "year"})
    w = w[(w["year"] >= LOO_YEAR_MIN) & (w["year"] <= LOO_YEAR_MAX)]
    if w.empty:
        return set()
    cutoff = float(w["season_rain_mm"].quantile(1.0 / 3.0))
    return set(w.loc[w["season_rain_mm"] <= cutoff, "year"].astype(int).tolist())


def _rotation_anomaly_pack(anom_all: np.ndarray, anom_dry: np.ndarray, bare: pd.Series) -> dict:
    n = int(len(anom_all))
    if n == 0:
        return {
            "n": 0,
            "n_dry_years": 0,
            "n_dry_rows": 0,
            "anomaly_median": None,
            "anomaly_median_shrunk": None,
            "anomaly_median_dry": None,
            "anomaly_median_dry_shrunk": None,
            "anomaly_p10": None,
            "anomaly_p90": None,
            "anomalies_empirical": [],
            "anomalies_mc_normal": [],
            "anomalies_mc_dry": [],
            "use_dry_effect_in_scenarios": False,
            "median_prev_year_bare_days": None,
            "confidence": "low",
            "shrink_k": SHRINK_K,
        }
    med = float(np.median(anom_all))
    shrunk = med * n / (n + SHRINK_K)
    w = n / (n + SHRINK_K)
    shrunk_tight = [shrunk + (float(a) - med) * w for a in anom_all]
    # Low-n rotations (e.g. C): keep empirical spread around shrunk centre
    if n < MIN_DRY_N:
        mc_normal = [shrunk + (float(a) - med) for a in anom_all]
    else:
        mc_normal = shrunk_tight

    nd = int(len(anom_dry))
    med_dry = float(np.median(anom_dry)) if nd else None
    shrunk_dry = med_dry * nd / (nd + SHRINK_K) if med_dry is not None and nd > 0 else None
    mc_dry: list[float] = []
    use_dry = False
    if nd > 0 and med_dry is not None:
        wd = nd / (nd + SHRINK_K)
        if nd >= MIN_DRY_N:
            mc_dry = [float(shrunk_dry) + (float(a) - med_dry) * wd for a in anom_dry]
            use_dry = True
        else:
            mc_dry = [float(shrunk_dry) + (float(a) - med_dry) for a in anom_dry]

    return {
        "n": n,
        "n_dry_years": nd,
        "anomaly_median": med,
        "anomaly_median_shrunk": float(shrunk),
        "anomaly_median_dry": med_dry,
        "anomaly_median_dry_shrunk": float(shrunk_dry) if shrunk_dry is not None else None,
        "anomaly_p10": float(np.quantile(anom_all, 0.1)),
        "anomaly_p90": float(np.quantile(anom_all, 0.9)),
        "anomalies_empirical": [float(x) for x in anom_all],
        "anomalies_mc_normal": [float(x) for x in mc_normal],
        "anomalies_mc_dry": mc_dry,
        "use_dry_effect_in_scenarios": use_dry,
        "n_dry_rows": nd,
        "median_prev_year_bare_days": float(bare.median()) if len(bare) else None,
        "confidence": _confidence_label(n),
        "shrink_k": SHRINK_K,
    }


def _field_stats(train: pd.DataFrame, dry_years: set[int]) -> dict[str, dict]:
    train = train.copy()
    train["county_mean"] = train.groupby("year")[TARGET].transform("mean")
    train["ndvi_anom"] = train[TARGET] - train["county_mean"]
    out: dict[str, dict] = {}
    for rot in ROTATIONS:
        sub = train[train["rotation_context"] == rot]
        anom = sub["ndvi_anom"].dropna().to_numpy()
        sub_dry = sub[sub["year"].isin(dry_years)]
        anom_dry = sub_dry["ndvi_anom"].dropna().to_numpy()
        bare = sub["prev_year_bare_days"].dropna()
        out[rot] = _rotation_anomaly_pack(anom, anom_dry, bare)
    return out


def _point_year_bare_lut() -> dict[int, dict[int, float]]:
    if not FEATURE_STORE.is_file():
        return {}
    fs = pd.read_parquet(FEATURE_STORE)
    lut: dict[int, dict[int, float]] = {}
    for _, row in fs.iterrows():
        pid = int(row["point_id"])
        trends = json.loads(row.get("yearly_trends_json") or "[]")
        lut[pid] = {
            int(t["year"]): float(t["bare_days"])
            for t in trends
            if t.get("bare_days") is not None and t.get("year") is not None
        }
    return lut


def _median_bare_days_per_rotation_year(train: pd.DataFrame) -> dict[str, float | None]:
    """Mean NDVI bare days across each rotation cycle window, median across field-years (per §10)."""
    bare_lut = _point_year_bare_lut()
    if not bare_lut:
        return {rot: None for rot in ROTATIONS}
    out: dict[str, list[float]] = {rot: [] for rot in ROTATIONS}
    for _, row in train.iterrows():
        rot = str(row["rotation_context"])
        if rot not in out:
            continue
        cycle = int(ROTATION_CYCLE_YEARS.get(rot, 2))
        hy = int(row["year"])
        pid = int(row["point_id"])
        years = range(hy - cycle + 1, hy + 1)
        days = [bare_lut.get(pid, {}).get(y) for y in years]
        days = [float(d) for d in days if d is not None]
        if days:
            out[rot].append(sum(days) / cycle)
    return {rot: (float(np.median(vals)) if vals else None) for rot, vals in out.items()}


def monte_carlo_quantiles(
    year_mu: float,
    year_residuals: list[float],
    anomalies: list[float],
    *,
    n_draws: int = MC_DRAWS,
    rng: np.random.Generator | None = None,
    interval_delta: float = 0.0,
) -> tuple[float, float, float]:
    """year_mu and residuals in NDVI units; anomalies are peak − county_mean (not added twice)."""
    if not anomalies:
        return float("nan"), float("nan"), float("nan")
    rng = rng or RNG
    yr = rng.choice(year_residuals, size=n_draws) if year_residuals else np.zeros(n_draws)
    an = rng.choice(anomalies, size=n_draws)
    draws = year_mu + yr + an
    p10, p50, p90 = np.quantile(draws, [0.1, 0.5, 0.9])
    if interval_delta > 0:
        p10 -= interval_delta
        p90 += interval_delta
    return float(p10), float(p50), float(p90)


def _interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def _loo_evaluate_final(
    df: pd.DataFrame,
    year_residuals_pool: list[float],
) -> dict:
    """OOF metrics: deterministic p50 = year LOO (unclipped) + rotation median anomaly from train fold."""
    dry_years = _dry_harvest_years()
    yearly = _yearly_table(df)
    years = sorted(df["year"].unique())
    ys: list[float] = []
    det_preds: list[float] = []
    mc_p50s: list[float] = []
    p10s: list[float] = []
    p90s: list[float] = []
    fair_base_preds: list[float] = []
    rot_mean_ndvi_preds: list[float] = []
    per_year_rows: list[dict] = []

    for hold in years:
        train = df[df["year"] != hold]
        test = df[df["year"] == hold]
        if train.empty or test.empty or len(yearly[yearly["year"] != hold]) < 3:
            continue

        year_mu = _year_loo_county_pred(yearly, int(hold), clip=False)
        obs_county = float(yearly.loc[yearly["year"] == hold, "county_mean_ndvi"].iloc[0])

        rot_stats = _field_stats(train, dry_years)
        overall_mean_train = float(train[TARGET].mean())
        rot_mean_ndvi = train.groupby("rotation_context")[TARGET].mean().to_dict()

        fold_ys: list[float] = []
        fold_dets: list[float] = []
        fold_mcs: list[float] = []

        for _, row in test.iterrows():
            rot = str(row["rotation_context"])
            st = rot_stats.get(rot) or {}
            med = st.get("anomaly_median_shrunk")
            anom_list = st.get("anomalies_mc_normal") or []
            if med is None or not anom_list:
                continue
            y = float(row[TARGET])
            det = year_mu + float(med)
            rng = np.random.default_rng(int(row["point_id"]) * 10_000 + int(hold))
            p10, p50_mc, p90 = monte_carlo_quantiles(
                year_mu, year_residuals_pool, anom_list, rng=rng, interval_delta=0.0
            )
            ys.append(y)
            det_preds.append(det)
            mc_p50s.append(p50_mc)
            p10s.append(p10)
            p90s.append(p90)
            fair_base_preds.append(overall_mean_train + float(med))
            rot_mean_ndvi_preds.append(float(rot_mean_ndvi.get(rot, overall_mean_train)))
            fold_ys.append(y)
            fold_dets.append(det)
            fold_mcs.append(p50_mc)

        if fold_ys:
            per_year_rows.append(
                {
                    "year": int(hold),
                    "county_mean_observed": obs_county,
                    "year_model_loo_pred": year_mu,
                    "mae_points_deterministic": float(mean_absolute_error(fold_ys, fold_dets)),
                    "mae_points_mc_p50": float(mean_absolute_error(fold_ys, fold_mcs)),
                }
            )

    y_arr = np.asarray(ys)
    det_arr = np.asarray(det_preds)
    p10_arr = np.asarray(p10s)
    p90_arr = np.asarray(p90s)

    delta, cov_b, cov_a = _conformal_delta(y_arr, p10_arr, p90_arr)

    p10_adj = p10_arr - delta
    p90_adj = p90_arr + delta

    mae_det = float(mean_absolute_error(y_arr, det_arr)) if len(ys) else float("nan")
    mae_mc = float(mean_absolute_error(y_arr, mc_p50s)) if len(ys) else float("nan")
    return {
        "n_oof": int(len(ys)),
        "mae_deterministic": mae_det,
        "mae_mc_p50": mae_mc,
        "mae": mae_det,
        "mae_baseline_fair_no_weather": float(mean_absolute_error(y_arr, fair_base_preds)) if len(ys) else float("nan"),
        "mae_baseline_rotation_mean_ndvi": float(mean_absolute_error(y_arr, rot_mean_ndvi_preds)) if len(ys) else float("nan"),
        "coverage_p10_p90_before_conformal": cov_b,
        "coverage_p10_p90": cov_a,
        "conformal_delta": delta,
        "beats_baseline_fair": bool(mae_det < mean_absolute_error(y_arr, fair_base_preds)) if len(ys) else False,
        "per_year": per_year_rows,
        "notes": {
            "year_residuals_units": "NDVI (observed county mean minus predicted county mean)",
            "loo_year_pred_clipped": False,
            "anomaly_definition": "peak_ndvi minus county_mean within fold train years only for field stats",
        },
    }


def _plot_rotation_anomalies(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["county_mean"] = df.groupby("year")[TARGET].transform("mean")
    df["ndvi_anom"] = df[TARGET] - df["county_mean"]
    data = []
    labels = []
    ns = []
    for rot in ("A", "B", "C", "other"):
        sub = df[df["rotation_context"] == rot]["ndvi_anom"].dropna()
        if len(sub) == 0:
            continue
        data.append(sub.to_numpy())
        labels.append(rot)
        ns.append(len(sub))

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=[f"{lab}\n(n={n})" for lab, n in zip(labels, ns)], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c5e4f3")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Observed NDVI anomaly (peak − county mean)")
    ax.set_xlabel("Rotation context")
    ax.set_title("Rotation effect on wheat NDVI anomaly (purity-filtered training rows)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _load_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _beats_baseline(mae: float | None, mae_base: float | None) -> str:
    if mae is None or mae_base is None or np.isnan(mae) or np.isnan(mae_base):
        return "unknown"
    return "yes" if mae < mae_base else "no"


def _write_comparison_md(rows: list[dict], eval_final: dict, path: Path) -> None:
    lines = [
        "# Model comparison (v1 vs v2 vs final)",
        "",
        "| Model | MAE | p10–p90 coverage | Beats fair LOO baseline |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['mae']:.4f} | {r['coverage']:.3f} | {r['beats']} |"
        )
    lines.extend(
        [
            "",
            f"Final LOO deterministic MAE **{eval_final.get('mae_deterministic', float('nan')):.4f}** "
            f"(MC p50 **{eval_final.get('mae_mc_p50', float('nan')):.4f}**). "
            f"Fair baseline (train mean NDVI + rotation median anomaly, no weather): "
            f"**{eval_final.get('mae_baseline_fair_no_weather', float('nan')):.4f}**. "
            f"Rotation mean NDVI only: **{eval_final.get('mae_baseline_rotation_mean_ndvi', float('nan')):.4f}**.",
            "",
            "The v2 LOO baseline that used the true held-out county mean is not fair at prediction time (farmers do not "
            "know that year's county average); we compare against the no-weather baseline instead.",
            "",
            "We ship the final model because it is transparent (county weather year layer plus resampled rotation "
            "anomalies), preserves the observed A vs B separation, and runs from a single JSON file with no LightGBM.",
            "",
            f"Monte Carlo draws per prediction: {MC_DRAWS}. "
            f"Conformal interval widening (LOO): **{eval_final.get('conformal_delta', 0):.4f}**.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _yearly_purity_high(table: pd.DataFrame, year_min: int = 2013, year_max: int = 2025) -> pd.DataFrame:
    sub = table[(table["year"] >= year_min) & (table["year"] <= year_max)].copy()
    sub = sub[sub["purity"] >= 0.7]
    return _yearly_table(sub)


def _print_year_model_sanity(year_params: dict, yearly: pd.DataFrame) -> None:
    """Part A: weather standardization, LOO coefficients, full-fit county means."""
    weather = pd.read_csv(WEATHER_CSV).rename(columns={"harvest_year": "year"})
    weather = weather[(weather["year"] >= 2013) & (weather["year"] <= 2025)]

    print("\n=== Part A: Year model sanity (2013–2025) ===")
    print(f"{'year':>6} {'rain_mm':>9} {'heat_d':>7} {'z_rain':>8} {'z_heat':>8}")
    for _, row in weather.iterrows():
        y = int(row["year"])
        rain, heat = float(row["season_rain_mm"]), float(row["heat_days"])
        zr = (rain - year_params["scaler_mean"]["season_rain_mm"]) / year_params["scaler_std"]["season_rain_mm"]
        zh = (heat - year_params["scaler_mean"]["heat_days"]) / year_params["scaler_std"]["heat_days"]
        print(f"{y:6d} {rain:9.2f} {heat:7.0f} {zr:8.3f} {zh:8.3f}")

    print("\nLOO year-model coefficients (county mean NDVI ~ rain + heat):")
    print(
        f"{'hold':>6} {'intercept':>10} {'b_rain':>10} {'b_heat':>10} {'pred_2024note':>14}"
    )
    for hold in sorted(yearly["year"].unique()):
        train = yearly[yearly["year"] != hold]
        if len(train) < 3:
            continue
        p = _year_params(_fit_year_pipe(train))
        row = yearly.loc[yearly["year"] == hold]
        pred = _predict_year_mu(p, float(row["season_rain_mm"].iloc[0]), float(row["heat_days"].iloc[0]), clip=False)
        note = ""
        if int(hold) == 2024:
            note = f"pred={pred:.3f}"
        print(
            f"{int(hold):6d} {p['intercept']:10.4f} "
            f"{p['coefficients']['season_rain_mm']:10.4f} "
            f"{p['coefficients']['heat_days']:10.4f} {note:>14}"
        )

    print(
        "\n2024 LOO was ~-0.04 before fix because _predict_year_mu standardized heat_days using "
        "season_rain_mm twice (copy-paste bug). sklearn Pipeline was always correct; manual/API path was wrong."
    )

    print("\nFull-fit county mean NDVI (purity >= 0.7):")
    print(f"{'year':>6} {'observed':>10} {'predicted':>10} {'in_0.4_0.8':>10}")
    ok = True
    for _, row in yearly.sort_values("year").iterrows():
        pred = _predict_year_mu(
            year_params, float(row["season_rain_mm"]), float(row["heat_days"]), clip=True
        )
        obs = float(row["county_mean_ndvi"])
        in_range = 0.4 <= pred <= 0.8
        ok = ok and in_range
        print(f"{int(row['year']):6d} {obs:10.4f} {pred:10.4f} {str(in_range):>10}")
    if not ok:
        print("WARNING: some full-fit predictions outside [0.4, 0.8] after clip.")


def run_final() -> dict:
    if not TABLE_V2.is_file():
        if not TABLE_V1.is_file():
            raise FileNotFoundError("Run src/05_build_table.py and src/06_train_v2.py first.")
        table = pd.read_parquet(TABLE_V1)
        table["purity"] = 1.0
    else:
        table = pd.read_parquet(TABLE_V2)

    df = table[(table["year"] >= LOO_YEAR_MIN) & (table["year"] <= LOO_YEAR_MAX)].copy()
    df = df[_purity_mask(df)]

    df["county_mean_ndvi"] = df.groupby("year")[TARGET].transform("mean")
    df["ndvi_anom"] = df[TARGET] - df["county_mean_ndvi"]

    yearly_high = _yearly_purity_high(table)
    yearly = _yearly_table(df)
    year_pipe = _fit_year_pipe(yearly_high)
    year_params = _year_params(year_pipe)
    _print_year_model_sanity(year_params, yearly_high)
    loo_residuals, loo_mae, loo_folds = _loo_year_residuals(yearly_high)
    year_params["loo_residuals"] = [float(r) for r in loo_residuals]
    year_params["loo_mae_county_mean_ndvi"] = loo_mae
    year_params["loo_fold_details"] = loo_folds

    dry_years = _dry_harvest_years()
    rot_stats = _field_stats(df, dry_years)
    bare_by_rot = _median_bare_days_per_rotation_year(df)
    for rot in ROTATIONS:
        rot_stats[rot]["median_bare_days_per_rotation_year"] = bare_by_rot.get(rot)
    eval_final = _loo_evaluate_final(df, loo_residuals)
    effect_rows: list[dict] = []
    print("\nRotation effects (empirical Bayes k=100; dry = lowest third of season rain years):")
    print(f"{'rot':>5} {'normal_raw':>11} {'normal_shrunk':>14} {'n':>6} {'dry_raw':>10} {'dry_shrunk':>11} {'n_dry':>6}")
    for rot in ROTATIONS:
        s = rot_stats[rot]
        effect_rows.append(
            {
                "rotation": rot,
                "normal_effect_raw": s.get("anomaly_median"),
                "normal_effect_shrunk": s.get("anomaly_median_shrunk"),
                "n": s.get("n"),
                "dry_effect_raw": s.get("anomaly_median_dry"),
                "dry_effect_shrunk": s.get("anomaly_median_dry_shrunk"),
                "n_dry_years": s.get("n_dry_years"),
                "dry_years_used_for_scenario": bool(s.get("use_dry_effect_in_scenarios")),
            }
        )
        print(
            f"{rot:>5} "
            f"{(s.get('anomaly_median') or 0):11.4f} "
            f"{(s.get('anomaly_median_shrunk') or 0):14.4f} "
            f"{s.get('n',0):6d} "
            f"{(s.get('anomaly_median_dry') if s.get('anomaly_median_dry') is not None else float('nan')):10.4f} "
            f"{(s.get('anomaly_median_dry_shrunk') if s.get('anomaly_median_dry_shrunk') is not None else float('nan')):11.4f} "
            f"{s.get('n_dry_years',0):6d}"
        )
    eval_final["before_bugfix_loo"] = {
        "mae_mc_p50_wrong": 0.2545,
        "coverage_p10_p90_wrong": 0.418,
        "mae_baseline_rotation_mean_only": 0.1283,
        "bugs": [
            "LOO year prediction was clip-biased and year residuals were taken from the wrong yearly table",
            "Shared RNG and pooled train-year residuals inflated point MAE vs deterministic year+median",
        ],
    }

    weather = pd.read_csv(WEATHER_CSV) if WEATHER_CSV.is_file() else None
    ref_weather = {}
    if weather is not None and not weather.empty:
        ref_weather = {
            "season_rain_mm": float(weather["season_rain_mm"].median()),
            "heat_days": float(weather["heat_days"].median()),
        }

    payload = {
        "version": "final",
        "description": "County weather year layer + empirical rotation anomaly resampling (Monte Carlo).",
        "monte_carlo_draws": MC_DRAWS,
        "purity_rules": {"default": 0.7, "C": 0.5},
        "year_model": {**year_params, "ndvi_clip": list(YEAR_NDVI_CLIP)},
        "rotations": rot_stats,
        "dry_harvest_years": sorted(dry_years),
        "rotation_effects_table": effect_rows,
        "empirical_bayes_k": SHRINK_K,
        "reference_weather": ref_weather,
        "interval_conformal_delta": eval_final.get("conformal_delta", 0.0),
        "evaluation_loo": eval_final,
    }

    FINAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot_rotation_anomalies(df, ROTATION_PNG)

    m1 = _load_metrics(METRICS_V1)
    m2 = _load_metrics(METRICS_V2)
    v1_mae = float(m1.get("mae_model_p50", float("nan")))
    v1_cov = float(m1.get("interval_coverage_p10_p90", float("nan")))
    v1_base = float(m1.get("mae_baseline", float("nan")))
    v2_loo = m2.get("loo_cv_field_combined", {})
    v2_mae = float(v2_loo.get("mae_ndvi_overall", float("nan")))
    v2_cov = float(m2.get("interval_calibration", {}).get("coverage_p10_p90_after_target_0.80", float("nan")))
    v2_base = float(v2_loo.get("mae_ndvi_baseline_rotation_mean_anom", float("nan")))
    final_mae = float(eval_final.get("mae_deterministic", float("nan")))
    final_cov = float(eval_final.get("coverage_p10_p90", float("nan")))
    final_base = float(eval_final.get("mae_baseline_fair_no_weather", float("nan")))

    comparison_rows = [
        {
            "name": "v1 LightGBM (fixed split 2022+ test)",
            "mae": v1_mae,
            "coverage": v1_cov,
            "beats": _beats_baseline(v1_mae, v1_base),
        },
        {
            "name": "v2 year+LightGBM (LOO 2014–25)",
            "mae": v2_mae,
            "coverage": v2_cov,
            "beats": _beats_baseline(v2_mae, v2_base),
        },
        {
            "name": "final year+empirical (LOO 2014–25)",
            "mae": final_mae,
            "coverage": final_cov,
            "beats": _beats_baseline(final_mae, final_base),
        },
    ]
    _write_comparison_md(comparison_rows, eval_final, COMPARISON_MD)

    print("Final model — rotation stats:")
    for rot in ROTATIONS:
        s = rot_stats[rot]
        print(
            f"  {rot}: n={s['n']}, median anom={s.get('anomaly_median')}, "
            f"confidence={s.get('confidence')}"
        )
    bb = eval_final["before_bugfix_loo"]
    print("\n--- LOO before/after bugfix ---")
    print(f"BEFORE: MAE {bb['mae_mc_p50_wrong']:.4f} | coverage {bb['coverage_p10_p90_wrong']:.3f}")
    print(
        f"AFTER:  MAE det {eval_final['mae_deterministic']:.4f} | MC p50 {eval_final['mae_mc_p50']:.4f} "
        f"| coverage {eval_final['coverage_p10_p90']:.3f}"
    )
    print(f"\nYear LOO MAE (county mean): {loo_mae:.4f}")
    print(f"Final LOO MAE deterministic: {final_mae:.4f} | MC p50: {eval_final.get('mae_mc_p50'):.4f}")
    print(f"Fair baseline (no weather): {final_base:.4f} | rotation-mean NDVI: {eval_final.get('mae_baseline_rotation_mean_ndvi'):.4f}")
    print(f"Beats fair baseline: {eval_final.get('beats_baseline_fair')}")
    print(f"LOO coverage before conformal: {eval_final.get('coverage_p10_p90_before_conformal'):.3f} | after: {final_cov:.3f}")
    print(f"Conformal delta: {eval_final.get('conformal_delta'):.4f}")
    print("\nPer-year LOO (deterministic):")
    print(f"{'year':>6} {'county_obs':>10} {'year_pred':>10} {'point_MAE':>10}")
    for row in eval_final.get("per_year", []):
        print(
            f"{row['year']:6d} {row['county_mean_observed']:10.4f} "
            f"{row['year_model_loo_pred']:10.4f} {row['mae_points_deterministic']:10.4f}"
        )
    print(f"Wrote {FINAL_JSON}")
    print(f"Wrote {COMPARISON_MD}")
    print(f"Wrote {ROTATION_PNG}")
    return payload


def self_test() -> None:
    rng = np.random.default_rng(0)
    yearly = pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2021, 2022],
            "county_mean_ndvi": rng.uniform(0.5, 0.7, 5),
            "season_rain_mm": rng.uniform(200, 400, 5),
            "heat_days": rng.integers(10, 30, 5),
        }
    )
    params = _year_params(_fit_year_pipe(yearly))
    resids, _, _ = _loo_year_residuals(yearly)
    assert len(resids) >= 3
    p10, p50, p90 = monte_carlo_quantiles(
        _predict_year_mu(params, 300, 20), resids, [-0.05, 0.0, 0.05], n_draws=2000, rng=rng
    )
    assert p10 <= p50 <= p90
    print("06_train_final self_test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    run_final()