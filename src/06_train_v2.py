# V2 training: field purity filter, year + field decomposition, LOO CV, conformal intervals.

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_V1 = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"
TABLE_V2 = PROJECT_ROOT / "data" / "processed" / "model_table_v2.parquet"
CROP_HISTORY = PROJECT_ROOT / "data" / "processed" / "crop_history.csv"
CDL_DIR = PROJECT_ROOT / "data" / "raw" / "cdl"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_V1 = REPORTS_DIR / "model_metrics.json"
METRICS_V2 = REPORTS_DIR / "model_metrics_v2.json"
SHAP_V2 = REPORTS_DIR / "shap_field_v2.png"

REF_YEAR = 2024
FIPS_SUFFIX = "20055"
WHEAT = {24, 236}
LOO_YEAR_MIN = 2014
LOO_YEAR_MAX = 2025
TRAIN_YEAR_MAX = 2021
TEST_YEAR_MIN = 2022
TARGET = "peak_ndvi"

FIELD_CAT = ["rotation_context", "prev1_crop", "prev2_crop"]
FIELD_NUM = [
    "sm_planting",
    "prev_year_bare_days",
    "om_pct",
    "sand_pct",
    "clay_pct",
    "awc",
    "kwfact",
    "slope_r",
    "elevation_m",
]
YEAR_FEATURES = ["season_rain_mm", "heat_days"]


def _cdl_path(year: int) -> Path:
    return CDL_DIR / f"CDL_{year}_{FIPS_SUFFIX}.tif"


def _load_reference_meta() -> tuple[rasterio.Affine, rasterio.crs.CRS, int, int]:
    with rasterio.open(_cdl_path(REF_YEAR)) as ref:
        return ref.transform, ref.crs, ref.height, ref.width


def _read_aligned_year(year: int, ref_transform, ref_crs, height: int, width: int) -> np.ndarray:
    from rasterio.warp import Resampling, reproject

    with rasterio.open(_cdl_path(year)) as src:
        if src.height == height and src.width == width and src.transform == ref_transform:
            return src.read(1)
        out = np.zeros((height, width), dtype=np.uint16)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )
        return out


def _point_rowcol(history: pd.DataFrame, transform: rasterio.Affine) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for _, rec in history.iterrows():
        pid = int(rec["point_id"])
        row, col = rasterio.transform.rowcol(transform, float(rec["x"]), float(rec["y"]))
        out[pid] = (int(row), int(col))
    return out


def _window_purity(cdl: np.ndarray, row: int, col: int) -> float:
    h, w = cdl.shape
    r0, r1 = max(0, row - 4), min(h, row + 5)
    c0, c1 = max(0, col - 4), min(w, col + 5)
    window = cdl[r0:r1, c0:c1]
    if window.size == 0:
        return float("nan")
    return float(np.isin(window, list(WHEAT)).mean())


def add_purity(table: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    ref_transform, ref_crs, height, width = _load_reference_meta()
    rc = _point_rowcol(history, ref_transform)
    cdl_cache: dict[int, np.ndarray] = {}

    purities: list[float] = []
    for _, row in table.iterrows():
        pid = int(row["point_id"])
        year = int(row["year"])
        if year not in cdl_cache:
            cdl_cache[year] = _read_aligned_year(year, ref_transform, ref_crs, height, width)
        cdl = cdl_cache[year]
        r, c = rc.get(pid, (0, 0))
        purities.append(_window_purity(cdl, r, c))

    out = table.copy()
    out["purity"] = purities
    return out


def _rotation_counts(df: pd.DataFrame, label: str) -> dict:
    counts = {str(k): int(v) for k, v in df["rotation_context"].value_counts().items()}
    counts["n_total"] = int(len(df))
    counts["_label"] = label
    return counts


def _choose_purity_threshold(counts_07: dict, counts_05: dict) -> float:
    a07 = counts_07.get("A", 0)
    b07 = counts_07.get("B", 0)
    if a07 >= 300 and b07 >= 300:
        return 0.7
    return 0.5


def _field_features(df: pd.DataFrame) -> list[str]:
    cats = [c for c in FIELD_CAT if c in df.columns]
    nums = [c for c in FIELD_NUM if c in df.columns]
    return cats + nums


def _prepare_field_xy(df: pd.DataFrame, features: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy()
    for c in FIELD_CAT:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df[target_col].astype(float)
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def _lgb_common() -> dict:
    return {
        "verbosity": -1,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
    }


def _fit_field_quantiles(X: pd.DataFrame, y: pd.Series) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, lgb.LGBMRegressor]:
    common = _lgb_common()
    cats = [c for c in FIELD_CAT if c in X.columns]
    m50 = lgb.LGBMRegressor(objective="regression", **common)
    m10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **common)
    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **common)
    for m in (m10, m50, m90):
        m.fit(X, y, categorical_feature=cats)
    return m10, m50, m90


def _sort_q(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.sort(np.column_stack([p10, p50, p90]), axis=1)
    return s[:, 0], s[:, 1], s[:, 2]


def _yearly_means(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("year", as_index=False)
        .agg(
            county_mean_ndvi=(TARGET, "mean"),
            season_rain_mm=("season_rain_mm", "first"),
            heat_days=("heat_days", "first"),
        )
        .sort_values("year")
    )
    return g


def _fit_year_model(yearly: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )
    X = yearly[YEAR_FEATURES]
    y = yearly["county_mean_ndvi"]
    pipe.fit(X, y)
    return pipe


def _year_loo_mae(yearly: pd.DataFrame) -> tuple[float, list[dict]]:
    rows = []
    for hold in yearly["year"]:
        train = yearly[yearly["year"] != hold]
        if len(train) < 3:
            continue
        pipe = _fit_year_model(train)
        pred = float(pipe.predict(yearly.loc[yearly["year"] == hold, YEAR_FEATURES])[0])
        obs = float(yearly.loc[yearly["year"] == hold, "county_mean_ndvi"].iloc[0])
        rows.append({"year": int(hold), "obs": obs, "pred": pred, "abs_err": abs(obs - pred)})
    mae = float(np.mean([r["abs_err"] for r in rows])) if rows else float("nan")
    return mae, rows


def _add_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    county = df.groupby("year")[TARGET].transform("mean")
    out = df.copy()
    out["ndvi_anom"] = out[TARGET] - county
    out["county_mean_ndvi"] = county
    return out


def _anom_baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby("rotation_context")["ndvi_anom"].mean()
    return test["rotation_context"].map(means).fillna(train["ndvi_anom"].mean()).to_numpy()


def _interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def _conformal_widen(
    y: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    target_cov: float = 0.80,
) -> tuple[float, float, float]:
    """Symmetric widen of [p10,p90] so ~target_cov OOF rows are covered."""
    del p50  # reserved for alternate calibrations
    n = len(y)
    if n == 0:
        return 0.0, 0.0, 0.0
    need = np.maximum(p10 - y, y - p90)
    need = np.maximum(need, 0.0)
    order = np.sort(need)
    idx = int(np.ceil(target_cov * n)) - 1
    idx = min(max(idx, 0), n - 1)
    delta = float(order[idx])
    cov_before = _interval_coverage(y, p10, p90)
    cov_after = _interval_coverage(y, p10 - delta, p90 + delta)
    return delta, cov_before, cov_after


def _v1_split_metrics(df: pd.DataFrame, features_v1: list[str]) -> dict:
    """Replay v1-style direct NDVI model on filtered v2 table for comparison."""
    train = df[df["year"] <= TRAIN_YEAR_MAX]
    test = df[df["year"] >= TEST_YEAR_MIN]
    if train.empty or test.empty:
        return {"status": "insufficient_data"}

    X_train = train[features_v1].copy()
    X_test = test[features_v1].copy()
    for c in ("rotation_context", "prev1_crop", "prev2_crop"):
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    means = train.groupby("rotation_context")[TARGET].mean()
    base = test["rotation_context"].map(means).fillna(train[TARGET].mean()).to_numpy()

    m50 = lgb.LGBMRegressor(objective="regression", **_lgb_common())
    m10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **_lgb_common())
    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **_lgb_common())
    cats = [c for c in ("rotation_context", "prev1_crop", "prev2_crop") if c in features_v1]
    for m in (m10, m50, m90):
        m.fit(X_train, y_train, categorical_feature=cats)

    p10, p50, p90 = m10.predict(X_test), m50.predict(X_test), m90.predict(X_test)
    p10, p50, p90 = _sort_q(p10, p50, p90)

    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_baseline": float(mean_absolute_error(y_test, base)),
        "mae_model_p50": float(mean_absolute_error(y_test, p50)),
        "r2_model_p50": float(r2_score(y_test, p50)),
        "interval_coverage_p10_p90": _interval_coverage(y_test.to_numpy(), p10, p90),
    }


def train_v2() -> dict:
    if not TABLE_V1.is_file():
        raise FileNotFoundError(f"Missing {TABLE_V1}")
    if not CROP_HISTORY.is_file():
        raise FileNotFoundError(f"Missing {CROP_HISTORY}")

    table = pd.read_parquet(TABLE_V1)
    history = pd.read_csv(CROP_HISTORY)
    table = add_purity(table, history)

    counts_all = _rotation_counts(table, "all")
    hi07 = table[table["purity"] >= 0.7]
    hi05 = table[table["purity"] >= 0.5]
    counts_07 = _rotation_counts(hi07, "purity>=0.7")
    counts_05 = _rotation_counts(hi05, "purity>=0.5")
    threshold = _choose_purity_threshold(counts_07, counts_05)
    filtered = table[table["purity"] >= threshold].copy()
    filtered = filtered[(filtered["year"] >= LOO_YEAR_MIN) & (filtered["year"] <= LOO_YEAR_MAX)]

    TABLE_V2.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE_V2, index=False)

    filtered = _add_anomaly(filtered)
    yearly = _yearly_means(filtered)
    year_model = _fit_year_model(yearly)
    year_loo_mae, year_loo_rows = _year_loo_mae(yearly)

    coef = dict(
        zip(
            ["intercept"] + YEAR_FEATURES,
            [float(year_model.named_steps["reg"].intercept_)]
            + [float(x) for x in year_model.named_steps["reg"].coef_],
        )
    )

    features = _field_features(filtered)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- LOO CV for combined v2 model on field anomaly + year effect ---
    years = sorted(filtered["year"].unique())
    oof_records: list[dict] = []

    for hold_year in years:
        train_df = filtered[filtered["year"] != hold_year]
        test_df = filtered[filtered["year"] == hold_year]
        if train_df.empty or test_df.empty:
            continue

        y_train_yearly = _yearly_means(train_df)
        if len(y_train_yearly) < 3:
            continue
        year_fold = _fit_year_model(y_train_yearly)

        X_tr, y_tr = _prepare_field_xy(train_df, features, "ndvi_anom")
        X_te, y_te = _prepare_field_xy(test_df, features, "ndvi_anom")
        if len(X_tr) < 30 or len(X_te) < 1:
            continue

        m10, m50, m90 = _fit_field_quantiles(X_tr, y_tr)
        fp10, fp50, fp90 = m10.predict(X_te), m50.predict(X_te), m90.predict(X_te)
        fp10, fp50, fp90 = _sort_q(fp10, fp50, fp90)

        y_mean_pred = year_fold.predict(test_df[YEAR_FEATURES]).astype(float)

        pred_ndvi = y_mean_pred + fp50
        pred_p10 = y_mean_pred + fp10
        pred_p90 = y_mean_pred + fp90
        pred_p10, _, pred_p90 = _sort_q(pred_p10, pred_ndvi, pred_p90)

        base_anom = _anom_baseline_predict(train_df, test_df)
        county_obs = test_df["county_mean_ndvi"].to_numpy()
        y_obs = test_df[TARGET].to_numpy()

        for i in range(len(test_df)):
            oof_records.append(
                {
                    "year": int(hold_year),
                    "point_id": int(test_df.iloc[i]["point_id"]),
                    "rotation_context": str(test_df.iloc[i]["rotation_context"]),
                    "y": float(y_obs[i]),
                    "y_pred": float(pred_ndvi[i]),
                    "y_anom": float(y_te.iloc[i]),
                    "y_anom_pred": float(fp50[i]),
                    "y_anom_baseline": float(base_anom[i]),
                    "y_ndvi_rotation_baseline": float(base_anom[i] + county_obs[i]),
                    "p10": float(pred_p10[i]),
                    "p90": float(pred_p90[i]),
                    "county_mean_pred": float(y_mean_pred[i]),
                }
            )

    oof = pd.DataFrame(oof_records)
    metrics: dict = {
        "purity_threshold_chosen": threshold,
        "purity_threshold_rule": "0.7 if A and B both >=300 rows else 0.5",
        "rows_per_rotation_context": {
            "all": {k: v for k, v in counts_all.items() if not str(k).startswith("_")},
            "purity_gte_0.7": {k: v for k, v in counts_07.items() if not str(k).startswith("_")},
            "purity_gte_0.5": {k: v for k, v in counts_05.items() if not str(k).startswith("_")},
            "after_purity_filter_used_for_train": filtered["rotation_context"].value_counts().to_dict(),
        },
        "n_rows_filtered_train_pool": int(len(filtered)),
        "model_table_v2": str(TABLE_V2),
        "year_model": {
            "features": YEAR_FEATURES,
            "coefficients": coef,
            "loo_mae_county_mean_ndvi": year_loo_mae,
            "loo_fold_details": year_loo_rows,
            "n_years_in_year_model": int(len(yearly)),
        },
    }

    if not oof.empty:
        metrics["loo_cv_field_combined"] = {
            "years": f"{LOO_YEAR_MIN}-{LOO_YEAR_MAX}",
            "n_oof_rows": int(len(oof)),
            "mae_ndvi_overall": float(mean_absolute_error(oof["y"], oof["y_pred"])),
            "mae_ndvi_baseline_rotation_mean_anom": float(
                mean_absolute_error(oof["y"], oof["y_ndvi_rotation_baseline"])
            ),
            "mae_anom_overall": float(mean_absolute_error(oof["y_anom"], oof["y_anom_pred"])),
            "mae_anom_baseline_rotation_loo": float(
                mean_absolute_error(oof["y_anom"], oof["y_anom_baseline"])
            ),
            "mae_anom_by_year": {
                int(y): float(mean_absolute_error(g["y_anom"], g["y_anom_pred"]))
                for y, g in oof.groupby("year")
            },
            "mae_ndvi_by_year": {
                int(y): float(mean_absolute_error(g["y"], g["y_pred"])) for y, g in oof.groupby("year")
            },
        }

        p50_oof = oof["y_pred"].to_numpy()
        delta, cov_b, cov_a = _conformal_widen(
            oof["y"].to_numpy(), oof["p10"].to_numpy(), p50_oof, oof["p90"].to_numpy()
        )
        interval = {"delta_symmetric": delta, "coverage_p10_p90_before": cov_b, "coverage_p10_p90_after_target_0.80": cov_a}
        metrics["interval_calibration"] = interval
        with open(MODELS_DIR / "interval_adjust_v2.json", "w", encoding="utf-8") as f:
            json.dump(interval, f, indent=2)

    # --- Final models on all filtered data ---
    joblib.dump(year_model, MODELS_DIR / "year_model_v2.pkl")

    X_all, y_anom = _prepare_field_xy(filtered, features, "ndvi_anom")
    m10, m50, m90 = _fit_field_quantiles(X_all, y_anom)
    joblib.dump(m10, MODELS_DIR / "field_p10_v2.pkl")
    joblib.dump(m50, MODELS_DIR / "field_p50_v2.pkl")
    joblib.dump(m90, MODELS_DIR / "field_p90_v2.pkl")

    # Rotation effects
    rot_overall = (
        filtered.groupby("rotation_context")
        .agg(observed_anom_mean=("ndvi_anom", "mean"), observed_ndvi_mean=(TARGET, "mean"))
        .reset_index()
    )
    year_feat = filtered.groupby("year")[YEAR_FEATURES].first()
    year_eff = year_model.predict(year_feat)
    filtered = filtered.copy()
    filtered["year_effect_pred"] = filtered["year"].map(pd.Series(year_eff, index=year_feat.index))
    filtered["field_anom_pred"] = np.nan
    filtered.loc[X_all.index, "field_anom_pred"] = m50.predict(X_all)
    filtered["ndvi_pred_v2"] = filtered["year_effect_pred"] + filtered["field_anom_pred"]

    rot_overall = rot_overall.merge(
        filtered.groupby("rotation_context")
        .agg(predicted_anom_mean=("field_anom_pred", "mean"), predicted_ndvi_mean=("ndvi_pred_v2", "mean"))
        .reset_index(),
        on="rotation_context",
    )

    per_year = (
        filtered.groupby(["year", "rotation_context"])
        .agg(
            observed_anom_mean=("ndvi_anom", "mean"),
            predicted_anom_mean=("field_anom_pred", "mean"),
            n=("ndvi_anom", "size"),
        )
        .reset_index()
    )

    a_mask = filtered["rotation_context"].isin(["A", "B"])
    ab = filtered[a_mask].groupby("rotation_context").agg(
        observed_anom_mean=("ndvi_anom", "mean"),
        predicted_anom_mean=("field_anom_pred", "mean"),
    )
    if "A" in ab.index and "B" in ab.index:
        ab_diff = {
            "observed_A_minus_B_anom": float(ab.loc["A", "observed_anom_mean"] - ab.loc["B", "observed_anom_mean"]),
            "predicted_A_minus_B_anom": float(ab.loc["A", "predicted_anom_mean"] - ab.loc["B", "predicted_anom_mean"]),
        }
    else:
        ab_diff = {}

    metrics["rotation_effects"] = {
        "overall_by_rotation": rot_overall.to_dict(orient="records"),
        "per_year_by_rotation": per_year.to_dict(orient="records"),
        "A_vs_B_anomaly_difference": ab_diff,
    }

    # SHAP top 10
    explainer = shap.TreeExplainer(m50)
    sample = X_all.sample(n=min(400, len(X_all)), random_state=42)
    shv = explainer.shap_values(sample)
    mean_abs = np.abs(shv).mean(axis=0)
    order = np.argsort(-mean_abs)[:10]
    top_shap = [
        {"feature": features[i], "mean_abs_shap": float(mean_abs[i])} for i in order
    ]
    metrics["shap_field_p50_top10"] = top_shap

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shv, sample, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(SHAP_V2, dpi=150, bbox_inches="tight")
    plt.close()
    metrics["shap_plot"] = str(SHAP_V2)

    # V1 comparison on same filtered rows
    v1_features = [
        c
        for c in [
            "rotation_context",
            "prev1_crop",
            "prev2_crop",
            "fallow_rain_mm",
            "season_rain_mm",
            "heavy_rain_days",
            "heat_days",
            "frost_days",
            "et0_mm",
            "water_balance_mm",
            "mean_tmax_c",
            "sm_planting",
            "prev_year_bare_days",
            "om_pct",
            "sand_pct",
            "silt_pct",
            "clay_pct",
            "awc",
            "kwfact",
            "slope_r",
            "elevation_m",
        ]
        if c in filtered.columns
    ]
    metrics["v1_style_split_on_filtered_table"] = _v1_split_metrics(filtered, v1_features)
    if METRICS_V1.is_file():
        with open(METRICS_V1, encoding="utf-8") as f:
            metrics["v1_original_metrics_file"] = json.load(f)

    # Fixed split v2 combined
    tr = filtered[filtered["year"] <= TRAIN_YEAR_MAX]
    te = filtered[filtered["year"] >= TEST_YEAR_MIN]
    if not tr.empty and not te.empty:
        X_tr, y_tr = _prepare_field_xy(tr, features, "ndvi_anom")
        X_te, y_te = _prepare_field_xy(te, features, "ndvi_anom")
        _, m50s, _ = _fit_field_quantiles(X_tr, y_tr)
        fp = m50s.predict(X_te)
        y_year = year_model.predict(te[YEAR_FEATURES])
        pred = y_year + fp
        y_obs = te[TARGET].to_numpy()
        metrics["v2_fixed_split_2022plus"] = {
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "mae_ndvi": float(mean_absolute_error(y_obs, pred)),
            "r2_ndvi": float(r2_score(y_obs, pred)),
        }

    metrics["status"] = "ok"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_V2, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _print_summary(metrics: dict) -> None:
    print(f"\nPurity threshold used: {metrics.get('purity_threshold_chosen')}")
    print("Rows per rotation_context (after filter):")
    rc = metrics.get("rows_per_rotation_context", {}).get("after_purity_filter_used_for_train", {})
    for k in ("A", "B", "C", "other"):
        print(f"  {k}: {rc.get(k, 0)}")

    ym = metrics.get("year_model", {})
    print("\nYear model coefficients:")
    for k, v in ym.get("coefficients", {}).items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  LOO MAE (county mean NDVI): {ym.get('loo_mae_county_mean_ndvi'):.4f}")

    loo = metrics.get("loo_cv_field_combined", {})
    ic = metrics.get("interval_calibration", {})
    rows = [
        ("OOF rows", loo.get("n_oof_rows")),
        ("MAE NDVI v2 LOO", loo.get("mae_ndvi_overall")),
        ("MAE NDVI baseline (rot)", loo.get("mae_ndvi_baseline_rotation_mean_anom")),
        ("MAE anomaly v2 LOO", loo.get("mae_anom_overall")),
        ("MAE anomaly baseline LOO", loo.get("mae_anom_baseline_rotation_loo")),
        ("Interval cov before", ic.get("coverage_p10_p90_before")),
        ("Interval cov after", ic.get("coverage_p10_p90_after_target_0.80")),
        ("Conformal delta", ic.get("delta_symmetric")),
    ]
    v1f = metrics.get("v1_style_split_on_filtered_table", {})
    v1o = metrics.get("v1_original_metrics_file", {})
    rows.extend(
        [
            ("V1 split MAE (filtered table)", v1f.get("mae_model_p50")),
            ("V1 file MAE (original table)", v1o.get("mae_model_p50")),
            ("V2 fixed split MAE", metrics.get("v2_fixed_split_2022plus", {}).get("mae_ndvi")),
        ]
    )

    print(f"\n{'Metric':<32} {'Value'}")
    print("-" * 44)
    for name, val in rows:
        if isinstance(val, float):
            print(f"{name:<32} {val:.4f}")
        else:
            print(f"{name:<32} {val}")


def self_test() -> None:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "point_id": rng.integers(0, 10, size=n),
            "year": rng.choice([2018, 2019, 2020, 2021, 2022], size=n),
            "rotation_context": rng.choice(["A", "B", "other"], size=n),
            "prev1_crop": "fallow",
            "prev2_crop": "winter_wheat",
            "peak_ndvi": rng.uniform(0.3, 0.6, size=n),
            "purity": rng.uniform(0.5, 1.0, size=n),
            "season_rain_mm": rng.uniform(200, 400, size=n),
            "heat_days": rng.integers(5, 25, size=n),
            "sm_planting": rng.uniform(0.1, 0.3, size=n),
            "prev_year_bare_days": rng.uniform(50, 200, size=n),
            "om_pct": rng.uniform(1, 3, size=n),
            "sand_pct": rng.uniform(10, 40, size=n),
            "clay_pct": rng.uniform(20, 50, size=n),
            "awc": rng.uniform(0.1, 0.25, size=n),
            "kwfact": rng.uniform(0.2, 0.4, size=n),
            "slope_r": rng.uniform(0, 3, size=n),
            "elevation_m": rng.uniform(800, 900, size=n),
        }
    )
    df = _add_anomaly(df)
    yearly = _yearly_means(df)
    pipe = _fit_year_model(yearly)
    assert pipe is not None
    features = _field_features(df)
    X, y = _prepare_field_xy(df, features, "ndvi_anom")
    m10, m50, m90 = _fit_field_quantiles(X, y)
    assert m50.predict(X).shape[0] == len(X)
    delta, _, _ = _conformal_widen(y.to_numpy(), m10.predict(X), m50.predict(X), m90.predict(X))
    assert delta >= 0
    print("06_train_v2 self_test: ok")


def _print_purity_tables(metrics: dict) -> None:
    rpc = metrics.get("rows_per_rotation_context", {})
    for key in ("all", "purity_gte_0.7", "purity_gte_0.5"):
        block = rpc.get(key, {})
        print(f"\nRows per rotation_context ({key}):")
        for rot in ("A", "B", "C", "other"):
            print(f"  {rot}: {block.get(rot, 0)}")
        print(f"  total: {block.get('n_total', sum(block.get(r, 0) for r in ('A','B','C','other')))}")


def main() -> int:
    metrics = train_v2()
    _print_purity_tables(metrics)
    _print_summary(metrics)
    print(f"\nWrote {TABLE_V2}")
    print(f"Wrote {METRICS_V2}")
    ab = metrics.get("rotation_effects", {}).get("A_vs_B_anomaly_difference", {})
    if ab:
        print(
            f"A vs B anom: observed {ab.get('observed_A_minus_B_anom', float('nan')):.4f}, "
            f"predicted {ab.get('predicted_A_minus_B_anom', float('nan')):.4f}"
        )
    return 0 if metrics.get("status") == "ok" else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    sys.exit(main())