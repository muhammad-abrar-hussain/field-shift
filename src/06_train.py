# Train LightGBM quantile models for peak NDVI (CURSOR_CONTEXT §8).

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
import shap
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_PARQUET = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_JSON = REPORTS_DIR / "model_metrics.json"
SHAP_PNG = REPORTS_DIR / "shap_summary.png"

TRAIN_YEAR_MAX = 2021
TEST_YEAR_MIN = 2022
TARGET = "peak_ndvi"

CAT_FEATURES = ["rotation_context", "prev1_crop", "prev2_crop"]
NUM_FEATURES = [
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


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cats = [c for c in CAT_FEATURES if c in df.columns]
    nums = [c for c in NUM_FEATURES if c in df.columns]
    return cats + nums


def _prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy()
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df[TARGET].astype(float)
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    return X, y


def _baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby("rotation_context")[TARGET].mean()
    return test["rotation_context"].map(means).fillna(train[TARGET].mean()).to_numpy()


def _sort_quantiles(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.sort(np.column_stack([p10, p50, p90]), axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def _interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def train_models(
    table: pd.DataFrame,
    *,
    models_dir: Path,
    reports_dir: Path,
) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    features = _feature_columns(table)
    train = table[table["year"] <= TRAIN_YEAR_MAX].copy()
    test = table[table["year"] >= TEST_YEAR_MIN].copy()

    X_train, y_train = _prepare_xy(train, features)
    X_test, y_test = _prepare_xy(test, features)

    metrics: dict = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": features,
        "train_counts_rotation_context": train["rotation_context"].value_counts().to_dict(),
        "test_counts_rotation_context": test["rotation_context"].value_counts().to_dict(),
    }

    if len(X_train) < 20 or len(X_test) < 1:
        metrics["status"] = "insufficient_data"
        return metrics

    base_pred = _baseline_predict(train, test)
    metrics["mae_baseline"] = float(mean_absolute_error(y_test, base_pred))

    common = {
        "verbosity": -1,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
    }

    m50 = lgb.LGBMRegressor(objective="regression", **common)
    m50.fit(X_train, y_train, categorical_feature=[c for c in CAT_FEATURES if c in features])

    m10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **common)
    m10.fit(X_train, y_train, categorical_feature=[c for c in CAT_FEATURES if c in features])

    m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **common)
    m90.fit(X_train, y_train, categorical_feature=[c for c in CAT_FEATURES if c in features])

    joblib.dump(m10, models_dir / "model_p10.pkl")
    joblib.dump(m50, models_dir / "model_p50.pkl")
    joblib.dump(m90, models_dir / "model_p90.pkl")

    p10, p50, p90 = m10.predict(X_test), m50.predict(X_test), m90.predict(X_test)
    p10, p50, p90 = _sort_quantiles(p10, p50, p90)

    metrics["mae_model_p50"] = float(mean_absolute_error(y_test, p50))
    metrics["r2_model_p50"] = float(r2_score(y_test, p50))
    metrics["interval_coverage_p10_p90"] = _interval_coverage(y_test.to_numpy(), p10, p90)

    explainer = shap.TreeExplainer(m50)
    sample = X_train.sample(n=min(300, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    metrics["status"] = "ok"
    metrics["shap_plot"] = str(SHAP_PNG)
    return metrics


def _print_metrics_table(metrics: dict) -> None:
    rows = [
        ("Train rows", metrics.get("n_train")),
        ("Test rows", metrics.get("n_test")),
        ("MAE baseline", metrics.get("mae_baseline")),
        ("MAE model (p50)", metrics.get("mae_model_p50")),
        ("R² (p50)", metrics.get("r2_model_p50")),
        ("p10–p90 coverage", metrics.get("interval_coverage_p10_p90")),
    ]
    print(f"{'Metric':<22} {'Value'}")
    print("-" * 36)
    for name, val in rows:
        print(f"{name:<22} {val}")


def run_train() -> int:
    if not TABLE_PARQUET.is_file():
        print(f"Missing {TABLE_PARQUET} — run src/05_build_table.py first.")
        return 1

    table = pd.read_parquet(TABLE_PARQUET)
    metrics = train_models(table, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _print_metrics_table(metrics)
    print(f"\nWrote models to {MODELS_DIR}")
    print(f"Wrote {METRICS_JSON}")
    if metrics.get("r2_model_p50", 0) > 0.95:
        print("Warning: test R² > 0.95 — check for leakage.")
    if metrics.get("mae_model_p50", 1e9) > metrics.get("mae_baseline", 1e9):
        print("Note: model MAE did not beat baseline on test.")
    return 0 if metrics.get("status") == "ok" else 1


def self_test() -> None:
    rng = np.random.default_rng(0)
    n = 120
    years = rng.choice([2018, 2019, 2020, 2021, 2022, 2023], size=n)
    df = pd.DataFrame(
        {
            "year": years,
            "rotation_context": rng.choice(["A", "B", "other"], size=n),
            "prev1_crop": "fallow",
            "prev2_crop": "winter_wheat",
            "peak_ndvi": rng.uniform(0.2, 0.7, size=n),
            "season_rain_mm": rng.uniform(100, 500, size=n),
            "heat_days": rng.integers(0, 30, size=n),
            "frost_days": rng.integers(0, 10, size=n),
            "et0_mm": rng.uniform(400, 900, size=n),
            "water_balance_mm": rng.uniform(-600, 100, size=n),
            "sm_planting": rng.uniform(0.1, 0.35, size=n),
            "slope_r": rng.uniform(0, 5, size=n),
            "elevation_m": rng.uniform(850, 950, size=n),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        mdir = Path(tmp) / "models"
        rdir = Path(tmp) / "reports"
        metrics = train_models(df, models_dir=mdir, reports_dir=rdir)
        assert metrics["status"] == "ok"
        assert (mdir / "model_p50.pkl").is_file()
    print("06_train self_test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    sys.exit(run_train())