# Sanity-check county mean peak NDVI vs NASS winter wheat yield (CURSOR_CONTEXT §8).

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_V2 = PROJECT_ROOT / "data" / "processed" / "model_table_v2.parquet"
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
NASS_CSV = PROJECT_ROOT / "data" / "raw" / "nass" / "nass_finney_wheat_yield.csv"
FINAL_JSON = PROJECT_ROOT / "models" / "final_model.json"
METRICS_V2 = PROJECT_ROOT / "reports" / "model_metrics_v2.json"
OUT_PNG = PROJECT_ROOT / "reports" / "nass_validation.png"

NASS_YEARS = {2013, 2014, 2015, 2020, 2021, 2022}
DISCLAIMER = (
    "n is only 6 overlapping years — a sanity check, not proof of predictive accuracy."
)


def _load_nass() -> pd.DataFrame:
    df = pd.read_csv(NASS_CSV)
    out = df[df["Data Item"].str.contains("YIELD", case=False, na=False)].copy()
    out["year"] = out["Year"].astype(int)
    out["yield_bu_ac"] = pd.to_numeric(out["Value"], errors="coerce")
    return out[["year", "yield_bu_ac"]].dropna()


def _county_means_observed() -> pd.DataFrame:
    if not TABLE_V2.is_file():
        raise FileNotFoundError(TABLE_V2)
    t = pd.read_parquet(TABLE_V2)
    t = t[t["purity"] >= 0.7]
    return t.groupby("year", as_index=False).agg(county_mean_ndvi=("peak_ndvi", "mean"))


def _predicted_county_means(final: dict) -> pd.DataFrame:
    w = pd.read_csv(WEATHER_CSV).rename(columns={"harvest_year": "year"})
    ym = final["year_model"]
    rows = []
    for _, row in w.iterrows():
        y = int(row["year"])
        rain, heat = float(row["season_rain_mm"]), float(row["heat_days"])
        mu = float(ym["intercept"])
        vals = {"season_rain_mm": rain, "heat_days": heat}
        for feat in ym["features"]:
            z = (vals[feat] - ym["scaler_mean"][feat]) / ym["scaler_std"][feat]
            mu += ym["coefficients"][feat] * z
        lo, hi = ym.get("ndvi_clip", [0.4, 0.8])
        rows.append({"year": y, "county_mean_ndvi_pred": float(np.clip(mu, lo, hi))})
    return pd.DataFrame(rows)


def _pearson(xs: np.ndarray, ys: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return float("nan"), int(len(xs))
    r, _ = stats.pearsonr(xs, ys)
    return float(r), int(len(xs))


def _plot_panel(ax, years, x, y, xlabel: str, title_suffix: str, r: float) -> None:
    ax.scatter(x, y, c="#2c7bb6", s=60, zorder=3)
    for yr, xi, yi in zip(years, x, y):
        ax.annotate(str(yr), (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)
    if len(x) >= 2 and np.std(x) > 0:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coef, xs), color="#d7191c", alpha=0.7, lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NASS winter wheat yield (bu/ac)")
    ax.set_title(f"{title_suffix} (r = {r:.2f}, n = {len(x)})")
    ax.grid(True, alpha=0.3)


def run_validate() -> dict:
    if not FINAL_JSON.is_file():
        raise FileNotFoundError("Run src/06_train_final.py first.")
    final = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
    nass = _load_nass()
    obs = _county_means_observed()
    pred = _predicted_county_means(final)

    base = nass.merge(obs, on="year", how="inner")
    base = base.merge(pred, on="year", how="left")
    base = base[base["year"].isin(NASS_YEARS)].sort_values("year")

    r_obs, n_obs = _pearson(base["county_mean_ndvi"].to_numpy(), base["yield_bu_ac"].to_numpy())
    r_pred, n_pred = _pearson(base["county_mean_ndvi_pred"].to_numpy(), base["yield_bu_ac"].to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _plot_panel(
        axes[0],
        base["year"].tolist(),
        base["county_mean_ndvi"].to_numpy(),
        base["yield_bu_ac"].to_numpy(),
        "Observed county mean peak NDVI",
        "Observed NDVI vs yield",
        r_obs,
    )
    _plot_panel(
        axes[1],
        base["year"].tolist(),
        base["county_mean_ndvi_pred"].to_numpy(),
        base["yield_bu_ac"].to_numpy(),
        "Predicted county mean peak NDVI",
        "Model NDVI vs yield",
        r_pred,
    )
    fig.suptitle("Satellite crop health tracks real wheat yield", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Source: NASA MODIS MOD13Q1 (peak NDVI) and USDA NASS QuickStats county yield. "
        + DISCLAIMER,
        ha="center",
        fontsize=9,
        wrap=True,
    )
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    nass_block = {
        "disclaimer": DISCLAIMER,
        "n_years": n_obs,
        "years": base["year"].astype(int).tolist(),
        "pearson_r_observed_county_ndvi_vs_yield": r_obs,
        "pearson_r_predicted_county_ndvi_vs_yield": r_pred,
        "observed_ndvi_by_year": dict(zip(base["year"].astype(int), base["county_mean_ndvi"].round(4))),
        "predicted_ndvi_by_year": dict(zip(base["year"].astype(int), base["county_mean_ndvi_pred"].round(4))),
        "nass_yield_bu_ac_by_year": dict(zip(base["year"].astype(int), base["yield_bu_ac"].round(2))),
        "plot": str(OUT_PNG),
    }

    final["nass_validation"] = nass_block
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    m2 = {}
    if METRICS_V2.is_file():
        m2 = json.loads(METRICS_V2.read_text(encoding="utf-8"))
    m2["nass_validation"] = nass_block
    with open(METRICS_V2, "w", encoding="utf-8") as f:
        json.dump(m2, f, indent=2)

    print(DISCLAIMER)
    print(f"Years: {nass_block['years']}")
    print(f"Pearson r (observed county NDVI vs yield): {r_obs:.3f} (n={n_obs})")
    print(f"Pearson r (predicted county NDVI vs yield): {r_pred:.3f} (n={n_pred})")
    print(f"Wrote {OUT_PNG}")
    return nass_block


if __name__ == "__main__":
    run_validate()
    sys.exit(0)