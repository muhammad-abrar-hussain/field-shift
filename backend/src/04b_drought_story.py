# Join growing-season POWER rain with NASS county wheat yields for presentation chart (CURSOR_CONTEXT §7).

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
NASS_CSV = PROJECT_ROOT / "data" / "raw" / "nass" / "nass_finney_wheat_yield.csv"
OUT_PNG = PROJECT_ROOT / "reports" / "drought_story.png"

YEAR_MIN = 2013
YEAR_MAX = 2025
DROUGHT_YEAR = 2022


def _load_weather() -> pd.DataFrame:
    w = pd.read_csv(WEATHER_CSV)
    w["harvest_year"] = w["harvest_year"].astype(int)
    return w[(w["harvest_year"] >= YEAR_MIN) & (w["harvest_year"] <= YEAR_MAX)]


def _load_nass_yields() -> pd.DataFrame:
    nass = pd.read_csv(NASS_CSV)
    nass["harvest_year"] = pd.to_numeric(nass["Year"], errors="coerce").astype("Int64")
    val = nass["Value"].astype(str).str.replace(",", "", regex=False)
    nass["wheat_yield_bu_acre"] = pd.to_numeric(val, errors="coerce")
    return nass[["harvest_year", "wheat_yield_bu_acre"]].dropna(subset=["harvest_year"])


def _pearson_rain_yield(merged: pd.DataFrame) -> float | None:
    sub = merged.dropna(subset=["season_rain_mm", "wheat_yield_bu_acre"])
    if len(sub) < 2:
        return None
    x = sub["season_rain_mm"].to_numpy(dtype=float)
    y = sub["wheat_yield_bu_acre"].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _plot(merged: pd.DataFrame) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    years = merged["harvest_year"].to_numpy()
    rain = merged["season_rain_mm"].to_numpy()
    yields = merged["wheat_yield_bu_acre"]

    bar_colors = ["#c0392b" if y == DROUGHT_YEAR else "#5b9bd5" for y in years]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(years, rain, color=bar_colors, width=0.72, alpha=0.9, label="Growing-season rain")
    ax1.set_xlabel("Winter wheat harvest year")
    ax1.set_ylabel("Growing-season rain (mm)", color="#2c5282")
    ax1.set_xticks(years)
    ax1.tick_params(axis="y", labelcolor="#2c5282")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    yield_years = merged.loc[yields.notna(), "harvest_year"]
    yield_vals = merged.loc[yields.notna(), "wheat_yield_bu_acre"]
    ax2.plot(
        yield_years,
        yield_vals,
        color="#e67e22",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="NASS winter wheat yield",
    )
    ax2.set_ylabel("County wheat yield (bu/acre)", color="#b35900")
    ax2.tick_params(axis="y", labelcolor="#b35900")

    drow = merged.loc[merged["harvest_year"] == DROUGHT_YEAR].iloc[0]
    ax1.annotate(
        "2022 drought: 166 mm rain,\nyield fell to 30.6 bu/acre",
        xy=(DROUGHT_YEAR, drow["season_rain_mm"]),
        xytext=(DROUGHT_YEAR + 2.2, drow["season_rain_mm"] + 120),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
    )

    ax1.set_title("Less rain, less wheat: Finney County, Kansas", fontsize=14, fontweight="bold", pad=14)
    fig.text(
        0.12,
        0.02,
        "Sources: NASA POWER (growing-season rain, county centroid); USDA NASS (county winter wheat yield)",
        fontsize=9,
        color="#555555",
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_PNG, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    weather = _load_weather()
    nass = _load_nass_yields()
    merged = weather.merge(nass, on="harvest_year", how="left")
    merged = merged.sort_values("harvest_year")

    r = _pearson_rain_yield(merged)
    _plot(merged)

    print("harvest_year  season_rain_mm  wheat_yield_bu_acre")
    for _, row in merged.iterrows():
        yld = row["wheat_yield_bu_acre"]
        yld_s = f"{yld:.1f}" if pd.notna(yld) else "—"
        print(f"  {int(row['harvest_year']):>4}        {row['season_rain_mm']:>7.2f}            {yld_s:>6}")

    if r is not None:
        n = merged["wheat_yield_bu_acre"].notna().sum()
        print(f"\nPearson r (season rain vs NASS yield, n={int(n)} years with yield): {r:.3f}")
    else:
        print("\nPearson r: not enough overlapping years.")

    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()