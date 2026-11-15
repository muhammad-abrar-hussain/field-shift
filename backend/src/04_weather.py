# Aggregate NASA POWER daily weather into harvest-year features for Finney County (see CURSOR_CONTEXT.md §8).

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWER_CSV = PROJECT_ROOT / "data" / "raw" / "power" / "power_finney_daily.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
OUT_PNG = PROJECT_ROOT / "reports" / "weather_by_year.png"

SKIP_HEADER_ROWS = 15
LATITUDE_DEG = 37.98
ELEVATION_M = 858.0
MISSING_VALUE = -999
HEAVY_RAIN_MM = 25.0
HEAT_TMAX_C = 32.0
FROST_TMIN_C = -2.0
HARVEST_YEARS = range(2013, 2026)


def _load_power() -> pd.DataFrame:
    df = pd.read_csv(POWER_CSV, skiprows=SKIP_HEADER_ROWS)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str), format="%Y%j")
    for col in ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= MISSING_VALUE, col] = np.nan
    return df.sort_values("date").reset_index(drop=True)


def _fallow_window(harvest_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=harvest_year - 2, month=7, day=1)
    end = pd.Timestamp(year=harvest_year - 1, month=9, day=30)
    return start, end


def _season_window(harvest_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=harvest_year - 1, month=10, day=1)
    end = pd.Timestamp(year=harvest_year, month=6, day=30)
    return start, end


def _between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _sum_et0_mm(season: pd.DataFrame) -> float | None:
    cols = ("T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN")
    valid = season.dropna(subset=list(cols)).set_index("date").sort_index()
    if valid.empty:
        return None
    et0 = pyet.pm_fao56(
        tmean=valid["T2M"],
        tmax=valid["T2M_MAX"],
        tmin=valid["T2M_MIN"],
        wind=valid["WS2M"],
        rs=valid["ALLSKY_SFC_SW_DWN"],
        rh=valid["RH2M"],
        elevation=ELEVATION_M,
        lat=np.deg2rad(LATITUDE_DEG),
    )
    return float(np.nansum(np.asarray(et0, dtype=float)))


def _harvest_features(df: pd.DataFrame, harvest_year: int) -> dict:
    fallow_start, fallow_end = _fallow_window(harvest_year)
    season_start, season_end = _season_window(harvest_year)
    fallow = _between(df, fallow_start, fallow_end)
    season = _between(df, season_start, season_end)

    heat = season[(season["date"] >= pd.Timestamp(harvest_year, 4, 1)) & (season["date"] <= season_end)]
    frost = season[
        (season["date"] >= pd.Timestamp(harvest_year, 3, 15))
        & (season["date"] <= pd.Timestamp(harvest_year, 4, 30))
    ]

    fallow_rain = fallow["PRECTOTCORR"].sum(skipna=True)
    season_rain = season["PRECTOTCORR"].sum(skipna=True)
    heavy = int((season["PRECTOTCORR"] > HEAVY_RAIN_MM).sum())
    heat_days = int((heat["T2M_MAX"] > HEAT_TMAX_C).sum())
    frost_days = int((frost["T2M_MIN"] < FROST_TMIN_C).sum())
    et0_mm = _sum_et0_mm(season)
    mean_tmax = season["T2M_MAX"].mean(skipna=True)

    water_balance = None
    if pd.notna(season_rain) and et0_mm is not None:
        water_balance = float(season_rain) - et0_mm

    return {
        "harvest_year": harvest_year,
        "fallow_rain_mm": round(float(fallow_rain), 2) if pd.notna(fallow_rain) else None,
        "season_rain_mm": round(float(season_rain), 2) if pd.notna(season_rain) else None,
        "heavy_rain_days": heavy,
        "heat_days": heat_days,
        "frost_days": frost_days,
        "et0_mm": round(et0_mm, 2) if et0_mm is not None else None,
        "water_balance_mm": round(water_balance, 2) if water_balance is not None else None,
        "mean_tmax_c": round(float(mean_tmax), 2) if pd.notna(mean_tmax) else None,
    }


def _plot_yearly(summary: pd.DataFrame) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = summary["harvest_year"].astype(int)
    ax1.bar(x, summary["season_rain_mm"], color="#4a90d9", label="season_rain_mm", alpha=0.85)
    ax1.set_xlabel("Harvest year")
    ax1.set_ylabel("Growing-season rain (mm)", color="#4a90d9")
    ax1.tick_params(axis="y", labelcolor="#4a90d9")

    ax2 = ax1.twinx()
    ax2.plot(x, summary["heat_days"], color="#d94a4a", marker="o", linewidth=2, label="heat_days")
    ax2.set_ylabel("Heat days (Tmax > 32 °C, Apr–Jun)", color="#d94a4a")
    ax2.tick_params(axis="y", labelcolor="#d94a4a")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Finney County — growing-season rain and heat days by harvest year")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    if not POWER_CSV.is_file():
        raise FileNotFoundError(POWER_CSV)

    daily = _load_power()
    rows = [_harvest_features(daily, y) for y in HARVEST_YEARS]
    summary = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    _plot_yearly(summary)

    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()