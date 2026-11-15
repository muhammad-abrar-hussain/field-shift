# Field calendar: monthly NDVI by rotation + county weather risk (CURSOR_CONTEXT §2, §9).

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NDVI_PARQUET = PROJECT_ROOT / "data" / "processed" / "ndvi_points.parquet"
CROP_HISTORY = PROJECT_ROOT / "data" / "processed" / "crop_history.csv"
POWER_CSV = PROJECT_ROOT / "data" / "raw" / "power" / "power_finney_daily.csv"
OUT_JSON = PROJECT_ROOT / "data" / "processed" / "field_calendar.json"
OUT_PNG = PROJECT_ROOT / "reports" / "field_calendar.png"
MONTHLY_CACHE = PROJECT_ROOT / "data" / "processed" / "field_calendar_monthly.parquet"

NDVI_YEARS = range(2013, 2026)
WEATHER_YEARS = range(2000, 2026)
MONTHS = list(range(1, 13))
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LOW_COVER_NDVI = 0.25
LOW_COVER_FOOTNOTE = (
    "Low-cover months use NDVI < 0.25 (not 0.2) because 250 m MODIS pixels blend bare stubble "
    "with neighbours, so fallow fields often read 0.2–0.3."
)
POWER_SKIP = 15
MISSING = -999
HEAT_TMAX_C = 32.0
FROST_TMIN_C = -2.0
HEAVY_RAIN_MM = 25.0
WHEAT_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5, 6}
HEAT_VULN = {4, 5, 6}
FROST_VULN = {3, 4}
SUMMER_MONTHS = {7, 8, 9}

SYSTEM_TO_ROT = {
    "wheat_fallow": "A",
    "wheat_sorghum_fallow": "B",
    "wheat_forage": "C",
}
ROTATIONS = ("A", "B", "C")


def _load_power_daily() -> pd.DataFrame:
    df = pd.read_csv(POWER_CSV, skiprows=POWER_SKIP)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str), format="%Y%j")
    for col in ("T2M_MAX", "T2M_MIN", "PRECTOTCORR"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= MISSING, col] = np.nan
    return df


def _daily_mean_ndvi_in_month(grp: pd.DataFrame, year: int, month: int) -> float | None:
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    sub = grp[(grp["date"] >= start) & (grp["date"] <= end)]
    if sub.empty:
        return None
    daily = (
        sub.groupby("date")["ndvi"]
        .mean()
        .reindex(pd.date_range(start, end, freq="D"))
        .interpolate(method="linear", limit_direction="both")
    )
    if daily.isna().all():
        return None
    return float(daily.mean())


def _monthly_mean_from_group(g: pd.DataFrame) -> float:
    year, month = int(g["year"].iloc[0]), int(g["month"].iloc[0])
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    daily = (
        g.groupby("date")["ndvi"]
        .mean()
        .reindex(pd.date_range(start, end, freq="D"))
        .interpolate(method="linear", limit_direction="both")
    )
    return float(daily.mean()) if not daily.isna().all() else np.nan


def _monthly_means_all_points(ndvi: pd.DataFrame) -> pd.DataFrame:
    """One row per point-year-month: interpolated daily mean NDVI."""
    ndvi = ndvi.copy()
    ndvi["year"] = ndvi["date"].dt.year
    ndvi["month"] = ndvi["date"].dt.month
    ndvi = ndvi[ndvi["year"].isin(NDVI_YEARS)]
    rows: list[dict] = []
    for (pid, year, month), g in ndvi.groupby(["point_id", "year", "month"], sort=False):
        mu = _monthly_mean_from_group(g)
        if not np.isnan(mu):
            rows.append({"point_id": int(pid), "year": int(year), "month": int(month), "ndvi_mean": mu})
    return pd.DataFrame(rows)


def _rotation_ndvi_stats(monthly: pd.DataFrame, rot_by_point: pd.Series) -> dict[str, dict[int, dict]]:
    """Per point: median NDVI across years for each calendar month; then mean/p25/p75 across points in rotation."""
    monthly = monthly.merge(rot_by_point.rename("rotation"), on="point_id", how="inner")
    monthly = monthly[monthly["rotation"].isin(ROTATIONS)]
    clim = monthly.groupby(["point_id", "rotation", "month"], as_index=False)["ndvi_mean"].median()
    out: dict[str, dict[int, dict]] = {r: {} for r in ROTATIONS}
    for rot in ROTATIONS:
        for month in MONTHS:
            vals = clim.loc[(clim["rotation"] == rot) & (clim["month"] == month), "ndvi_mean"]
            if vals.empty:
                continue
            out[rot][month] = {
                "mean": float(vals.mean()),
                "p25": float(vals.quantile(0.25)),
                "p75": float(vals.quantile(0.75)),
                "n": int(len(vals)),
            }
    return out


def _weather_by_month(power: pd.DataFrame) -> dict[int, dict]:
    power = power[power["date"].dt.year.isin(WEATHER_YEARS)].copy()
    power["month"] = power["date"].dt.month
    power["year"] = power["date"].dt.year
    rows: dict[int, dict] = {}
    for month in MONTHS:
        sub = power[power["month"] == month]
        per_year: list[dict] = []
        for year in sorted(sub["year"].unique()):
            y = sub[sub["year"] == year]
            per_year.append(
                {
                    "heat_days": int((y["T2M_MAX"] > HEAT_TMAX_C).sum()),
                    "frost_days": int((y["T2M_MIN"] < FROST_TMIN_C).sum()),
                    "heavy_rain_days": int((y["PRECTOTCORR"] > HEAVY_RAIN_MM).sum()),
                    "rain_mm": float(y["PRECTOTCORR"].sum()),
                }
            )
        if not per_year:
            continue
        rows[month] = {
            "month": month,
            "month_name": MONTH_NAMES[month - 1],
            "heat_days_mean": round(float(np.mean([r["heat_days"] for r in per_year])), 2),
            "frost_days_mean": round(float(np.mean([r["frost_days"] for r in per_year])), 2),
            "heavy_rain_days_mean": round(float(np.mean([r["heavy_rain_days"] for r in per_year])), 2),
            "rain_mm_mean": round(float(np.mean([r["rain_mm"] for r in per_year])), 1),
            "wheat_window": month in WHEAT_MONTHS,
            "vulnerable_heat_grain_fill": month in HEAT_VULN,
            "vulnerable_frost_greenup": month in FROST_VULN,
        }
    return rows


def _rotation_calendar(ndvi_stats: dict[str, dict[int, dict]], weather: dict[int, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rot in ROTATIONS:
        bare_months: list[int] = []
        months_detail = []
        mean_by_month: dict[str, float] = {}
        p25_by_month: dict[str, float] = {}
        p75_by_month: dict[str, float] = {}
        for m in MONTHS:
            st = ndvi_stats[rot].get(m)
            w = weather.get(m, {})
            if st:
                mean_by_month[str(m)] = round(st["mean"], 4)
                p25_by_month[str(m)] = round(st["p25"], 4)
                p75_by_month[str(m)] = round(st["p75"], 4)
                low = st["mean"] < LOW_COVER_NDVI
                if low:
                    bare_months.append(m)
                heavy = w.get("heavy_rain_days_mean") or 0.0
                erosion = low and heavy >= 1.0
            else:
                low = False
                erosion = False
            months_detail.append(
                {
                    "month": m,
                    "month_name": MONTH_NAMES[m - 1],
                    "ndvi_mean": None if not st else round(st["mean"], 4),
                    "ndvi_p25": None if not st else round(st["p25"], 4),
                    "ndvi_p75": None if not st else round(st["p75"], 4),
                    "is_low_cover_month": low,
                    "erosion_risk_heavy_rain_when_low_cover": erosion,
                }
            )
        out[rot] = {
            "rotation_id": rot,
            "mean_ndvi_by_month": mean_by_month,
            "ndvi_p25_by_month": p25_by_month,
            "ndvi_p75_by_month": p75_by_month,
            "bare_months": bare_months,
            "months": months_detail,
        }
    return out


def _plot_calendar(ndvi_stats: dict[str, dict[int, dict]], weather: dict[int, dict]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 4.8), sharex=True)
    fig.suptitle("When fields are bare and when wheat is at risk, Finney County", fontsize=12, y=1.02)

    all_means = [
        ndvi_stats[r][m]["mean"]
        for r in ROTATIONS
        for m in MONTHS
        if m in ndvi_stats[r]
    ]
    vmin, vmax = (min(all_means), max(all_means)) if all_means else (0.0, 0.8)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["YlGn"]

    for i, rot in enumerate(ROTATIONS):
        ax = axes[i]
        for j, m in enumerate(MONTHS):
            st = ndvi_stats[rot].get(m)
            if st:
                color = cmap(norm(st["mean"]))
                low = st["mean"] < LOW_COVER_NDVI
                edge = "#8b6914" if low else "white"
                lw = 2.5 if low else 0.5
            else:
                color = "#e0e0e0"
                edge = "white"
                lw = 0.5
            ax.add_patch(mpatches.Rectangle((j, 0), 1, 1, facecolor=color, edgecolor=edge, linewidth=lw))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(f"Rot {rot}", rotation=0, ha="right", va="center", fontweight="bold", fontsize=9)

    ax_w = axes[3]
    max_heat = max((weather[m]["heat_days_mean"] for m in MONTHS if m in weather), default=1.0) or 1.0
    max_frost = max((weather[m]["frost_days_mean"] for m in MONTHS if m in weather), default=1.0) or 1.0
    for j, m in enumerate(MONTHS):
        w = weather.get(m, {})
        heat_n = (w.get("heat_days_mean") or 0) / max_heat
        frost_n = (w.get("frost_days_mean") or 0) / max_frost
        if m in HEAT_VULN and heat_n > 0.4:
            c = "#e07a5f"
        elif m in FROST_VULN and frost_n > 0.2:
            c = "#81b0d4"
        elif (w.get("heavy_rain_days_mean") or 0) >= 2:
            c = "#3d6b8c"
        elif m in WHEAT_MONTHS:
            c = "#dde8f0"
        else:
            c = "#f0f0f0"
        ax_w.add_patch(mpatches.Rectangle((j, 0), 1, 1, facecolor=c, edgecolor="white", linewidth=0.5))
    ax_w.set_ylim(0, 1)
    ax_w.set_yticks([])
    ax_w.set_ylabel("Weather", rotation=0, ha="right", va="center", fontweight="bold")

    axes[-1].set_xticks(np.arange(12) + 0.5)
    axes[-1].set_xticklabels(MONTH_NAMES, fontsize=9)
    axes[-1].set_xlim(0, 12)
    fig.text(
        0.5,
        0.01,
        LOW_COVER_FOOTNOTE + " Low-cover months (mean NDVI < 0.25) have a dark outline.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.98))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summer_cover_comparison(ndvi_stats: dict[str, dict[int, dict]]) -> dict:
    """Jul–Sep mean NDVI: B higher than A means more summer green cover."""
    def summer_mean(rot: str) -> float | None:
        vals = [ndvi_stats[rot][m]["mean"] for m in SUMMER_MONTHS if m in ndvi_stats[rot]]
        return float(np.mean(vals)) if vals else None

    a, b = summer_mean("A"), summer_mean("B")
    if a is None or b is None:
        note = "Insufficient data for A/B summer comparison."
        higher = None
    else:
        higher = b > a
        diff = b - a
        note = (
            f"B Jul-Sep mean NDVI {b:.3f} vs A {a:.3f} (diff {diff:+.3f}); "
            f"{'B shows higher summer cover than A (sorghum/fallow mix).' if higher else 'B does not show higher summer cover than A.'}"
        )
    return {"A_jul_sep_mean_ndvi": a, "B_jul_sep_mean_ndvi": b, "B_higher_summer_cover_than_A": higher, "note": note}


def build_field_calendar() -> dict:
    if not NDVI_PARQUET.is_file():
        raise FileNotFoundError(f"Missing {NDVI_PARQUET}")
    if not CROP_HISTORY.is_file():
        raise FileNotFoundError(f"Missing {CROP_HISTORY}")
    if not POWER_CSV.is_file():
        raise FileNotFoundError(f"Missing {POWER_CSV}")

    hist = pd.read_csv(CROP_HISTORY)
    hist["rotation"] = hist["system"].map(SYSTEM_TO_ROT)
    points_by_rot = {
        r: hist.loc[hist["rotation"] == r, "point_id"].astype(int).tolist() for r in ROTATIONS
    }
    rot_by_point = hist.dropna(subset=["rotation"]).set_index("point_id")["rotation"].astype(str)

    force = "--rebuild-monthly" in sys.argv
    if MONTHLY_CACHE.is_file() and not force:
        monthly = pd.read_parquet(MONTHLY_CACHE)
        print(f"Loaded cached monthly NDVI from {MONTHLY_CACHE}", flush=True)
    else:
        ndvi = pd.read_parquet(NDVI_PARQUET)
        ndvi["date"] = pd.to_datetime(ndvi["date"])
        ndvi = ndvi[(ndvi["date"].dt.year >= min(NDVI_YEARS)) & (ndvi["date"].dt.year <= max(NDVI_YEARS))]
        print("Computing monthly NDVI (daily interpolation); this takes several minutes...", flush=True)
        monthly = _monthly_means_all_points(ndvi)
        MONTHLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_parquet(MONTHLY_CACHE, index=False)
        print(f"Wrote {MONTHLY_CACHE}", flush=True)
    ndvi_stats = _rotation_ndvi_stats(monthly, rot_by_point)
    weather = _weather_by_month(_load_power_daily())
    rotations = _rotation_calendar(ndvi_stats, weather)
    summer = _summer_cover_comparison(ndvi_stats)

    payload = {
        "region": "Finney County, Kansas",
        "ndvi_years": [min(NDVI_YEARS), max(NDVI_YEARS)],
        "weather_years": [min(WEATHER_YEARS), max(WEATHER_YEARS)],
        "low_cover_ndvi_threshold": LOW_COVER_NDVI,
        "low_cover_threshold_note": LOW_COVER_FOOTNOTE,
        "interpolation": "16-day MODIS linear to daily mean; per-point monthly median across years, then rotation aggregate",
        "wheat_crop_months": sorted(WHEAT_MONTHS),
        "vulnerable_heat_months": sorted(HEAT_VULN),
        "vulnerable_frost_months": sorted(FROST_VULN),
        "weather_by_month": {str(k): v for k, v in weather.items()},
        "rotations": rotations,
        "point_counts": {r: len(points_by_rot[r]) for r in ROTATIONS},
        "summer_cover_comparison": summer,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _plot_calendar(ndvi_stats, weather)
    return payload


def print_month_table(data: dict) -> None:
    print(f"\nMean NDVI by month (daily interp, clean points, {data['ndvi_years'][0]}–{data['ndvi_years'][1]})")
    print(LOW_COVER_FOOTNOTE)
    header = "Rot " + " ".join(f"{n:>6}" for n in MONTH_NAMES)
    print(header)
    for rot in ROTATIONS:
        means = data["rotations"][rot]["mean_ndvi_by_month"]
        row = f" {rot}  " + " ".join(f"{means.get(str(m), float('nan')):6.3f}" for m in MONTHS)
        print(row)
        print(f"      bare_months (mean < {LOW_COVER_NDVI}): {data['rotations'][rot]['bare_months']}")
    print("\n" + data["summer_cover_comparison"]["note"])
    print("\nWeather (means 2000-2025) — heat / frost / heavy rain days, rain mm")
    print("Mo   heat  frost  hvyRn  rain_mm  wheat  heatV  frostV")
    for m in MONTHS:
        w = data["weather_by_month"][str(m)]
        print(
            f"{MONTH_NAMES[m-1]:>3} {w['heat_days_mean']:5.1f} {w['frost_days_mean']:6.1f} "
            f"{w['heavy_rain_days_mean']:6.1f} {w['rain_mm_mean']:8.1f} "
            f"{'Y' if w['wheat_window'] else 'N':>5} {'Y' if w['vulnerable_heat_grain_fill'] else 'N':>6} "
            f"{'Y' if w['vulnerable_frost_greenup'] else 'N':>7}"
        )


def main() -> None:
    data = build_field_calendar()
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_PNG}", flush=True)
    for r in ROTATIONS:
        print(f"  Rotation {r}: n_points={data['point_counts'][r]}, bare_months={data['rotations'][r]['bare_months']}")
    print_month_table(data)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        t = _daily_mean_ndvi_in_month(
            pd.DataFrame({"date": pd.to_datetime(["2015-07-01", "2015-07-17"]), "ndvi": [0.1, 0.15]}),
            2015,
            7,
        )
        assert t is not None and t < LOW_COVER_NDVI
        print("09_field_calendar self_test: ok")
    else:
        main()