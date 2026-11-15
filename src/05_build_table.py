# Join wheat years with NDVI target, SMAP, soil, elevation, and weather (CURSOR_CONTEXT §8).

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHEAT_CSV = PROJECT_ROOT / "data" / "processed" / "wheat_years.csv"
NDVI_PARQUET = PROJECT_ROOT / "data" / "processed" / "ndvi_points.parquet"
SMAP_PARQUET = PROJECT_ROOT / "data" / "processed" / "smap_points.parquet"
SOIL_CSV = PROJECT_ROOT / "data" / "raw" / "soil" / "soil_points.csv"
ELEV_CSV = PROJECT_ROOT / "data" / "processed" / "elev_points.csv"
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"

BARE_NDVI_THRESH = 0.2
MIN_PEAK_OBS = 3
MIN_SMAP_OBS = 3


def _load_ndvi() -> pd.DataFrame:
    if not NDVI_PARQUET.is_file():
        raise FileNotFoundError(f"Missing {NDVI_PARQUET} — run src/03_parse_appeears.py first")
    df = pd.read_parquet(NDVI_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_smap() -> pd.DataFrame:
    if not SMAP_PARQUET.is_file():
        raise FileNotFoundError(f"Missing {SMAP_PARQUET}")
    df = pd.read_parquet(SMAP_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


def peak_ndvi(point_id: int, harvest_year: int, ndvi: pd.DataFrame) -> float | None:
    start = pd.Timestamp(harvest_year, 3, 1)
    end = pd.Timestamp(harvest_year, 6, 30)
    sub = ndvi[(ndvi["date"] >= start) & (ndvi["date"] <= end)]
    if len(sub) < MIN_PEAK_OBS:
        return None
    return float(sub["ndvi"].max())


def prev_year_bare_days(point_id: int, harvest_year: int, ndvi: pd.DataFrame) -> float | None:
    year = harvest_year - 1
    start = pd.Timestamp(year, 1, 1)
    end = pd.Timestamp(year, 12, 31)
    sub = ndvi[(ndvi["date"] >= start) & (ndvi["date"] <= end)]
    if sub.empty:
        return None
    daily = (
        sub.groupby("date", as_index=True)["ndvi"]
        .mean()
        .reindex(pd.date_range(start, end, freq="D"))
        .interpolate(method="linear", limit_direction="both")
    )
    return float((daily < BARE_NDVI_THRESH).sum())


def sm_planting(point_id: int, harvest_year: int, smap: pd.DataFrame) -> tuple[float | None, int]:
    start = pd.Timestamp(harvest_year - 1, 9, 15)
    end = pd.Timestamp(harvest_year - 1, 10, 31)
    sub = smap[(smap["date"] >= start) & (smap["date"] <= end)]
    n = len(sub)
    if n < MIN_SMAP_OBS:
        return None, n
    return float(sub["sm"].mean()), n


def build_table(
    wheat: pd.DataFrame,
    ndvi: pd.DataFrame,
    smap: pd.DataFrame,
    soil: pd.DataFrame,
    elev: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    soil = soil.rename(
        columns={
            "om_0_30_pct": "om_pct",
            "sand_0_30_pct": "sand_pct",
            "silt_0_30_pct": "silt_pct",
            "clay_0_30_pct": "clay_pct",
            "awc_0_30": "awc",
            "kwfact_0_30": "kwfact",
            "slope_r_pct": "slope_r",
        }
    )
    soil_cols = ["point_id", "om_pct", "sand_pct", "silt_pct", "clay_pct", "awc", "kwfact", "slope_r"]
    soil_slim = soil[[c for c in soil_cols if c in soil.columns]]

    elev_slim = elev[["point_id", "elevation_m"]] if not elev.empty else pd.DataFrame(columns=["point_id", "elevation_m"])
    weather = weather.rename(columns={"harvest_year": "year"})

    ndvi_by_point = {int(pid): grp for pid, grp in ndvi.groupby("point_id", sort=False)}
    smap_by_point = {int(pid): grp for pid, grp in smap.groupby("point_id", sort=False)}

    rows: list[dict] = []
    for _, w in wheat.iterrows():
        pid = int(w["point_id"])
        hy = int(w["year"])
        pt_ndvi = ndvi_by_point.get(pid)
        if pt_ndvi is None:
            continue
        peak = peak_ndvi(pid, hy, pt_ndvi)
        if peak is None:
            continue
        bare = prev_year_bare_days(pid, hy, pt_ndvi)
        pt_smap = smap_by_point.get(pid, pd.DataFrame(columns=["date", "sm"]))
        sm, sm_n = sm_planting(pid, hy, pt_smap)
        rows.append(
            {
                "point_id": pid,
                "year": hy,
                "rotation_context": w["rotation_context"],
                "prev1_crop": w["prev1_crop"],
                "prev2_crop": w["prev2_crop"],
                "peak_ndvi": peak,
                "prev_year_bare_days": bare,
                "sm_planting": sm,
                "sm_planting_n_obs": sm_n,
            }
        )

    base = pd.DataFrame(rows)
    if base.empty:
        return base

    base = base.merge(soil_slim, on="point_id", how="left")
    base = base.merge(elev_slim, on="point_id", how="left")
    base = base.merge(weather, on="year", how="left")
    return base


def _missing_share(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).round(2)


def run_build() -> int:
    for path in (WHEAT_CSV, SOIL_CSV, WEATHER_CSV):
        if not path.is_file():
            print(f"Missing required file: {path}")
            return 1

    wheat = pd.read_csv(WHEAT_CSV)
    ndvi = _load_ndvi()
    try:
        smap = _load_smap()
    except FileNotFoundError as exc:
        print(exc)
        smap = pd.DataFrame(columns=["point_id", "date", "sm"])

    soil = pd.read_csv(SOIL_CSV)
    elev = pd.read_csv(ELEV_CSV) if ELEV_CSV.is_file() else pd.DataFrame(columns=["point_id", "elevation_m"])
    weather = pd.read_csv(WEATHER_CSV)

    table = build_table(wheat, ndvi, smap, soil, elev, weather)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PARQUET, index=False)

    print(f"model_table rows: {len(table):,}")
    if table.empty:
        return 0

    print("\nRows per rotation_context and year:")
    print(table.groupby(["rotation_context", "year"], dropna=False).size().unstack(fill_value=0))

    print("\nMissing share per column (%):")
    print(_missing_share(table).sort_values(ascending=False).to_string())
    print(f"\nWrote {OUT_PARQUET}")
    return 0


def self_test() -> None:
    dates = pd.to_datetime(["2024-03-01", "2024-04-01", "2024-05-01", "2023-06-01", "2023-10-20"])
    ndvi = pd.DataFrame(
        {
            "point_id": [0, 0, 0, 0, 0],
            "date": dates,
            "ndvi": [0.4, 0.5, 0.55, 0.1, 0.3],
        }
    )
    smap = pd.DataFrame({"point_id": [0, 0, 0], "date": pd.to_datetime(["2023-09-20", "2023-10-01", "2023-10-15"]), "sm": [0.2, 0.22, 0.21]})
    wheat = pd.DataFrame(
        {
            "point_id": [0],
            "year": [2024],
            "prev1_crop": ["fallow"],
            "prev2_crop": ["winter_wheat"],
            "rotation_context": ["A"],
        }
    )
    soil = pd.DataFrame(
        {
            "point_id": [0],
            "om_0_30_pct": [2.0],
            "sand_0_30_pct": [10.0],
            "silt_0_30_pct": [50.0],
            "clay_0_30_pct": [40.0],
            "awc_0_30": [0.2],
            "kwfact_0_30": [0.3],
            "slope_r_pct": [1.0],
        }
    )
    elev = pd.DataFrame({"point_id": [0], "elevation_m": [900.0]})
    weather = pd.DataFrame(
        {
            "harvest_year": [2024],
            "season_rain_mm": [200.0],
            "heavy_rain_days": [1],
            "heat_days": [10],
            "frost_days": [2],
            "et0_mm": [500.0],
            "water_balance_mm": [-300.0],
        }
    )
    out = build_table(wheat, ndvi, smap, soil, elev, weather)
    assert len(out) == 1
    assert np.isclose(out["peak_ndvi"].iloc[0], 0.55)
    assert out["sm_planting"].iloc[0] is not None
    print("05_build_table self_test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    sys.exit(run_build())