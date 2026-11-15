# Build API feature store: one row per sample point (CURSOR_CONTEXT §5, §11).

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "sample_points.csv"
CROP_HISTORY = PROJECT_ROOT / "data" / "processed" / "crop_history.csv"
SOIL_CSV = PROJECT_ROOT / "data" / "raw" / "soil" / "soil_points.csv"
ELEV_CSV = PROJECT_ROOT / "data" / "processed" / "elev_points.csv"
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
MODEL_TABLE = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"
NDVI_PARQUET = PROJECT_ROOT / "data" / "processed" / "ndvi_points.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "feature_store.parquet"

CDL_NAMES = {
    24: "winter_wheat",
    236: "double_crop_wheat_sorghum",
    61: "fallow",
    4: "sorghum",
    205: "triticale",
    53: "peas",
    37: "other_hay",
    1: "corn",
    36: "alfalfa",
    5: "soybeans",
    176: "grassland",
}
YEARS = list(range(2013, 2026))
BARE_THRESH = 0.2


def _texture(sand: float | None, clay: float | None) -> str:
    if sand is None or clay is None or (isinstance(sand, float) and np.isnan(sand)):
        return "medium"
    if clay >= 35:
        return "heavy"
    if sand >= 50:
        return "light"
    return "medium"


def _system_to_rotation(system: str) -> str:
    return {"wheat_fallow": "A", "wheat_sorghum_fallow": "B", "wheat_forage": "C"}.get(system, "other")


def _guess_rotation_last4(codes: list[int]) -> str:
    """Map last 4 CDL codes to A/B/C/other using CURSOR_CONTEXT rotation_context rules on wheat years only."""
    if not codes:
        return "other"
    last = codes[-4:]
    if 4 in last:
        return "B"
    forage = {205, 37, 53}
    if any(c in forage for c in last):
        return "C"
    if 61 in last and 24 in last:
        return "A"
    return "other"


def _annual_bare_days(point_ndvi: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for year in YEARS:
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        sub = point_ndvi[(point_ndvi["date"] >= start) & (point_ndvi["date"] <= end)]
        if sub.empty:
            continue
        daily = (
            sub.groupby("date")["ndvi"]
            .mean()
            .reindex(pd.date_range(start, end, freq="D"))
            .interpolate(method="linear", limit_direction="both")
        )
        out[year] = float((daily < BARE_THRESH).sum())
    return out


def build_feature_store() -> pd.DataFrame:
    pts = pd.read_csv(SAMPLE_CSV).rename(columns={"ID": "point_id", "Latitude": "lat", "Longitude": "lon"})
    hist = pd.read_csv(CROP_HISTORY)
    soil = pd.read_csv(SOIL_CSV)
    elev = pd.read_csv(ELEV_CSV) if ELEV_CSV.is_file() else pd.DataFrame(columns=["point_id", "elevation_m"])
    weather = pd.read_csv(WEATHER_CSV).rename(columns={"harvest_year": "year"})

    peak = None
    if MODEL_TABLE.is_file():
        peak = pd.read_parquet(MODEL_TABLE)[["point_id", "year", "peak_ndvi"]]

    ndvi = pd.read_parquet(NDVI_PARQUET) if NDVI_PARQUET.is_file() else pd.DataFrame()
    if not ndvi.empty:
        ndvi["date"] = pd.to_datetime(ndvi["date"])
        ndvi_by_pid = {int(pid): grp for pid, grp in ndvi.groupby("point_id", sort=False)}
    else:
        ndvi_by_pid = {}

    rows = []
    for _, pt in pts.iterrows():
        pid = int(pt["point_id"])
        h = hist[hist["point_id"] == pid]
        s = soil[soil["point_id"] == pid]
        e = elev[elev["point_id"] == pid]

        crop_hist: dict[str, str] = {}
        codes: list[int] = []
        if not h.empty:
            r = h.iloc[0]
            for y in YEARS:
                col = f"cdl_{y}"
                if col in r.index and pd.notna(r[col]):
                    code = int(r[col])
                    codes.append(code)
                    crop_hist[str(y)] = CDL_NAMES.get(code, f"cdl_{code}")

        current_guess = _system_to_rotation(str(h.iloc[0]["system"])) if not h.empty else "other"
        if len(codes) >= 4:
            current_guess = _guess_rotation_last4(codes)

        soil_row = {
            "om_pct": None,
            "sand_pct": None,
            "silt_pct": None,
            "clay_pct": None,
            "awc": None,
            "kwfact": None,
            "slope_r": None,
            "hydgrp": None,
            "texture_class": "medium",
        }
        if not s.empty:
            sr = s.iloc[0]
            soil_row = {
                "om_pct": sr.get("om_0_30_pct"),
                "sand_pct": sr.get("sand_0_30_pct"),
                "silt_pct": sr.get("silt_0_30_pct"),
                "clay_pct": sr.get("clay_0_30_pct"),
                "awc": sr.get("awc_0_30"),
                "kwfact": sr.get("kwfact_0_30"),
                "slope_r": sr.get("slope_r_pct"),
                "hydgrp": sr.get("hydrologic_soil_group"),
                "texture_class": _texture(
                    float(sr["sand_0_30_pct"]) if pd.notna(sr.get("sand_0_30_pct")) else None,
                    float(sr["clay_0_30_pct"]) if pd.notna(sr.get("clay_0_30_pct")) else None,
                ),
            }

        elev_m = float(e.iloc[0]["elevation_m"]) if not e.empty else None

        bare_by_year = _annual_bare_days(ndvi_by_pid[pid]) if pid in ndvi_by_pid else {}
        median_bare = float(np.median(list(bare_by_year.values()))) if bare_by_year else None

        peak_by_year: dict[int, float] = {}
        if peak is not None:
            sub = peak[peak["point_id"] == pid]
            for _, pr in sub.iterrows():
                peak_by_year[int(pr["year"])] = float(pr["peak_ndvi"])

        yearly = []
        for _, w in weather.iterrows():
            y = int(w["year"])
            yearly.append(
                {
                    "year": y,
                    "season_rain_mm": float(w["season_rain_mm"]),
                    "heat_days": int(w["heat_days"]),
                    "peak_ndvi": peak_by_year.get(y),
                    "bare_days": bare_by_year.get(y),
                    "et0_mm": float(w["et0_mm"]) if pd.notna(w.get("et0_mm")) else None,
                }
            )

        rows.append(
            {
                "point_id": pid,
                "lat": float(pt["lat"]),
                "lon": float(pt["lon"]),
                "elevation_m": elev_m,
                **soil_row,
                "current_rotation_guess": current_guess,
                "crop_history_json": json.dumps(crop_hist),
                "median_bare_days_per_year": median_bare,
                "yearly_trends_json": json.dumps(yearly),
            }
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"feature_store rows: {len(df)} -> {OUT}")
    return df


if __name__ == "__main__":
    build_feature_store()
    sys.exit(0)