# Backup point extraction via Google Earth Engine when AppEEARS CSVs are unavailable.

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTS_CSV = PROJECT_ROOT / "data" / "processed" / "sample_points.csv"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

NDVI_COLLECTION = "MODIS/061/MOD13Q1"
SMAP_COLLECTION = "NASA/SMAP/SPL3SMP_E/006"
SRTM_IMAGE = "USGS/SRTMGL1_003"
NDVI_SCALE = 0.0001
NDVI_OK_QA = {0, 1}
SM_MIN, SM_MAX = 0.0, 0.6
MAX_GETINFO_ATTEMPTS = 3
RETRY_WAIT_S = 8

NDVI_START = "2012-07-01"
NDVI_END = "2026-01-01"
NDVI_SCALE_M = 250
SMAP_SCALE_M = 9000
ELEV_SCALE_M = 30


def _get_info(obj, label: str) -> dict | list:
    last_err: Exception | None = None
    for attempt in range(1, MAX_GETINFO_ATTEMPTS + 1):
        try:
            return obj.getInfo()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  {label}: getInfo attempt {attempt} failed: {exc}")
            if attempt < MAX_GETINFO_ATTEMPTS:
                time.sleep(RETRY_WAIT_S)
    raise RuntimeError(f"getInfo failed for {label}") from last_err


def _load_points_table() -> pd.DataFrame:
    df = pd.read_csv(POINTS_CSV)
    df = df.rename(columns={"ID": "point_id", "Latitude": "lat", "Longitude": "lon"})
    return df[["point_id", "lat", "lon"]].dropna()


def _build_feature_collection(ee, points: pd.DataFrame):
    features = [
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"point_id": int(row.point_id)})
        for row in points.itertuples(index=False)
    ]
    return ee.FeatureCollection(features)


def _index_to_date(index: str, time_ms: int | None) -> pd.Timestamp:
    if time_ms is not None:
        return pd.to_datetime(time_ms, unit="ms", utc=True).tz_localize(None)
    m = re.search(r"(\d{4})[-_]?(\d{3})", index)
    if m:
        year, doy = int(m.group(1)), int(m.group(2))
        return pd.to_datetime(f"{year}{doy:03d}", format="%Y%j")
    m2 = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", index)
    if m2:
        return pd.to_datetime("-".join(m2.groups()))
    return pd.NaT


def _band_prefix_map(collection) -> dict[str, pd.Timestamp]:
    idx_list = _get_info(collection.aggregate_array("system:index"), "system:index")
    time_list = _get_info(collection.aggregate_array("system:time_start"), "time_start")
    return {_idx: _index_to_date(_idx, _t) for _idx, _t in zip(idx_list, time_list, strict=False)}


def _scale_ndvi(value: float) -> float:
    if value > 1:
        return float(value) * NDVI_SCALE
    return float(value)


def _melt_modis_features(features: list[dict], prefix_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict] = []
    for feat in features:
        pid = feat.get("properties", {}).get("point_id")
        if pid is None:
            continue
        props = feat.get("properties", {})
        ndvi_keys = [k for k in props if k.endswith("_NDVI") or k == "NDVI"]
        for ndvi_key in ndvi_keys:
            if ndvi_key == "NDVI":
                prefix = ""
                qa_key = "SummaryQA"
            else:
                prefix = ndvi_key[: -len("_NDVI")]
                qa_key = f"{prefix}_SummaryQA"
            ndvi_raw = props.get(ndvi_key)
            qa = props.get(qa_key)
            if ndvi_raw is None:
                continue
            try:
                rel = int(qa) if qa is not None else 99
            except (TypeError, ValueError):
                rel = 99
            if rel not in NDVI_OK_QA:
                continue
            date = prefix_dates.get(prefix, pd.NaT)
            if pd.isna(date):
                continue
            rows.append(
                {
                    "point_id": int(pid),
                    "date": date,
                    "ndvi": _scale_ndvi(float(ndvi_raw)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["point_id", "date", "ndvi"])
    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset=["point_id", "date"], keep="last").sort_values(["point_id", "date"])


def _melt_smap_features(features: list[dict], prefix_dates: dict[str, pd.Timestamp], sm_suffix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for feat in features:
        pid = feat.get("properties", {}).get("point_id")
        if pid is None:
            continue
        props = feat.get("properties", {})
        sm_keys = [k for k in props if k.endswith(f"_{sm_suffix}") or k == sm_suffix]
        for sm_key in sm_keys:
            val = props.get(sm_key)
            if val is None:
                continue
            try:
                sm = float(val)
            except (TypeError, ValueError):
                continue
            if sm < SM_MIN or sm > SM_MAX:
                continue
            if sm_key == sm_suffix:
                prefix = ""
            else:
                prefix = sm_key[: -len(f"_{sm_suffix}")]
            date = prefix_dates.get(prefix, pd.NaT)
            if pd.isna(date):
                continue
            rows.append({"point_id": int(pid), "date": date, "sm": sm})
    if not rows:
        return pd.DataFrame(columns=["point_id", "date", "sm"])
    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset=["point_id", "date"], keep="last").sort_values(["point_id", "date"])


def extract_ndvi(ee, fc) -> pd.DataFrame:
    years = list(range(2012, 2026))
    frames: list[pd.DataFrame] = []
    for year in years:
        start = NDVI_START if year == 2012 else f"{year}-01-01"
        end = f"{year + 1}-01-01" if year < 2025 else NDVI_END
        col = (
            ee.ImageCollection(NDVI_COLLECTION)
            .filterDate(start, end)
            .select(["NDVI", "SummaryQA"])
        )
        count = _get_info(col.size(), f"NDVI {year} count")
        print(f"NDVI {year}: {count} images")
        if count == 0:
            continue
        prefix_dates = _band_prefix_map(col)
        stacked = col.toBands()
        reduced = stacked.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=NDVI_SCALE_M)
        info = _get_info(reduced, f"NDVI reduceRegions {year}")
        frame = _melt_modis_features(info["features"], prefix_dates)
        print(f"  -> {len(frame):,} NDVI rows")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["point_id", "date", "ndvi"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["point_id", "date"])


def _smap_sm_band(ee, col) -> str:
    names = _get_info(col.first().bandNames(), "SMAP bandNames")
    print("SMAP band names:", names)
    for name in names:
        low = name.lower()
        if "soil_moisture" in low and "am" in low:
            return name
    for name in names:
        if "soil_moisture" in name.lower():
            return name
    raise RuntimeError(f"No AM soil moisture band found in {names}")


def extract_smap(ee, fc) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    sm_band: str | None = None
    for year in range(2015, 2025):
        start = f"{year}-09-15"
        end = f"{year}-11-01"
        col = ee.ImageCollection(SMAP_COLLECTION).filterDate(start, end)
        count = _get_info(col.size(), f"SMAP {year} count")
        print(f"SMAP {year}: {count} images")
        if count == 0:
            continue
        if sm_band is None:
            sm_band = _smap_sm_band(ee, col)
            print(f"Using SMAP band: {sm_band}")
        col = col.select(sm_band)
        prefix_dates = _band_prefix_map(col)
        stacked = col.toBands()
        reduced = stacked.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=SMAP_SCALE_M)
        info = _get_info(reduced, f"SMAP reduceRegions {year}")
        frame = _melt_smap_features(info["features"], prefix_dates, sm_band)
        print(f"  -> {len(frame):,} SMAP rows")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["point_id", "date", "sm"])
    return pd.concat(frames, ignore_index=True).sort_values(["point_id", "date"])


def extract_elevation(ee, fc) -> pd.DataFrame:
    dem = ee.Image(SRTM_IMAGE).select("elevation")
    reduced = dem.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=ELEV_SCALE_M)
    info = _get_info(reduced, "SRTM elevation")
    rows = []
    for feat in info["features"]:
        pid = feat.get("properties", {}).get("point_id")
        elev = feat.get("properties", {}).get("elevation")
        if pid is None or elev is None:
            continue
        rows.append({"point_id": int(pid), "elevation_m": float(elev)})
    return pd.DataFrame(rows).sort_values("point_id")


def _date_range_str(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return "n/a"
    return f"{df['date'].min().date()} .. {df['date'].max().date()}"


def main() -> int:
    import os

    import ee

    project = (os.getenv("GEE_PROJECT_ID") or "").strip()
    if not project:
        print("Set GEE_PROJECT_ID in .env (Google Cloud project with Earth Engine enabled).")
        return 1
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001
        print(f"ee.Initialize failed ({exc}). Run: earthengine authenticate")
        return 1

    if not POINTS_CSV.is_file():
        print(f"Missing {POINTS_CSV}")
        return 1

    points = _load_points_table()
    fc = _build_feature_collection(ee, points)
    print(f"Points: {len(points)}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    ndvi = extract_ndvi(ee, fc)
    smap = extract_smap(ee, fc)
    elev = extract_elevation(ee, fc)

    ndvi.to_parquet(PROC_DIR / "ndvi_points.parquet", index=False)
    smap.to_parquet(PROC_DIR / "smap_points.parquet", index=False)
    elev.to_csv(PROC_DIR / "elev_points.csv", index=False)

    print(f"\nndvi_points.parquet: {len(ndvi):,} rows, dates {_date_range_str(ndvi)}")
    print(f"smap_points.parquet: {len(smap):,} rows, dates {_date_range_str(smap)}")
    print(f"elev_points.csv: {len(elev):,} rows")
    return 0


def self_test() -> None:
    """Offline checks for NDVI scaling and property melting."""
    prefix_dates = {"2024_01_01": pd.Timestamp("2024-01-01")}
    features = [
        {
            "properties": {
                "point_id": 0,
                "2024_01_01_NDVI": 5500,
                "2024_01_01_SummaryQA": 0,
            }
        }
    ]
    ndvi = _melt_modis_features(features, prefix_dates)
    assert len(ndvi) == 1
    assert abs(ndvi["ndvi"].iloc[0] - 0.55) < 1e-6

    sm = _melt_smap_features(
        [{"properties": {"point_id": 0, "20240101_soil_moisture_am": 0.25}}],
        {"20240101": pd.Timestamp("2024-01-01")},
        "soil_moisture_am",
    )
    assert len(sm) == 1
    print("03b_gee_extract self_test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    sys.exit(main())