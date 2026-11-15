# Rotation comparison logic for POST /compare (CURSOR_CONTEXT §10–11).

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROP_TABLE = PROJECT_ROOT / "data" / "crop_table.csv"
FEATURE_STORE = PROJECT_ROOT / "data" / "processed" / "feature_store.parquet"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
SAMPLE_POINTS = PROJECT_ROOT / "data" / "processed" / "sample_points.csv"
MODEL_P10 = PROJECT_ROOT / "models" / "model_p10.pkl"
MODEL_P50 = PROJECT_ROOT / "models" / "model_p50.pkl"
MODEL_P90 = PROJECT_ROOT / "models" / "model_p90.pkl"
FINAL_MODEL = PROJECT_ROOT / "models" / "final_model.json"
FIELD_CALENDAR_JSON = PROJECT_ROOT / "data" / "processed" / "field_calendar.json"
WEATHER_CSV = PROJECT_ROOT / "data" / "processed" / "weather_by_year.csv"
POWER_CSV = PROJECT_ROOT / "data" / "raw" / "power" / "power_finney_daily.csv"
SOIL_CSV = PROJECT_ROOT / "data" / "raw" / "soil" / "soil_points.csv"
FINAL_JSON_PATH = FINAL_MODEL
MC_DRAWS = 5000
HEAT_TMAX_C = 32.0
CLIMATE_YEARS = range(2013, 2026)

DISCLAIMER = (
    "This tool supports, and does not replace, the judgment of farmers and local agricultural experts."
)

ROTATIONS = {
    "A": {"name": "Wheat, Fallow", "cycle_years": 2, "crops": ["winter_wheat", "fallow"]},
    "B": {"name": "Wheat, Sorghum, Fallow", "cycle_years": 3, "crops": ["winter_wheat", "sorghum", "fallow"]},
    "C": {
        "name": "Wheat, Forage (triticale, hay, peas)",
        "cycle_years": 2,
        "crops": ["winter_wheat", "triticale"],
    },
}

CONTEXT_FOR_ROTATION = {"A": "A", "B": "B", "C": "C"}
PRIORITY_KEYS = ("water", "soil", "income", "erosion", "forage", "market")
DEFAULT_PRIORITIES = ["water", "soil", "income"]
MIN_DRY_ROWS = 100
BARE_DAYS_MAX = 365.0
KWFACT_REF_MAX = 0.6
SLOPE_REF_MAX_PCT = 10.0
WATER_NEED_REF_MM = 800.0
CROP_PRICE_REF_USD = 12.0
CROP_BARE_FRACTION = {
    "fallow": 0.82,
    "winter_wheat": 0.06,
    "sorghum": 0.16,
    "triticale": 0.12,
    "other_hay": 0.10,
    "field_pea": 0.14,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _sample_points() -> pd.DataFrame | None:
    if not SAMPLE_POINTS.is_file():
        return None
    df = pd.read_csv(SAMPLE_POINTS)
    df = df.rename(columns={"ID": "point_id", "Latitude": "lat", "Longitude": "lon"})
    return df[["point_id", "lat", "lon"]].dropna()


def nearest_point(lat: float, lon: float) -> tuple[int, float] | None:
    pts = _sample_points()
    if pts is None or pts.empty:
        return None
    dists = pts.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
    idx = int(dists.idxmin())
    return int(pts.loc[idx, "point_id"]), float(dists.loc[idx])


def _final_model_ready() -> bool:
    return FINAL_MODEL.is_file()


def _legacy_artifacts_ready() -> bool:
    return all(
        p.is_file()
        for p in (FEATURE_STORE, MODEL_P50, MODEL_P10, MODEL_P90, METRICS_PATH)
    )


def _artifacts_ready() -> bool:
    return _final_model_ready() and FEATURE_STORE.is_file()


@lru_cache(maxsize=1)
def _load_crop_table() -> pd.DataFrame | None:
    if not CROP_TABLE.is_file():
        return None
    df = pd.read_csv(CROP_TABLE)
    if df.empty:
        return None
    return df.set_index("crop")


@lru_cache(maxsize=1)
def _load_feature_store() -> pd.DataFrame | None:
    if not FEATURE_STORE.is_file():
        return None
    return pd.read_parquet(FEATURE_STORE)


def feature_point(point_id: int) -> dict | None:
    store = _load_feature_store()
    if store is None:
        return None
    row = store[store["point_id"] == point_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_models() -> tuple[Any, Any, Any] | None:
    if not _legacy_artifacts_ready():
        return None
    return joblib.load(MODEL_P10), joblib.load(MODEL_P50), joblib.load(MODEL_P90)


@lru_cache(maxsize=1)
def _load_final_model() -> dict | None:
    if not _final_model_ready():
        return None
    return json.loads(FINAL_MODEL.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_field_calendar() -> dict | None:
    if not FIELD_CALENDAR_JSON.is_file():
        return None
    return json.loads(FIELD_CALENDAR_JSON.read_text(encoding="utf-8"))


def bare_months_for_rotation(rotation_id: str) -> list[int]:
    cal = load_field_calendar()
    if not cal:
        return []
    rot = cal.get("rotations", {}).get(rotation_id, {})
    return [int(m) for m in rot.get("bare_months", [])]


def _predict_year_mu(final: dict, season_rain_mm: float, heat_days: float) -> float:
    ym = final["year_model"]
    vals = {"season_rain_mm": float(season_rain_mm), "heat_days": float(heat_days)}
    mu = float(ym["intercept"])
    for feat in ym["features"]:
        z = (vals[feat] - ym["scaler_mean"][feat]) / ym["scaler_std"][feat]
        mu += ym["coefficients"][feat] * z
    clip = ym.get("ndvi_clip", [0.4, 0.8])
    return float(np.clip(mu, float(clip[0]), float(clip[1])))


def _rotation_dry_row_count(rot: dict) -> int:
    return int(rot.get("n_dry_rows") or rot.get("n_dry_years") or 0)


def _rotation_uses_dry_effect(rot: dict) -> bool:
    if bool(rot.get("use_dry_effect_in_scenarios")):
        return True
    return _rotation_dry_row_count(rot) >= MIN_DRY_ROWS and bool(rot.get("anomalies_mc_dry"))


def _structural_bare_days_per_year(crops: list[str], cycle_years: int, crop_tbl: pd.DataFrame | None) -> float:
    total = 0.0
    for crop in crops:
        if crop_tbl is not None and crop in crop_tbl.index:
            if crop == "fallow":
                frac = 0.82
            else:
                lvl = str(crop_tbl.loc[crop].get("residue_level", "low")).lower()
                frac = {"low": 0.45, "medium": 0.22, "high": 0.08}.get(lvl, 0.22)
        else:
            frac = CROP_BARE_FRACTION.get(crop, 0.25)
        total += frac * 365.0
    return total / max(cycle_years, 1)


def _bare_days_per_year(rot_meta: dict, crops: list[str], cycle_years: int, crop_tbl: pd.DataFrame | None) -> float:
    """NDVI bare days averaged over the rotation cycle, floored by crop-calendar bare exposure."""
    ndvi = rot_meta.get("median_bare_days_per_rotation_year")
    structural = _structural_bare_days_per_year(crops, cycle_years, crop_tbl)
    if ndvi is None or (isinstance(ndvi, float) and math.isnan(ndvi)):
        return structural
    return float(max(float(ndvi), structural))


def _rotation_mc_anomalies(final: dict, ctx: str, year_type: str) -> list[float]:
    rot = final["rotations"].get(ctx) or {}
    use_dry = _rotation_uses_dry_effect(rot)
    if year_type in ("dry", "hot") and use_dry:
        anomalies = rot.get("anomalies_mc_dry") or []
    else:
        anomalies = rot.get("anomalies_mc_normal") or []
    if not anomalies:
        anomalies = rot.get("anomalies_mc_normal") or rot.get("anomalies_empirical") or []
    return [float(x) for x in anomalies]


def _monte_carlo_ndvi(
    final: dict,
    rotation: str,
    season_rain_mm: float,
    heat_days: float,
    *,
    year_type: str = "normal",
) -> tuple[float, float, float]:
    ym = final["year_model"]
    ctx = rotation if rotation in (final.get("rotations") or {}) else CONTEXT_FOR_ROTATION.get(rotation, rotation)
    anomalies = _rotation_mc_anomalies(final, ctx, year_type)
    if not anomalies:
        return None, None, None
    year_mu = _predict_year_mu(final, season_rain_mm, heat_days)
    resids = ym.get("loo_residuals") or []
    if not resids:
        return None, None, None
    yr = np.random.choice(resids, size=MC_DRAWS)
    an = np.random.choice(anomalies, size=MC_DRAWS)
    draws = year_mu + yr + an
    p10, p50, p90 = np.quantile(draws, [0.1, 0.5, 0.9])
    delta = float(final.get("interval_conformal_delta") or 0.0)
    if delta > 0:
        p10 -= delta
        p90 += delta
    return float(p10), float(p50), float(p90)


@lru_cache(maxsize=1)
def _climate_baseline() -> tuple[float, float, float, int]:
    if not WEATHER_CSV.is_file():
        return 300.0, 20.0, 900.0, 2021
    w = pd.read_csv(WEATHER_CSV)
    w = w[w["harvest_year"].between(min(CLIMATE_YEARS), max(CLIMATE_YEARS))]
    rain = float(w["season_rain_mm"].median())
    heat = float(w["heat_days"].median())
    et0 = float(w["et0_mm"].median()) if "et0_mm" in w.columns else 900.0
    median_year = int(w.loc[(w["season_rain_mm"] - rain).abs().idxmin(), "harvest_year"])
    return rain, heat, et0, median_year


@lru_cache(maxsize=1)
def _heat_days_plus2_delta(median_harvest_year: int) -> int:
    if not POWER_CSV.is_file():
        return 5
    df = pd.read_csv(POWER_CSV, skiprows=15)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str), format="%Y%j")
    df["T2M_MAX"] = pd.to_numeric(df["T2M_MAX"], errors="coerce")
    start = pd.Timestamp(median_harvest_year, 4, 1)
    end = pd.Timestamp(median_harvest_year, 6, 30)
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    base = int((sub["T2M_MAX"] > HEAT_TMAX_C).sum())
    hot = int((sub["T2M_MAX"] + 2 > HEAT_TMAX_C).sum())
    return max(0, hot - base)


def _scenario_weather(year_type: str) -> tuple[float, float, float, dict]:
    rain, heat, et0, median_year = _climate_baseline()
    meta = {"year_type": year_type, "rain_change_pct": 0, "temp_change_c": 0, "median_climate_year": median_year}
    if year_type == "dry":
        rain *= 0.8
        meta["rain_change_pct"] = -20
        return rain, heat, et0, meta
    if year_type == "hot":
        rain *= 0.8
        delta = _heat_days_plus2_delta(median_year)
        heat = heat + delta
        meta["rain_change_pct"] = -20
        meta["temp_change_c"] = 2
        meta["heat_days_added_from_plus2c"] = delta
        et0 *= 1.1
        return rain, heat, et0, meta
    return rain, heat, et0, meta


def _scenario(year_type: str) -> dict:
    _, _, _, meta = _scenario_weather(year_type)
    return meta


def _confidence_final(rotation_id: str, final: dict, interval: float | None, irrigated: bool) -> str:
    ctx = CONTEXT_FOR_ROTATION.get(rotation_id, rotation_id)
    rot = final.get("rotations", {}).get(ctx, {})
    n = int(rot.get("n") or 0)
    if n >= 300:
        level = "high"
    elif n >= 100:
        level = "medium"
    else:
        level = "low"
    if irrigated:
        level = {"high": "medium", "medium": "low"}.get(level, "low")
    return level


def _apply_scenario_row(row: pd.Series, scenario: dict) -> pd.Series:
    out = row.copy()
    if scenario["rain_change_pct"]:
        factor = 1 + scenario["rain_change_pct"] / 100
        for c in ("season_rain_mm", "fallow_rain_mm", "water_balance_mm"):
            if c in out.index and pd.notna(out[c]):
                out[c] = float(out[c]) * factor
    if scenario["temp_change_c"]:
        for c in ("mean_tmax_c",):
            if c in out.index and pd.notna(out[c]):
                out[c] = float(out[c]) + scenario["temp_change_c"]
        if "heat_days" in out.index and pd.notna(out["heat_days"]):
            out["heat_days"] = float(out["heat_days"]) + max(0, scenario["temp_change_c"])
        if "et0_mm" in out.index and pd.notna(out["et0_mm"]):
            out["et0_mm"] = float(out["et0_mm"]) * (1 + 0.05 * scenario["temp_change_c"])
    return out


def _norm_lower_better(values: dict[str, float | None]) -> dict[str, float | None]:
    nums = {k: v for k, v in values.items() if v is not None}
    if len(nums) < 2:
        return {k: None for k in values}
    vmax, vmin = max(nums.values()), min(nums.values())
    if vmax == vmin:
        return {k: 5.0 if v is not None else None for k, v in values.items()}
    return {k: (None if v is None else (vmax - v) / (vmax - vmin) * 10) for k, v in values.items()}


def _norm_higher_better(values: dict[str, float | None]) -> dict[str, float | None]:
    nums = {k: v for k, v in values.items() if v is not None}
    if len(nums) < 2:
        return {k: None for k in values}
    vmax, vmin = max(nums.values()), min(nums.values())
    if vmax == vmin:
        return {k: 5.0 if v is not None else None for k, v in values.items()}
    return {k: (None if v is None else (v - vmin) / (vmax - vmin) * 10) for k, v in values.items()}


def _forage_score(level: str | float) -> float | None:
    if level is None or (isinstance(level, float) and math.isnan(level)):
        return None
    s = str(level).lower()
    return {"low": 3.0, "medium": 6.0, "high": 9.0}.get(s)


def _erosion_risk_label(score: float | None) -> str | None:
    if score is None:
        return None
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def _erosion_risk_score(bare_days: float | None, k_factor: float | None, slope: float | None) -> float | None:
    """Composite erosion risk 0-1 on fixed reference ranges (bare, kwfact, slope)."""
    if bare_days is None or k_factor is None or slope is None:
        return None
    bare_n = min(max(float(bare_days), 0.0) / BARE_DAYS_MAX, 1.0)
    k_n = min(max(float(k_factor), 0.0) / KWFACT_REF_MAX, 1.0)
    slope_n = min(max(float(slope), 0.0) / SLOPE_REF_MAX_PCT, 1.0)
    return bare_n * 0.5 + k_n * 0.3 + slope_n * 0.2


def _absolute_soil_score(bare_days_per_year: float, nitrogen_fixing: bool) -> float:
    bare = min(max(float(bare_days_per_year), 0.0), BARE_DAYS_MAX)
    score = 10.0 * (1.0 - bare / BARE_DAYS_MAX)
    if nitrogen_fixing:
        score += 1.0
    return float(np.clip(score, 0.0, 10.0))


def _absolute_erosion_score(erosion_risk: float | None) -> float | None:
    if erosion_risk is None:
        return None
    return float(np.clip(10.0 * (1.0 - float(erosion_risk)), 0.0, 10.0))


def _absolute_water_score(water_need_mm: float | None) -> float | None:
    if water_need_mm is None:
        return None
    need = min(max(float(water_need_mm), 0.0), WATER_NEED_REF_MM)
    return float(np.clip(10.0 * (1.0 - need / WATER_NEED_REF_MM), 0.0, 10.0))


def _crop_table_forage_priority(crops: list[str], crop_tbl: pd.DataFrame | None) -> float | None:
    if crop_tbl is None:
        return None
    vals: list[float] = []
    for crop in crops:
        if crop not in crop_tbl.index:
            return None
        level = _forage_score(crop_tbl.loc[crop].get("forage_value"))
        if level is None:
            return None
        vals.append(float(level))
    return float(np.mean(vals)) if vals else None


def _crop_table_price_priority(crops: list[str], crop_tbl: pd.DataFrame | None) -> float | None:
    if crop_tbl is None:
        return None
    prices: list[float] = []
    for crop in crops:
        if crop not in crop_tbl.index:
            return None
        p = crop_tbl.loc[crop].get("price_usd_per_unit")
        if p is None or (isinstance(p, float) and math.isnan(p)) or str(p).strip() == "":
            return None
        prices.append(float(p))
    if not prices:
        return None
    mean_price = float(np.mean(prices))
    return float(np.clip(mean_price / CROP_PRICE_REF_USD * 10.0, 0.0, 10.0))


def _absolute_priority_scores(
    *,
    bare_days_per_year: float | None,
    erosion_risk: float | None,
    water_need_mm: float | None,
    crops: list[str],
    crop_tbl: pd.DataFrame | None,
    nitrogen_fixing: bool,
) -> dict[str, float | None]:
    soil = _absolute_soil_score(bare_days_per_year, nitrogen_fixing) if bare_days_per_year is not None else None
    return {
        "water": _absolute_water_score(water_need_mm),
        "soil": soil,
        "income": _crop_table_price_priority(crops, crop_tbl),
        "erosion": _absolute_erosion_score(erosion_risk),
        "forage": _crop_table_forage_priority(crops, crop_tbl),
        "market": _crop_table_price_priority(crops, crop_tbl),
    }


def _erosion_raw(bare_days: float | None, k_factor: float | None, slope: float | None) -> float | None:
    return _erosion_risk_score(bare_days, k_factor, slope)


def _water_need_mm(
    crops: list[str],
    et0_mm: float | None,
    season_rain: float | None,
    crop_tbl: pd.DataFrame | None,
    cycle_years: int,
) -> tuple[float | None, bool]:
    """Returns (mm per year need, crop_data_pending)."""
    if crop_tbl is None or et0_mm is None:
        return None, True
    total_kc_et = 0.0
    for crop in crops:
        if crop not in crop_tbl.index:
            return None, True
        row = crop_tbl.loc[crop]
        for col in ("kc_ini", "kc_mid", "kc_end"):
            if pd.isna(row.get(col)) or str(row.get(col)).strip() == "":
                return None, True
        kc = (float(row["kc_ini"]) + float(row["kc_mid"]) + float(row["kc_end"])) / 3
        total_kc_et += kc * float(et0_mm)
    per_year = total_kc_et / max(cycle_years, 1)
    effective_rain = float(season_rain) if season_rain is not None else 0.0
    return max(per_year - effective_rain, 0.0), False


def _rotation_why_evidence(final: dict, rid: str, p50: float | None) -> tuple[list[str], list[str]]:
    ctx = CONTEXT_FOR_ROTATION[rid]
    rot = final.get("rotations", {}).get(ctx, {})
    n = int(rot.get("n") or 0)
    anom = rot.get("anomaly_median")
    eval_loo = final.get("evaluation_loo", {})
    mae = eval_loo.get("mae_deterministic")
    why: list[str] = []
    evidence: list[str] = []
    if rid == "A":
        why.append("Wheat–fallow keeps a clean fallow window to store soil moisture before the next wheat crop.")
    elif rid == "B":
        why.append("Sorghum uses summer rain and can spread income across the rotation cycle.")
    else:
        why.append("Forage crops keep a living root in the rotation but add equipment and market uncertainty.")
    shrunk = rot.get("anomaly_median_shrunk", anom)
    if shrunk is not None and n > 0:
        direction = "above" if float(shrunk) >= 0 else "below"
        label = {
            "A": "wheat–fallow",
            "B": "wheat–sorghum–fallow",
            "C": "wheat–forage",
        }.get(rid, ctx)
        evidence.append(
            f"In {n} Finney County fields, {label} was {abs(float(shrunk)):.2f} NDVI {direction} "
            "the county average in the same year (MODIS peak season; effect shrunk by sample size)."
        )
    if mae is not None:
        evidence.append(
            f"Final model leave-one-year-out MAE {float(mae):.3f} on purity-filtered wheat years (weather + rotation offsets)."
        )
    if p50 is not None:
        evidence.append(f"Scenario median wheat peak NDVI p50 = {float(p50):.3f} (relative health, not yield bu/ac).")
    return why, evidence


def trends_for_point(point_id: int) -> dict:
    pt = feature_point(point_id)
    if pt is None:
        return {"error": "feature_store_missing"}
    trends = json.loads(pt.get("yearly_trends_json") or "[]")
    return {
        "years": [t["year"] for t in trends],
        "growing_season_rain_mm": [t.get("season_rain_mm") for t in trends],
        "mean_max_temp_c": [],
        "heat_days": [t.get("heat_days") for t in trends],
        "peak_ndvi": [t.get("peak_ndvi") for t in trends],
        "bare_days": [t.get("bare_days") for t in trends],
        "groundwater_depth_m": None,
        "mode": "live",
        "data_gaps": [],
    }


def field_info_response(lat: float, lon: float, point_id: int, dist_km: float) -> dict:
    pt = feature_point(point_id)
    if pt is None:
        return {"error": "feature_store_missing"}
    crop_history = json.loads(pt.get("crop_history_json") or "{}")
    last_years = sorted(crop_history.keys())[-3:]
    hist_short = {y: crop_history[y] for y in last_years}
    rain, heat, _, _ = _climate_baseline()
    bare = pt.get("median_bare_days_per_year")
    return {
        "nearest_point_id": point_id,
        "distance_km": round(dist_km, 2),
        "soil": {
            "texture_class": pt.get("texture_class") or "medium",
            "organic_matter_pct": pt.get("om_pct"),
            "awc": pt.get("awc"),
            "k_factor": pt.get("kwfact"),
        },
        "slope_pct": pt.get("slope_r"),
        "current_rotation_guess": pt.get("current_rotation_guess") or "other",
        "crop_history": hist_short,
        "bare_days_last_year": bare,
        "climate_summary": {
            "mean_growing_season_rain_mm": round(rain, 1),
            "mean_heat_days": round(heat, 1),
        },
        "mode": "live",
    }


def _confidence(
    rotation_id: str,
    interval: float | None,
    irrigated: bool,
    metrics: dict,
) -> str:
    ctx = CONTEXT_FOR_ROTATION.get(rotation_id, "other")
    counts = metrics.get("train_counts_rotation_context") or {}
    n = int(counts.get(ctx, counts.get(rotation_id, 0)) or 0)
    level = "low"
    if n >= 300 and interval is not None and interval < 0.12:
        level = "high"
    elif n >= 100 or (interval is not None and interval < 0.18):
        level = "medium"
    if irrigated:
        level = {"high": "medium", "medium": "low"}.get(level, "low")
    return level


def _priority_weights(priorities: list[str]) -> dict[str, float]:
    picks = priorities[:3] if priorities else DEFAULT_PRIORITIES
    weights = {k: 0.5 for k in PRIORITY_KEYS}
    for i, key in enumerate(picks):
        if key in weights:
            weights[key] = [3.0, 2.0, 1.0][i]
    return weights


def _health_score(p50: float | None) -> float | None:
    if p50 is None:
        return None
    return float(np.clip((p50 - 0.4) / 0.4 * 10, 0, 10))


def _total_score(priority_score: float | None, health: float | None) -> float | None:
    if priority_score is None or health is None:
        return None
    return round(0.7 * priority_score + 0.3 * health, 1)


def _weighted_priority_score(
    scores: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, dict[str, float | None], float]:
    """Null priority scores are omitted; remaining weights are renormalized (never treated as 0)."""
    active = {k: scores[k] for k in PRIORITY_KEYS if scores.get(k) is not None}
    if not active:
        return None, {k: None for k in PRIORITY_KEYS}, 0.0
    wsum = sum(weights[k] for k in active)
    if wsum <= 0:
        return None, {k: (weights[k] if k in active else None) for k in PRIORITY_KEYS}, 0.0
    eff_weights = {k: (weights[k] / wsum if k in active else None) for k in PRIORITY_KEYS}
    priority = sum(eff_weights[k] * active[k] for k in active if eff_weights[k] is not None)
    return float(priority), eff_weights, float(wsum)


def _pick_best_match(rotations_out: list[dict]) -> str:
    """Low-confidence rotations cannot win best_match."""
    scored = [r for r in rotations_out if r.get("scores", {}).get("total") is not None]
    if not scored:
        return "A"
    overall = max(scored, key=lambda r: r["scores"]["total"])
    eligible = [r for r in scored if str(r.get("confidence") or "low") != "low"]
    if str(overall.get("confidence") or "low") == "low":
        n = int(overall.get("training_n") or 0)
        msg = f"Promising but based on only {n} fields, confirm locally"
        why = list(overall.get("why") or [])
        if msg not in why:
            overall["why"] = why + [msg]
    if not eligible:
        return "A"
    best = max(eligible, key=lambda r: r["scores"]["total"])
    return str(best["id"])


def _compare_final(request_dict: dict, pid: int, dist_km: float, scenario: dict) -> dict | None:
    final = _load_final_model()
    if final is None:
        return None
    crop_tbl = _load_crop_table()
    year_type = str(request_dict.get("year_type", "normal"))
    rain, heat, et0, _ = _scenario_weather(year_type)
    pt = feature_point(pid) or {}

    irrigated = request_dict.get("water_source") == "irrigated"
    machinery = request_dict.get("machinery") or {}
    water_need: dict[str, float | None] = {}
    preds: dict[str, dict[str, float | None]] = {}
    bare_days: dict[str, float | None] = {}
    erosion_risk: dict[str, float | None] = {}
    nitrogen_fixing: dict[str, bool] = {}
    rotations_out: list[dict] = []

    kf = float(pt["kwfact"]) if pt.get("kwfact") is not None and pd.notna(pt.get("kwfact")) else None
    slope = float(pt["slope_r"]) if pt.get("slope_r") is not None and pd.notna(pt.get("slope_r")) else None

    for rid, meta in ROTATIONS.items():
        ctx = CONTEXT_FOR_ROTATION[rid]
        np.random.seed(pid * 1000 + hash(rid) % 10_000)
        p10, p50, p90 = _monte_carlo_ndvi(final, ctx, rain, heat, year_type=year_type)
        if p10 is not None:
            p10, p50, p90 = sorted([p10, p50, p90])
        preds[rid] = {"p10": p10, "p50": p50, "p90": p90}
        rot_meta = final["rotations"].get(ctx, {})
        bare = _bare_days_per_year(rot_meta, meta["crops"], meta["cycle_years"], crop_tbl)
        er = _erosion_risk_score(bare, kf, slope)
        water, pending = _water_need_mm(meta["crops"], et0, rain, crop_tbl, meta["cycle_years"])
        water_need[rid] = None if pending else water
        bare_days[rid] = bare
        erosion_risk[rid] = er
        n_fix = False
        if crop_tbl is not None:
            n_fix = any(
                str(crop_tbl.loc[c].get("nitrogen_fixing", "")).lower() == "yes"
                for c in meta["crops"]
                if c in crop_tbl.index
            )
        nitrogen_fixing[rid] = n_fix

    weights = _priority_weights(list(request_dict.get("priorities") or []))

    for rid, meta in ROTATIONS.items():
        scores = _absolute_priority_scores(
            bare_days_per_year=bare_days[rid],
            erosion_risk=erosion_risk[rid],
            water_need_mm=water_need[rid],
            crops=meta["crops"],
            crop_tbl=crop_tbl,
            nitrogen_fixing=nitrogen_fixing[rid],
        )
        if rid == "B" and not machinery.get("sorghum_planter", True) and scores.get("income") is not None:
            scores["income"] = max(0.0, scores["income"] - 1.0)
        if rid == "C" and not machinery.get("forage_equipment", True) and scores.get("forage") is not None:
            scores["forage"] = max(0.0, scores["forage"] - 1.0)
        priority_score, _, _ = _weighted_priority_score(scores, weights)
        health = _health_score(preds[rid]["p50"])
        scores["health"] = health
        scores["total"] = _total_score(priority_score, health)
        interval = None
        if preds[rid]["p10"] is not None and preds[rid]["p90"] is not None:
            interval = preds[rid]["p90"] - preds[rid]["p10"]
        data_gaps = []
        if water_need[rid] is None:
            data_gaps.append("Crop water data pending")
        if crop_tbl is None:
            data_gaps.append("Crop table missing — forage and market scores are data gaps.")
        rot_n = int(final["rotations"].get(CONTEXT_FOR_ROTATION[rid], {}).get("n", 0) or 0)
        rot_block = final["rotations"].get(CONTEXT_FOR_ROTATION[rid], {})
        if rot_n < 100:
            data_gaps.append(f"Fewer than 100 high-purity training rows for rotation {rid} (n={rot_n}).")
        if year_type in ("dry", "hot") and not _rotation_uses_dry_effect(rot_block):
            data_gaps.append("Not enough dry-year examples")
        risks = []
        if rid == "C":
            data_gaps.append("Rotation C has very few samples; treat as exploratory.")
        if rid == "B" and not machinery.get("sorghum_planter", True):
            risks.append("No sorghum planter declared — confirm feasibility.")
        if rid == "C" and not machinery.get("forage_equipment", True):
            risks.append("No forage equipment declared — confirm harvest plan.")
        if irrigated:
            data_gaps.append("Model trained on dryland fields; irrigation scenario is flagged.")
        why, evidence = _rotation_why_evidence(final, rid, preds[rid]["p50"])
        rotations_out.append(
            {
                "id": rid,
                "name": meta["name"],
                "is_current": rid == request_dict.get("current_rotation"),
                "crop_health": preds[rid],
                "water_need_mm": None if water_need[rid] is None else round(water_need[rid], 1),
                "bare_days_per_year": bare_days[rid],
                "bare_months": bare_months_for_rotation(rid),
                "erosion_risk": _erosion_risk_label(erosion_risk[rid]),
                "nitrogen_fixing": nitrogen_fixing[rid],
                "scores": scores,
                "confidence": _confidence_final(rid, final, interval, irrigated),
                "training_n": rot_n,
                "why": why,
                "evidence": evidence,
                "risks": risks,
                "data_gaps": data_gaps,
            }
        )

    best_match = _pick_best_match(rotations_out)
    for r in rotations_out:
        r.pop("training_n", None)

    return {
        "location": {"nearest_point_id": pid, "distance_km": round(dist_km, 2)},
        "scenario": scenario,
        "rotations": rotations_out,
        "best_match": best_match,
        "disclaimer": DISCLAIMER,
        "mode": "live",
        "model": "final",
    }


def compare(request_dict: dict) -> dict:
    """Return POST /compare response contract."""
    if "--self-test" in request_dict:
        if not _artifacts_ready():
            return {"error": "artifacts_not_ready"}
        pt = feature_point(0)
        assert pt is not None
        out = compare(
            {
                "lat": float(pt["lat"]),
                "lon": float(pt["lon"]),
                "current_rotation": "A",
                "year_type": "normal",
                "water_source": "dryland",
                "soil_texture": "medium",
                "priorities": ["water", "soil", "income"],
                "machinery": {"sorghum_planter": True, "forage_equipment": True},
            }
        )
        assert out.get("mode") == "live" and len(out.get("rotations", [])) == 3
        return out

    if not _artifacts_ready():
        return {"error": "artifacts_not_ready", "disclaimer": DISCLAIMER}

    near = nearest_point(float(request_dict["lat"]), float(request_dict["lon"]))
    if near is None or near[1] > 3.0:
        return {"error": "no_sample_point_within_3km", "disclaimer": DISCLAIMER}

    pid, dist_km = near
    scenario = _scenario(str(request_dict.get("year_type", "normal")))
    live = _compare_final(request_dict, pid, dist_km, scenario)
    if live is not None:
        return live
    return {"error": "model_not_available", "disclaimer": DISCLAIMER}


def compare_score_breakdown(request_dict: dict) -> dict | None:
    """Per-rotation scoring table for debugging (raw metrics, priority scores, renormalized weights)."""
    if not _artifacts_ready():
        return None
    near = nearest_point(float(request_dict["lat"]), float(request_dict["lon"]))
    if near is None or near[1] > 3.0:
        return None
    pid, _ = near
    year_type = str(request_dict.get("year_type", "normal"))
    scenario = _scenario(year_type)
    final = _load_final_model()
    if final is None:
        return None
    crop_tbl = _load_crop_table()
    rain, heat, et0, _ = _scenario_weather(year_type)
    pt = feature_point(pid) or {}
    kf = float(pt["kwfact"]) if pt.get("kwfact") is not None and pd.notna(pt.get("kwfact")) else None
    slope = float(pt["slope_r"]) if pt.get("slope_r") is not None and pd.notna(pt.get("slope_r")) else None
    machinery = request_dict.get("machinery") or {}
    weights = _priority_weights(list(request_dict.get("priorities") or []))

    water_need: dict[str, float | None] = {}
    erosion_risk: dict[str, float | None] = {}
    nitrogen_fixing: dict[str, bool] = {}
    preds: dict[str, dict[str, float | None]] = {}
    bare_days: dict[str, float] = {}

    for rid, meta in ROTATIONS.items():
        ctx = CONTEXT_FOR_ROTATION[rid]
        np.random.seed(pid * 1000 + hash(rid) % 10_000)
        p10, p50, p90 = _monte_carlo_ndvi(final, ctx, rain, heat, year_type=year_type)
        if p10 is not None:
            p10, p50, p90 = sorted([p10, p50, p90])
        preds[rid] = {"p10": p10, "p50": p50, "p90": p90}
        rot_meta = final["rotations"].get(ctx, {})
        bare = _bare_days_per_year(rot_meta, meta["crops"], meta["cycle_years"], crop_tbl)
        er = _erosion_risk_score(bare, kf, slope)
        water, pending = _water_need_mm(meta["crops"], et0, rain, crop_tbl, meta["cycle_years"])
        water_need[rid] = None if pending else water
        bare_days[rid] = bare
        erosion_risk[rid] = er
        n_fix = False
        if crop_tbl is not None:
            n_fix = any(
                str(crop_tbl.loc[c].get("nitrogen_fixing", "")).lower() == "yes"
                for c in meta["crops"]
                if c in crop_tbl.index
            )
        nitrogen_fixing[rid] = n_fix

    rows: list[dict] = []
    for rid, meta in ROTATIONS.items():
        scores = _absolute_priority_scores(
            bare_days_per_year=bare_days[rid],
            erosion_risk=erosion_risk[rid],
            water_need_mm=water_need[rid],
            crops=meta["crops"],
            crop_tbl=crop_tbl,
            nitrogen_fixing=nitrogen_fixing[rid],
        )
        if rid == "B" and not machinery.get("sorghum_planter", True) and scores.get("income") is not None:
            scores["income"] = max(0.0, scores["income"] - 1.0)
        if rid == "C" and not machinery.get("forage_equipment", True) and scores.get("forage") is not None:
            scores["forage"] = max(0.0, scores["forage"] - 1.0)
        priority_score, eff_w, active_wsum = _weighted_priority_score(scores, weights)
        health = _health_score(preds[rid]["p50"])
        total = _total_score(priority_score, health)
        raw_row = {
            "water_need_mm": water_need[rid],
            "bare_days_per_year": bare_days[rid],
            "erosion_risk_0_1": erosion_risk[rid],
            "nitrogen_fixing": nitrogen_fixing[rid],
        }
        rows.append(
            {
                "rotation": rid,
                "bare_days_per_year": round(bare_days[rid], 1),
                "p50": preds[rid]["p50"],
                "raw": raw_row,
                "priority_scores": scores,
                "weights_picked": weights,
                "weights_effective": eff_w,
                "active_weight_sum": active_wsum,
                "priority_score": priority_score,
                "health_score": health,
                "total": total,
            }
        )
    out = compare({**request_dict, "year_type": year_type})
    return {
        "point_id": pid,
        "year_type": year_type,
        "scenario": scenario,
        "priorities": list(request_dict.get("priorities") or DEFAULT_PRIORITIES),
        "pick_weights": weights,
        "rotations": rows,
        "best_match": out.get("best_match"),
        "confidence": {r["id"]: r["confidence"] for r in out.get("rotations", [])},
        "confidence_note": (
            "High/Medium/Low from training row count n (>=300 high, >=100 medium); "
            "wide p10-p90 intervals no longer downgrade high-n dryland rotations."
        ),
    }


def self_test() -> None:
    out = compare({"--self-test": True})
    assert out.get("mode") == "live" or out.get("error") == "artifacts_not_ready"
    print("compare self_test: ok")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        self_test()