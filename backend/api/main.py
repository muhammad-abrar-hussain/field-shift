"""Field Shift FastAPI — CURSOR_CONTEXT §11."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare import (  # noqa: E402
    FINAL_MODEL,
    _artifacts_ready,
    compare,
    field_info_response,
    load_field_calendar,
    nearest_point,
    trends_for_point,
)

PROJECT_ROOT = ROOT
WEB_DIR = ROOT.parent / "frontend"
REPORTS_DIR = ROOT / "reports"
ARCHIVE_METRICS = ROOT / "reports" / "archive" / "model_metrics_v2.json"
MAX_DIST_KM = 3.0
MODEL_VERSION = "final"

app = FastAPI(title="Field Shift API", version=MODEL_VERSION)
_CORS_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
# Extra origins for the deployed frontend (comma separated), e.g. the Vercel URL.
_CORS_ORIGINS += [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_SOURCES: list[dict[str, str]] = [
    {
        "name": "NASA POWER (MERRA-2 / CERES)",
        "url": "https://power.larc.nasa.gov/",
    },
    {
        "name": "NASA MODIS MOD13Q1 NDVI",
        "url": "https://lpdaac.usgs.gov/products/mod13q1v061/",
    },
    {
        "name": "NASA SMAP soil moisture",
        "url": "https://smap.jpl.nasa.gov/",
    },
    {
        "name": "NASA SRTM elevation",
        "url": "https://www2.jpl.nasa.gov/srtm/",
    },
    {
        "name": "USDA Cropland Data Layer",
        "url": "https://nassgeodata.gmu.edu/CropScape/",
    },
    {
        "name": "USDA SSURGO (Soil Data Access)",
        "url": "https://sdmdataaccess.sc.egov.usda.gov/",
    },
    {
        "name": "USDA NASS QuickStats",
        "url": "https://quickstats.nass.usda.gov/",
    },
]

KNOWN_LIMITATIONS = [
    "Dryland winter wheat focus; irrigated pivots excluded from training.",
    "Peas almost absent in Finney County; rotation C is low confidence.",
    "SMAP 9 km; MODIS NDVI 250 m.",
    "NASS yields are county level only.",
    "Prices in crop table are regional averages when provided.",
]


class FieldInfoRequest(BaseModel):
    lat: float
    lon: float


class MachineryIn(BaseModel):
    sorghum_planter: bool = True
    forage_equipment: bool = True


class CompareRequest(BaseModel):
    lat: float
    lon: float
    current_rotation: Literal["A", "B", "C", "other"] = "A"
    water_source: Literal["dryland", "irrigated"] = "dryland"
    soil_texture: Literal["light", "medium", "heavy"] = "medium"
    priorities: list[Literal["water", "income", "soil", "erosion", "forage", "market"]] = Field(
        default_factory=lambda: ["water", "soil", "income"]
    )
    year_type: Literal["normal", "dry", "hot"] = "normal"
    soil_om_pct: float | None = None
    machinery: MachineryIn = Field(default_factory=MachineryIn)


def _resolve_nearest(lat: float, lon: float) -> tuple[int, float]:
    near = nearest_point(lat, lon)
    if near is None:
        raise HTTPException(status_code=404, detail="No sample points loaded.")
    pid, dist = near
    if dist > MAX_DIST_KM:
        raise HTTPException(
            status_code=404,
            detail=f"No sample point within {MAX_DIST_KM} km (nearest is {dist:.1f} km away).",
        )
    return pid, dist


def _load_nass_validation() -> dict[str, Any] | None:
    if FINAL_MODEL.is_file():
        final = json.loads(FINAL_MODEL.read_text(encoding="utf-8"))
        if final.get("nass_validation"):
            return final["nass_validation"]
    if ARCHIVE_METRICS.is_file():
        data = json.loads(ARCHIVE_METRICS.read_text(encoding="utf-8"))
        return data.get("nass_validation")
    return None


@app.get("/health")
def health():
    ready = _artifacts_ready()
    return {
        "status": "ok" if ready else "unavailable",
        "model_version": MODEL_VERSION,
        "mode": "live" if ready else "unavailable",
        "artifacts_ready": ready,
    }


@app.post("/field-info")
def field_info(body: FieldInfoRequest):
    pid, dist = _resolve_nearest(body.lat, body.lon)
    out = field_info_response(body.lat, body.lon, pid, dist)
    if out.get("error"):
        raise HTTPException(status_code=503, detail=out["error"])
    return out


@app.post("/compare")
def compare_endpoint(body: CompareRequest):
    if not _artifacts_ready():
        raise HTTPException(
            status_code=503,
            detail="Trained model artifacts missing (models/final_model.json and feature_store.parquet).",
        )
    req = body.model_dump()
    result = compare(req)
    if result.get("error"):
        status = 503 if result["error"] in ("artifacts_not_ready", "model_not_available") else 404
        raise HTTPException(status_code=status, detail=result["error"])
    return result


@app.get("/trends")
def trends(lat: float = Query(...), lon: float = Query(...)):
    if not _artifacts_ready():
        raise HTTPException(status_code=503, detail="feature_store_missing")
    pid, _ = _resolve_nearest(lat, lon)
    data = trends_for_point(pid)
    if data.get("error"):
        raise HTTPException(status_code=503, detail=data["error"])
    return data


@app.get("/calendar")
def calendar():
    data = load_field_calendar()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="field_calendar.json missing — run src/09_field_calendar.py",
        )
    return data


@app.get("/model-info")
def model_info():
    ready = _artifacts_ready()
    evaluation_loo: dict[str, Any] | None = None
    nass_validation = _load_nass_validation()
    if FINAL_MODEL.is_file():
        final = json.loads(FINAL_MODEL.read_text(encoding="utf-8"))
        evaluation_loo = final.get("evaluation_loo")
    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not available on this server.",
        )
    summary = None
    if evaluation_loo:
        summary = {
            "loo_mae_mc_p50": evaluation_loo.get("mae_mc_p50"),
            "loo_mae_deterministic": evaluation_loo.get("mae_deterministic"),
            "fair_baseline_mae_no_weather": evaluation_loo.get("mae_baseline_fair_no_weather"),
            "beats_fair_baseline": evaluation_loo.get("beats_baseline_fair"),
            "coverage_p10_p90": evaluation_loo.get("coverage_p10_p90"),
            "conformal_delta": evaluation_loo.get("conformal_delta"),
        }
    nass_summary = None
    if nass_validation:
        r = nass_validation.get("pearson_r_observed_county_ndvi_vs_yield")
        nass_summary = {
            "pearson_r_observed_ndvi_vs_yield": r,
            "n_years": nass_validation.get("n_years"),
            "years": nass_validation.get("years"),
            "disclaimer": nass_validation.get("disclaimer"),
        }
    return {
        "mode": "live",
        "model": "final",
        "evaluation_loo": evaluation_loo,
        "validation_summary": summary,
        "nass_validation": nass_summary or nass_validation,
        "data_sources": DATA_SOURCES,
        "known_limitations": KNOWN_LIMITATIONS,
    }


if REPORTS_DIR.is_dir():
    app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def self_test() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.get("/health").json()["status"] in ("ok", "unavailable")
    print("api self_test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)