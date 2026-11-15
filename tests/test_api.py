"""API contract tests (live artifacts required for compare)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from src.compare import _artifacts_ready  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["model_version"] == "final"
    if _artifacts_ready():
        assert data["status"] == "ok"
        assert data["mode"] == "live"
        assert data["artifacts_ready"] is True
    else:
        assert data["mode"] == "unavailable"


def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Field Shift" in r.text


def test_field_info_near_finney(client):
    if not _artifacts_ready():
        pytest.skip("artifacts missing")
    r = client.post("/field-info", json={"lat": 38.05, "lon": -100.86})
    assert r.status_code == 200
    data = r.json()
    assert "nearest_point_id" in data
    assert data["distance_km"] <= 3.0
    assert "soil" in data
    assert "climate_summary" in data


def test_field_info_too_far(client):
    r = client.post("/field-info", json={"lat": 0.0, "lon": 0.0})
    assert r.status_code == 404


def test_compare_live_contract(client):
    if not _artifacts_ready():
        pytest.skip("artifacts missing")
    r = client.post(
        "/compare",
        json={
            "lat": 38.05,
            "lon": -100.86,
            "current_rotation": "A",
            "water_source": "dryland",
            "soil_texture": "medium",
            "priorities": ["water", "soil", "income"],
            "year_type": "dry",
            "soil_om_pct": None,
            "machinery": {"sorghum_planter": True, "forage_equipment": False},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("mode") == "live"
    assert len(data["rotations"]) == 3
    assert data["rotations"][0]["crop_health"]["p50"] is not None
    assert "best_match" in data
    assert data["disclaimer"]
    assert data["scenario"]["year_type"] == "dry"
    assert data["scenario"]["rain_change_pct"] == -20


def test_trends(client):
    if not _artifacts_ready():
        pytest.skip("artifacts missing")
    r = client.get("/trends", params={"lat": 38.05, "lon": -100.86})
    assert r.status_code == 200
    data = r.json()
    assert "years" in data
    assert "growing_season_rain_mm" in data
    assert len(data["years"]) > 0
    assert data.get("mode") == "live"


def test_model_info(client):
    if not _artifacts_ready():
        pytest.skip("artifacts missing")
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "data_sources" in data
    assert "known_limitations" in data
    assert data["mode"] == "live"
    assert data.get("validation_summary")


def test_calendar(client):
    r = client.get("/calendar")
    if r.status_code == 503:
        pytest.skip("field_calendar.json not built")
    assert r.status_code == 200
    data = r.json()
    assert "rotations" in data
    assert "weather_by_month" in data
    assert "A" in data["rotations"]
    assert "12" in data["weather_by_month"]