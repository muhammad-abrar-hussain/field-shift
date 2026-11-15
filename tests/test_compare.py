"""Compare logic and live stack tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compare import (  # noqa: E402
    FINAL_MODEL,
    FEATURE_STORE,
    FIELD_CALENDAR_JSON,
    _artifacts_ready,
    compare,
    feature_point,
)


@pytest.mark.skipif(not FEATURE_STORE.is_file(), reason="feature_store missing")
def test_feature_store_point0():
    pt = feature_point(0)
    assert pt is not None
    assert pt["point_id"] == 0
    assert "yearly_trends_json" in pt
    trends = json.loads(pt["yearly_trends_json"])
    assert len(trends) >= 10


@pytest.mark.skipif(not _artifacts_ready(), reason="live artifacts missing")
def test_compare_point0_normal_and_dry():
    pts = feature_point(0)
    lat, lon = float(pts["lat"]), float(pts["lon"])
    base = {
        "lat": lat,
        "lon": lon,
        "current_rotation": "A",
        "water_source": "dryland",
        "priorities": ["water", "soil", "income"],
        "machinery": {"sorghum_planter": True, "forage_equipment": True},
    }
    for yt in ("normal", "dry"):
        out = compare({**base, "year_type": yt, "soil_texture": "medium"})
        assert out.get("mode") == "live"
        assert len(out["rotations"]) == 3
        assert out["rotations"][0]["crop_health"]["p50"] is not None
        assert out["scenario"]["year_type"] == yt


@pytest.mark.skipif(not FINAL_MODEL.is_file(), reason="final model missing")
def test_final_model_has_nass():
    data = json.loads(FINAL_MODEL.read_text(encoding="utf-8"))
    assert "evaluation_loo" in data


@pytest.mark.skipif(not _artifacts_ready(), reason="live artifacts missing")
def test_compare_includes_bare_months():
    pt = feature_point(0)
    out = compare(
        {
            "lat": float(pt["lat"]),
            "lon": float(pt["lon"]),
            "current_rotation": "A",
            "year_type": "normal",
        }
    )
    for rot in out["rotations"]:
        assert "bare_months" in rot
        assert isinstance(rot["bare_months"], list)


@pytest.mark.skipif(not FIELD_CALENDAR_JSON.is_file(), reason="calendar missing")
def test_field_calendar_structure():
    data = json.loads(FIELD_CALENDAR_JSON.read_text(encoding="utf-8"))
    assert data["rotations"]["A"]["mean_ndvi_by_month"]
    assert data["low_cover_ndvi_threshold"] == 0.25
    assert len(data["weather_by_month"]) == 12