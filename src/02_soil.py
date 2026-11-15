# Query SSURGO via USDA Soil Data Access for each sample point; cache JSON and write soil_points.csv.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTS_CSV = PROJECT_ROOT / "data" / "processed" / "sample_points.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "soil" / "cache"
OUT_CSV = PROJECT_ROOT / "data" / "raw" / "soil" / "soil_points.csv"

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
REQUEST_TIMEOUT_S = 120
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 5
REQUEST_PAUSE_S = 0.15
DEPTH_TOP_CM = 0
DEPTH_BOTTOM_CM = 30


def _query_sql(lon: float, lat: float) -> str:
    wkt = f"POINT({lon} {lat})"
    return f"""
SELECT dom.mukey, dom.cokey, dom.compname, dom.comppct_r, dom.slope_r, dom.hydgrp,
       ch.hzdept_r, ch.hzdepb_r, ch.om_r, ch.sandtotal_r, ch.silttotal_r, ch.claytotal_r,
       ch.awc_r, ch.kwfact
FROM (
  SELECT TOP 1 mu.mukey, co.cokey, co.compname, co.comppct_r, co.slope_r, co.hydgrp
  FROM mapunit mu
  INNER JOIN component co ON co.mukey = mu.mukey
  WHERE mu.mukey IN (
    SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}')
  )
  ORDER BY co.comppct_r DESC
) dom
INNER JOIN chorizon ch ON ch.cokey = dom.cokey
WHERE ch.hzdept_r < {DEPTH_BOTTOM_CM}
ORDER BY ch.hzdept_r
"""


def _parse_table_response(payload: dict) -> list[dict]:
    table = payload.get("Table")
    if not table or len(table) < 2:
        return []
    headers = table[0]
    rows = table[2:]
    return [dict(zip(headers, row)) for row in rows]


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hz_thickness_cm(hzdept_r, hzdepb_r) -> float:
    top = _to_float(hzdept_r) or 0.0
    top = max(DEPTH_TOP_CM, top)
    bottom = _to_float(hzdepb_r)
    if bottom is None:
        bottom = float(DEPTH_BOTTOM_CM)
    else:
        bottom = min(DEPTH_BOTTOM_CM, bottom)
    return max(0.0, bottom - top)


def _weighted_0_30(hz_rows: list[dict], field: str) -> float | None:
    weighted_sum = 0.0
    thickness_sum = 0.0
    for row in hz_rows:
        thick = _hz_thickness_cm(row.get("hzdept_r"), row.get("hzdepb_r"))
        val = _to_float(row.get(field))
        if thick > 0 and val is not None:
            weighted_sum += val * thick
            thickness_sum += thick
    if thickness_sum <= 0:
        return None
    return weighted_sum / thickness_sum


def _summarize_rows(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    dom = rows[0]
    weighted = {
        "om_0_30_pct": _weighted_0_30(rows, "om_r"),
        "sand_0_30_pct": _weighted_0_30(rows, "sandtotal_r"),
        "silt_0_30_pct": _weighted_0_30(rows, "silttotal_r"),
        "clay_0_30_pct": _weighted_0_30(rows, "claytotal_r"),
        "awc_0_30": _weighted_0_30(rows, "awc_r"),
        "kwfact_0_30": _weighted_0_30(rows, "kwfact"),
    }
    if all(v is None for v in weighted.values()):
        return None
    return {
        "mukey": dom.get("mukey"),
        "cokey": dom.get("cokey"),
        "compname": dom.get("compname"),
        "comppct_r": _to_float(dom.get("comppct_r")),
        "slope_r_pct": _to_float(dom.get("slope_r")),
        "hydrologic_soil_group": dom.get("hydgrp"),
        "horizons_used": len(rows),
        **weighted,
    }


def _fetch_sda(lon: float, lat: float) -> tuple[list[dict], str | None]:
    sql = _query_sql(lon, lat)
    last_err: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                SDA_URL,
                data={"query": sql, "format": "JSON+COLUMNNAME"},
                timeout=REQUEST_TIMEOUT_S,
            )
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                raise requests.HTTPError(last_err)
            rows = _parse_table_response(resp.json())
            return rows, None
        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            last_err = str(exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_S)
    return [], last_err


def _cache_path(point_id: int) -> Path:
    return CACHE_DIR / f"{point_id}.json"


def _load_cache(point_id: int) -> list[dict] | None:
    path = _cache_path(point_id)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rows")


def _save_cache(point_id: int, rows: list[dict], error: str | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "error": error}
    with open(_cache_path(point_id), "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main() -> int:
    if not POINTS_CSV.is_file():
        print(f"Missing {POINTS_CSV}", file=sys.stderr)
        return 1

    points = pd.read_csv(POINTS_CSV)
    id_col = "ID" if "ID" in points.columns else "point_id"
    lat_col = "Latitude" if "Latitude" in points.columns else "lat"
    lon_col = "Longitude" if "Longitude" in points.columns else "lon"

    records: list[dict] = []
    no_data: list[int] = []

    for i, row in points.iterrows():
        pid = int(row[id_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        cached = _load_cache(pid)
        if cached is None:
            hz_rows, err = _fetch_sda(lon, lat)
            _save_cache(pid, hz_rows, err)
            time.sleep(REQUEST_PAUSE_S)
        else:
            hz_rows = cached

        summary = _summarize_rows(hz_rows)
        base = {
            "point_id": pid,
            "latitude": lat,
            "longitude": lon,
            "status": "ok" if summary else "no_data",
        }
        if summary:
            records.append({**base, **summary})
        else:
            no_data.append(pid)
            records.append({**base, **{k: None for k in (
                "mukey", "cokey", "compname", "comppct_r", "slope_r_pct",
                "hydrologic_soil_group", "horizons_used",
                "om_0_30_pct", "sand_0_30_pct", "silt_0_30_pct", "clay_0_30_pct",
                "awc_0_30", "kwfact_0_30",
            )}})

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(points)} points...")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV} ({len(out_df)} rows, {len(out_df) - len(no_data)} with soil data)")
    if no_data:
        print(f"Points with no SSURGO data ({len(no_data)}): {no_data}")
    else:
        print("All points returned dominant-component soil data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())