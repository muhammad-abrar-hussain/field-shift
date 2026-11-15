# Tidy AppEEARS or Google Earth Engine point CSVs into NDVI, SMAP, and elevation tables.

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPEEARS_DIR = PROJECT_ROOT / "data" / "raw" / "appeears"
GEE_DIR = PROJECT_ROOT / "data" / "raw" / "gee"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

NDVI_GEE_FILE = "finney_ndvi_gee.csv"
SMAP_GEE_FILE = "finney_smap_gee.csv"
SRTM_GEE_FILE = "finney_srtm_gee.csv"

NDVI_SCALE = 0.0001
NDVI_VALID_RELIABILITY = {0, 1}
SMAP_FILL = -9999
SMAP_SM_MIN, SMAP_SM_MAX = 0.0, 0.6


def _find_column(columns: list[str], *substrings: str) -> str | None:
    for col in columns:
        low = col.lower()
        if all(s.lower() in low for s in substrings):
            return col
    return None


def _find_id_column(columns: list[str]) -> str | None:
    for col in columns:
        if col.lower() in ("id", "point_id"):
            return col
    return None


def _find_date_column(columns: list[str]) -> str | None:
    for col in columns:
        if col.lower() in ("date", "datetime", "time"):
            return col
    return None


def _normalize_id_date(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _find_id_column(list(df.columns))
    if id_col is None:
        raise ValueError("No ID column found")
    out = df.copy()
    out = out.rename(columns={id_col: "point_id"})
    out["point_id"] = pd.to_numeric(out["point_id"], errors="coerce").astype("Int64")
    date_col = _find_date_column(list(df.columns))
    if date_col:
        out = out.rename(columns={date_col: "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _scale_ndvi(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().any() and v.max() > 1:
        v = v * NDVI_SCALE
    return v


def parse_gee_ndvi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_id_date(df)
    ndvi_col = _find_column(list(df.columns), "ndvi") or "NDVI"
    qa_col = _find_column(list(df.columns), "summaryqa") or _find_column(list(df.columns), "summary", "qa")
    if ndvi_col not in df.columns:
        raise ValueError(f"{path.name}: missing NDVI column")
    out = df[["point_id", "date", ndvi_col]].rename(columns={ndvi_col: "ndvi"})
    out["ndvi"] = _scale_ndvi(out["ndvi"])
    if qa_col and qa_col in df.columns:
        qa = pd.to_numeric(df[qa_col], errors="coerce")
        out = out[qa.isin(NDVI_VALID_RELIABILITY)]
    out = out.dropna(subset=["point_id", "date", "ndvi"])
    return out.drop_duplicates(subset=["point_id", "date"], keep="last").sort_values(["point_id", "date"])


def _first_data_column(columns: list[str], skip: set[str]) -> str | None:
    for col in columns:
        if col not in skip and col.lower() not in skip:
            return col
    return None


def parse_gee_smap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_id_date(df)
    sm_col = _find_column(list(df.columns), "soil_moisture")
    if sm_col is None:
        sm_col = _first_data_column(
            list(df.columns),
            {c for c in df.columns if c.lower() in ("id", "point_id", "date", "datetime", "time")},
        )
    if sm_col is None:
        raise ValueError(f"{path.name}: no soil moisture column")
    out = df[["point_id", "date", sm_col]].rename(columns={sm_col: "sm"})
    out["sm"] = pd.to_numeric(out["sm"], errors="coerce")
    out = out[out["sm"] != SMAP_FILL]
    out = out[(out["sm"] >= SMAP_SM_MIN) & (out["sm"] <= SMAP_SM_MAX)]
    out = out.dropna(subset=["point_id", "date", "sm"])
    return out.drop_duplicates(subset=["point_id", "date"], keep="last").sort_values(["point_id", "date"])


def parse_gee_elevation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = _find_id_column(list(df.columns))
    if id_col is None:
        raise ValueError(f"{path.name}: missing ID")
    elev_col = _find_column(list(df.columns), "elevation") or _find_column(list(df.columns), "dem")
    if elev_col is None:
        elev_col = _first_data_column(list(df.columns), {id_col})
    if elev_col is None:
        raise ValueError(f"{path.name}: missing elevation column")
    out = pd.DataFrame(
        {
            "point_id": pd.to_numeric(df[id_col], errors="coerce"),
            "elevation_m": pd.to_numeric(df[elev_col], errors="coerce"),
        }
    )
    out = out.dropna()
    out["point_id"] = out["point_id"].astype(int)
    return out.groupby("point_id", as_index=False)["elevation_m"].mean().sort_values("point_id")


def parse_mod13_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["point_id", "date", "ndvi"])

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df = _normalize_id_date(df)
        ndvi_col = _find_column(list(df.columns), "ndvi")
        rel_col = _find_column(list(df.columns), "pixel_reliability")
        if ndvi_col is None:
            print(f"MOD13Q1: no NDVI column in {path.name} — skipped.")
            continue
        chunk = df[["point_id", "date", ndvi_col]].rename(columns={ndvi_col: "ndvi"})
        chunk["ndvi"] = _scale_ndvi(chunk["ndvi"])
        if rel_col is not None:
            chunk["_rel"] = pd.to_numeric(df[rel_col], errors="coerce")
            chunk = chunk[chunk["_rel"].isin(NDVI_VALID_RELIABILITY)]
            chunk = chunk.drop(columns="_rel")
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["point_id", "date", "ndvi"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["point_id", "date", "ndvi"])
    out = out.drop_duplicates(subset=["point_id", "date"], keep="last")
    return out.sort_values(["point_id", "date"])


def parse_smap_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["point_id", "date", "sm"])

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df = _normalize_id_date(df)
        sm_col = _find_column(list(df.columns), "soil_moisture")
        if sm_col is None:
            print(f"SMAP: no soil_moisture column in {path.name} — skipped.")
            continue
        chunk = df[["point_id", "date", sm_col]].rename(columns={sm_col: "sm"})
        chunk["sm"] = pd.to_numeric(chunk["sm"], errors="coerce")
        chunk = chunk[chunk["sm"] != SMAP_FILL]
        chunk = chunk[(chunk["sm"] >= SMAP_SM_MIN) & (chunk["sm"] <= SMAP_SM_MAX)]
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["point_id", "date", "sm"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["point_id", "date", "sm"])
    out = out.drop_duplicates(subset=["point_id", "date"], keep="last")
    return out.sort_values(["point_id", "date"])


def parse_srtm_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["point_id", "elevation_m"])

    rows: list[dict] = []
    for path in paths:
        df = pd.read_csv(path)
        id_col = _find_id_column(list(df.columns))
        dem_col = _find_column(list(df.columns), "dem")
        if id_col is None or dem_col is None:
            print(f"SRTM: missing ID or DEM column in {path.name} — skipped.")
            continue
        for _, r in df.iterrows():
            pid = pd.to_numeric(r[id_col], errors="coerce")
            elev = pd.to_numeric(r[dem_col], errors="coerce")
            if pd.notna(pid) and pd.notna(elev):
                rows.append({"point_id": int(pid), "elevation_m": float(elev)})

    if not rows:
        return pd.DataFrame(columns=["point_id", "elevation_m"])

    out = pd.DataFrame(rows)
    return out.groupby("point_id", as_index=False)["elevation_m"].mean()


def _pick_ndvi(gee_dir: Path, appeears_dir: Path) -> tuple[pd.DataFrame, str]:
    gee_path = gee_dir / NDVI_GEE_FILE
    if gee_path.is_file():
        return parse_gee_ndvi(gee_path), "GEE"
    csvs = sorted(appeears_dir.glob("*.csv")) if appeears_dir.is_dir() else []
    mod13 = [p for p in csvs if re.search(r"MOD13Q1", p.name, re.I)]
    if mod13:
        return parse_mod13_files(mod13), "AppEEARS"
    print("NDVI: no GEE or AppEEARS source found.")
    return pd.DataFrame(columns=["point_id", "date", "ndvi"]), "none"


def _pick_smap(gee_dir: Path, appeears_dir: Path) -> tuple[pd.DataFrame, str]:
    gee_path = gee_dir / SMAP_GEE_FILE
    if gee_path.is_file():
        return parse_gee_smap(gee_path), "GEE"
    csvs = sorted(appeears_dir.glob("*.csv")) if appeears_dir.is_dir() else []
    smap = [p for p in csvs if re.search(r"SPL3SMP", p.name, re.I)]
    if smap:
        return parse_smap_files(smap), "AppEEARS"
    print("SMAP: no GEE or AppEEARS source found.")
    return pd.DataFrame(columns=["point_id", "date", "sm"]), "none"


def _pick_elevation(gee_dir: Path, appeears_dir: Path) -> tuple[pd.DataFrame, str]:
    gee_path = gee_dir / SRTM_GEE_FILE
    if gee_path.is_file():
        return parse_gee_elevation(gee_path), "GEE"
    csvs = sorted(appeears_dir.glob("*.csv")) if appeears_dir.is_dir() else []
    srtm = [p for p in csvs if re.search(r"SRTMGL1", p.name, re.I)]
    if srtm:
        return parse_srtm_files(srtm), "AppEEARS"
    print("Elevation: no GEE or AppEEARS source found.")
    return pd.DataFrame(columns=["point_id", "elevation_m"]), "none"


def _date_range_str(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return "n/a"
    return f"{df['date'].min().date()} .. {df['date'].max().date()}"


def run_parse(gee_dir: Path = GEE_DIR, appeears_dir: Path = APPEEARS_DIR) -> int:
    ndvi, ndvi_src = _pick_ndvi(gee_dir, appeears_dir)
    sm, sm_src = _pick_smap(gee_dir, appeears_dir)
    elev, elev_src = _pick_elevation(gee_dir, appeears_dir)

    print("Sources: NDVI =", ndvi_src, "| SMAP =", sm_src, "| elevation =", elev_src)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ndvi.to_parquet(PROC_DIR / "ndvi_points.parquet", index=False)
    sm.to_parquet(PROC_DIR / "smap_points.parquet", index=False)
    elev.to_csv(PROC_DIR / "elev_points.csv", index=False)

    print(f"ndvi_points: {len(ndvi):,} rows, dates {_date_range_str(ndvi)}")
    print(f"smap_points: {len(sm):,} rows, dates {_date_range_str(sm)}")
    print(f"elev_points: {len(elev):,} rows")
    return 0 if any(src != "none" for src in (ndvi_src, sm_src, elev_src)) else 1


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gee = tmp_path / "gee"
        appeears = tmp_path / "appeears"
        gee.mkdir()
        appeears.mkdir()

        pd.DataFrame(
            {
                "ID": [0, 0, 1],
                "date": ["2024-03-01", "2024-04-01", "2024-03-15"],
                "NDVI": [5500, 6000, 4000],
                "SummaryQA": [0, 1, 2],
            }
        ).to_csv(gee / NDVI_GEE_FILE, index=False)

        pd.DataFrame({"ID": [0], "date": ["2023-10-01"], "soil_moisture_am": [0.25]}).to_csv(
            gee / SMAP_GEE_FILE, index=False
        )
        pd.DataFrame({"ID": [0, 1], "elevation": [920.0, 930.5]}).to_csv(gee / SRTM_GEE_FILE, index=False)

        pd.DataFrame(
            {
                "ID": [0],
                "Date": ["2024-05-01"],
                "250m_16_days_NDVI": [9999],
                "250m_16_days_pixel_reliability": [0],
            }
        ).to_csv(appeears / "task_MOD13Q1_finney.csv", index=False)

        ndvi, src = _pick_ndvi(gee, appeears)
        assert src == "GEE"
        assert len(ndvi) == 2
        assert np.isclose(ndvi["ndvi"].iloc[0], 0.55)

        sm, sm_src = _pick_smap(gee, appeears)
        assert sm_src == "GEE" and len(sm) == 1

        elev, el_src = _pick_elevation(gee, appeears)
        assert el_src == "GEE" and len(elev) == 2

        ndvi_only = parse_gee_ndvi(gee / NDVI_GEE_FILE)
        assert len(ndvi_only) == 2

    print("03_parse_appeears self_test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    sys.exit(run_parse())