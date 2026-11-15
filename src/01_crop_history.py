# Build crop rotation history and stratified sample points from Finney County CDL rasters (2013–2025).

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CDL_DIR = PROJECT_ROOT / "data" / "raw" / "cdl"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

YEARS = list(range(2013, 2026))
REF_YEAR = 2024
FIPS_SUFFIX = "20055"
RANDOM_SEED = 42
MIN_DISTANCE_M = 300
WHEAT_FALLOW_CONTEXT_FRAC = 0.60
TEST_YEARS_FROM = 2022
STRATA_CAP = 350
MAX_POINTS = 1_000

WHEAT = {24, 236}
FALLOW = 61
FORAGE = {205, 37, 53}
IRRIGATED = {1, 36, 5}
GRASSLAND = 176

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


def _cdl_path(year: int) -> Path:
    return CDL_DIR / f"CDL_{year}_{FIPS_SUFFIX}.tif"


def _load_reference_meta() -> tuple[rasterio.Affine, rasterio.crs.CRS, int, int]:
    with rasterio.open(_cdl_path(REF_YEAR)) as ref:
        return ref.transform, ref.crs, ref.height, ref.width


def _read_aligned_year(year: int, ref_transform, ref_crs, height: int, width: int) -> np.ndarray:
    with rasterio.open(_cdl_path(year)) as src:
        if src.height == height and src.width == width and src.transform == ref_transform:
            return src.read(1)
        out = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )
        return out


def _load_stack() -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    ref_transform, ref_crs, height, width = _load_reference_meta()
    layers = []
    aligned_years = []
    for year in YEARS:
        data = _read_aligned_year(year, ref_transform, ref_crs, height, width)
        layers.append(data)
        with rasterio.open(_cdl_path(year)) as src:
            if src.height != height or src.width != width or src.transform != ref_transform:
                aligned_years.append(year)
    if aligned_years:
        print(f"Aligned to {REF_YEAR} grid (nearest): {aligned_years}")
    stack = np.stack(layers, axis=0)
    return stack, ref_transform, ref_crs


def _pixel_xy(transform: rasterio.Affine, row: int, col: int) -> tuple[float, float]:
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    return float(x), float(y)


def _is_wheat(code: int) -> bool:
    return int(code) in WHEAT


def rotation_context(prev1: int, prev2: int) -> str:
    if int(prev1) == FALLOW and _is_wheat(prev2):
        return "A"
    if int(prev1) == FALLOW and int(prev2) == 4:
        return "B"
    if int(prev1) in FORAGE:
        return "C"
    return "other"


def classify_system(seq: np.ndarray) -> str:
    s = [int(c) for c in seq]
    for i in range(len(s) - 2):
        if _is_wheat(s[i]) and s[i + 1] == 4 and s[i + 2] == FALLOW:
            return "wheat_sorghum_fallow"
    for i in range(len(s) - 1):
        if _is_wheat(s[i]) and s[i + 1] in FORAGE:
            return "wheat_forage"
    if 4 not in s:
        contexts: list[str] = []
        for i, code in enumerate(s):
            if not _is_wheat(code):
                continue
            p1 = s[i - 1] if i >= 1 else 0
            p2 = s[i - 2] if i >= 2 else 0
            contexts.append(rotation_context(p1, p2))
        if contexts:
            a_frac = sum(1 for c in contexts if c == "A") / len(contexts)
            if a_frac >= WHEAT_FALLOW_CONTEXT_FRAC:
                return "wheat_fallow"
    return "other"


def _greedy_min_distance(
    rows: np.ndarray,
    cols: np.ndarray,
    transform: rasterio.Affine,
    *,
    min_dist_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(rows))
    rows, cols = rows[order], cols[order]
    cell_px = max(1, int(min_dist_m / 30))
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    out_r: list[int] = []
    out_c: list[int] = []

    for row, col in zip(rows, cols):
        x, y = _pixel_xy(transform, int(row), int(col))
        cx, cy = int(row) // cell_px, int(col) // cell_px
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for x2, y2 in buckets.get((cx + dr, cy + dc), []):
                    if (x - x2) ** 2 + (y - y2) ** 2 < min_dist_m**2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            out_r.append(int(row))
            out_c.append(int(col))
            buckets[(cx, cy)].append((x, y))

    return np.array(out_r, dtype=np.int32), np.array(out_c, dtype=np.int32)


def _stratified_select(
    systems: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    order: list[int] = []
    for sys_name in ("wheat_fallow", "wheat_sorghum_fallow", "wheat_forage"):
        idx = np.where(systems == sys_name)[0]
        rng.shuffle(idx)
        order.extend(idx[: min(STRATA_CAP, len(idx))].tolist())

    if len(order) < MAX_POINTS:
        other_idx = np.where(systems == "other")[0]
        rng.shuffle(other_idx)
        need = MAX_POINTS - len(order)
        order.extend(other_idx[:need].tolist())

    return np.array(order[:MAX_POINTS], dtype=np.int32)


def _crop_label(code: int) -> str:
    return CDL_NAMES.get(int(code), f"cdl_{int(code)}")


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stack, transform, crs = _load_stack()
    n_years, height, width = stack.shape

    in_county = (stack != 0).any(axis=0)
    wheat_years = np.isin(stack, list(WHEAT)).sum(axis=0)
    mask_wheat2 = wheat_years >= 2
    after_wheat = in_county & mask_wheat2
    print(f"After in-county + wheat in >=2 years: {int(after_wheat.sum()):,}")

    irrigated = np.isin(stack, list(IRRIGATED)).any(axis=0)
    after_irr = after_wheat & ~irrigated
    print(f"After removing corn/alfalfa/soy any year: {int(after_irr.sum()):,}")

    grass_counts = (stack == GRASSLAND).sum(axis=0)
    after_grass = after_irr & (grass_counts <= 6)
    print(f"After removing grassland >6 years: {int(after_grass.sum()):,}")

    rows, cols = np.where(after_grass)
    print(f"Eligible pixels entering spacing filter: {len(rows):,}")
    rng = np.random.default_rng(RANDOM_SEED)

    cand_r, cand_c = _greedy_min_distance(
        rows, cols, transform, min_dist_m=MIN_DISTANCE_M, rng=rng
    )
    print(f"After {MIN_DISTANCE_M} m spacing (seed={RANDOM_SEED}): {len(cand_r):,} candidates")

    systems = np.array([classify_system(stack[:, r, c]) for r, c in zip(cand_r, cand_c)], dtype=object)
    sys_counts = Counter(systems)
    print("Candidates by system:", dict(sys_counts))

    pick_idx = _stratified_select(systems, rng)
    final_r = cand_r[pick_idx]
    final_c = cand_c[pick_idx]
    final_sys = systems[pick_idx]
    final_counts = Counter(final_sys)
    print("Final points per system:")
    for name in ("wheat_fallow", "wheat_sorghum_fallow", "wheat_forage", "other"):
        print(f"  {name}: {final_counts.get(name, 0)}")
    print(f"  total: {len(final_r)}")

    to_4326 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    history_rows = []
    for pid, (row, col, system) in enumerate(zip(final_r, final_c, final_sys)):
        x, y = _pixel_xy(transform, int(row), int(col))
        lon, lat = to_4326.transform(x, y)
        seq = stack[:, row, col]
        rec = {
            "point_id": pid,
            "x": round(x, 3),
            "y": round(y, 3),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "system": system,
        }
        for yi, year in enumerate(YEARS):
            rec[f"cdl_{year}"] = int(seq[yi])
        history_rows.append(rec)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(PROC_DIR / "crop_history.csv", index=False)

    wheat_rows = []
    for _, rec in history_df.iterrows():
        pid = int(rec["point_id"])
        seq = [int(rec[f"cdl_{y}"]) for y in YEARS]
        for yi, year in enumerate(YEARS):
            if not _is_wheat(seq[yi]):
                continue
            prev1 = seq[yi - 1] if yi >= 1 else 0
            prev2 = seq[yi - 2] if yi >= 2 else 0
            wheat_rows.append(
                {
                    "point_id": pid,
                    "year": year,
                    "prev1_crop": _crop_label(prev1),
                    "prev2_crop": _crop_label(prev2),
                    "rotation_context": rotation_context(prev1, prev2),
                }
            )

    wheat_df = pd.DataFrame(wheat_rows)
    wheat_df.to_csv(PROC_DIR / "wheat_years.csv", index=False)

    ctx_all = Counter(wheat_df["rotation_context"])
    ctx_test = Counter(wheat_df.loc[wheat_df["year"] >= TEST_YEARS_FROM, "rotation_context"])
    print("Wheat-year counts by rotation_context (all years):")
    for key in ("A", "B", "C", "other"):
        print(f"  {key}: {ctx_all.get(key, 0)}")
    print(f"Wheat-year counts by rotation_context (years >= {TEST_YEARS_FROM}):")
    for key in ("A", "B", "C", "other"):
        print(f"  {key}: {ctx_test.get(key, 0)}")

    sample_df = pd.DataFrame(
        {
            "ID": history_df["point_id"],
            "Category": history_df["system"],
            "Latitude": history_df["lat"],
            "Longitude": history_df["lon"],
        }
    )
    sample_df.to_csv(PROC_DIR / "sample_points.csv", index=False)

    cdl_2024 = stack[YEARS.index(REF_YEAR)]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cdl_2024, cmap="gist_ncar", interpolation="nearest")
    colors = {
        "wheat_fallow": "#e41a1c",
        "wheat_sorghum_fallow": "#377eb8",
        "wheat_forage": "#4daf4a",
        "other": "#984ea3",
    }
    for system, color in colors.items():
        mask = history_df["system"] == system
        if not mask.any():
            continue
        ax.scatter(
            history_df.loc[mask, "x"],
            history_df.loc[mask, "y"],
            s=12,
            c=color,
            label=system,
            edgecolors="white",
            linewidths=0.3,
        )
    ax.set_title(f"Sample points on CDL {REF_YEAR} (EPSG:5070)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "sample_points_map.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {PROC_DIR / 'crop_history.csv'} ({len(history_df)} rows)")
    print(f"Wrote {PROC_DIR / 'wheat_years.csv'} ({len(wheat_df)} rows)")
    print(f"Wrote {PROC_DIR / 'sample_points.csv'}")
    print(f"Wrote {REPORT_DIR / 'sample_points_map.png'}")


if __name__ == "__main__":
    main()