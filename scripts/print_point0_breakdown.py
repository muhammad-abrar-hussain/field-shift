"""Print point 0 compare score breakdown (normal + dry)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compare import compare, compare_score_breakdown, feature_point  # noqa: E402


def _print_breakdown(yt: str, base: dict) -> None:
    req = {**base, "year_type": yt}
    bd = compare_score_breakdown(req)
    out = compare(req)
    print(f"\n{'=' * 60}")
    print(f"year_type={yt}  best_match={out['best_match']}")
    print(bd["confidence_note"])
    for rot in out["rotations"]:
        print(f"  {rot['id']}: confidence={rot['confidence']} total={rot['scores']['total']} p50={rot['crop_health']['p50']:.4f}")
    print(f"\nScoring breakdown (priorities={bd['priorities']}, pick weights={bd['pick_weights']})")
    for row in bd["rotations"]:
        print(f"\n--- Rotation {row['rotation']} bare_days/yr={row['bare_days_per_year']} ---")
        print(f"  raw metrics: {row['raw']}")
        print(f"  priority scores (0-10): {row['priority_scores']}")
        print(f"  effective weights (renormalized): {row['weights_effective']}")
        print(
            f"  priority_score={row['priority_score']:.2f} health={row['health_score']:.2f} "
            f"total={row['total']}  (active weight sum={row['active_weight_sum']:.1f})"
        )


def main() -> None:
    pt = feature_point(0)
    base = {
        "lat": float(pt["lat"]),
        "lon": float(pt["lon"]),
        "current_rotation": "A",
        "water_source": "dryland",
        "priorities": ["water", "soil", "income"],
        "machinery": {"sorghum_planter": True, "forage_equipment": True},
        "soil_texture": "medium",
    }
    for yt in ("normal", "dry"):
        _print_breakdown(yt, base)


if __name__ == "__main__":
    main()