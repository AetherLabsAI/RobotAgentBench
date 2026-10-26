"""Required-category shelf inventory plan; no poses, USD or physics side effects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random


POOLS = {
    "tier3": ("coke_c3", "cayman_jacked", "good2grow", "gatorade_t",
              "gatorade_m", "grapefruit_can", "starbucks", "sprite_c1"),
    "tier2": ("chomps_r", "pring_bbq", "oreo_gf", "honeymaid", "zbar_t", "skit_littles"),
}
LABELS = dict(zip(POOLS["tier3"] + POOLS["tier2"], (
    "可口可乐", "Cayman Jacked", "Good2Grow", "蓝色佳得乐", "橙色佳得乐",
    "西柚气泡水", "星巴克", "雪碧", "Chomps 牛肉棒", "烧烤味品客",
    "奥利奥", "Honey Maid 饼干", "Zbar 能量棒", "彩虹糖",
)))
POLICY = "shelf_required_categories_v1"
SOURCE_DIR = "/mnt/agent-bench-data/lingjun/ziche/work/shelf_training_visible360_20260916"
SOURCE_HASHES = {
    "snack_diversity.py": "93b9e67ad02fca411e17b876b363dedc09740afbbd28fb52edfb221d69060533",
    "random_shelf_scene.py": "688f4bcc17a2b6ff9467349ccffdcf3fc47b683bf0280171e797f82d66526a58",
}


def choose_inventory(rng, pool, required, count):
    """Reserve required kinds, finish the first variety round, then repeat."""
    chosen = [name for name in pool if name in required]
    remainder = [name for name in pool if name not in required]
    rng.shuffle(remainder)
    chosen.extend(remainder[:count - len(chosen)])
    while len(chosen) < count:
        cycle = list(pool)
        rng.shuffle(cycle)
        chosen.extend(cycle[:count - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def compose(request):
    if not isinstance(request, dict):
        raise ValueError("Request must be an object")
    if set(request) - {"seed", "required_categories"}:
        raise ValueError("Unknown request fields")
    seed = request.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    requested = request.get("required_categories", [])
    if not isinstance(requested, list) or any(not isinstance(n, str) for n in requested):
        raise ValueError("required_categories must be a list of product-family IDs")
    unknown = set(requested) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    required = set(requested)
    layers = []
    for support_id, pool in POOLS.items():
        rng = random.Random(f"{POLICY}:{seed}:{support_id}")
        needed = required.intersection(pool)
        drawn_count = rng.randint(4, 10)
        count = max(drawn_count, len(needed))
        chosen = choose_inventory(rng, pool, needed, count)
        copies = Counter()
        objects = []
        for name in chosen:
            copies[name] += 1
            objects.append({
                "instance_id": f"shelf/{support_id}/{name}/{copies[name]:02d}",
                "product_family": name, "label": LABELS[name],
                "support_id": support_id,
                "asset_relative_to_all_assets": f"{name}/asset.usda",
                "required_instance": name in needed and copies[name] == 1,
            })
        layers.append({
            "support_id": support_id, "drawn_count": drawn_count, "count": count,
            "count_raised_for_required_categories": count != drawn_count,
            "required_categories": sorted(needed),
            "minimum_footprint_gap_m": .085 if count <= 5 else .055 if count <= 9 else .035,
            "objects": objects,
        })
    return {
        "policy": POLICY, "status": "inventory_plan_ready", "seed": seed,
        "required_categories": sorted(required), "layers": layers,
        "source_reference": {"directory_on_202": SOURCE_DIR, "sha256": SOURCE_HASHES},
        "required_category_coverage_passed": all(
            name in {o["product_family"] for layer in layers for o in layer["objects"]}
            for name in required),
        "poses_sampled": False, "usd_written": False, "runtime_ready": False,
        "placement_contract": {
            "full_footprint_on_support": True,
            "missing_required_action": "reject_scene_and_resample_never_silently_drop",
            "inventory_shortfall_action": "reject_scene_and_resample",
            "geometry_collision_visibility_and_reachability": "not_checked",
            "physics_wake_action": "reuse_existing_state_do_not_resample",
        },
    }


def verify_placed_inventory(plan, placed):
    """Fail-closed inventory-only gate for a future placement adapter.

    Rows must carry instance_id, product_family and support_id. Passing this
    does NOT validate geometry, visibility, reachability or native physics.
    """
    expected = {o["instance_id"]: o for layer in plan["layers"] for o in layer["objects"]}
    actual = {}
    for row in placed:
        identity = row["instance_id"]
        if identity in actual:
            raise ValueError(f"Duplicate instance: {identity}")
        actual[identity] = row
    missing = set(plan["required_categories"]) - {o["product_family"] for o in placed}
    if missing:
        raise ValueError(f"Required categories missing: {sorted(missing)}")
    if set(actual) != set(expected):
        raise ValueError("Placed inventory differs from plan; reject or explicitly replan")
    for identity, row in actual.items():
        if any(row[key] != expected[identity][key] for key in ("product_family", "support_id")):
            raise ValueError(f"Wrong product or tier: {identity}")
    return {"inventory_matches_plan": True, "required_categories_present": True,
            "geometry_or_physics_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        result = compose(json.loads(args.request.read_text()))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
