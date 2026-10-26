#!/usr/bin/env python3
"""Generate deterministic, fully physical, cup-free kitchen tableware.

The generated USD layer never disables rigid bodies or collisions.  Objects are
authored a few millimetres above their intended supports so Isaac can settle
them through real contact.  A sidecar records both the sampled layout and the
invariants needed to reproduce and audit it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


POLICY_VERSION = "real2sim.kitchen_tableware_randomization.v2"
DEFAULT_BASE_LAYER = "../scene_room.usda"
ASSET_ROOT = "./assets/tableware"
TABLE_Z = 0.920
SUPPORTS = {
    "central_table": {
        # Canonical room_shell_v2 central table (CounterTop_17).
        "top_z_m": 0.920,
        "safe_region": {
            "x_min": 2.45,
            "x_max": 2.95,
            "y_min": -1.95,
            "y_max": -0.85,
        },
        "single_anchor": (2.62, -1.40),
        "double_anchors": ((2.62, -1.73), (2.62, -1.05)),
        "max_clusters": 2,
    },
    "wall_counter": {
        # BP_Kitchen7 is a complete wall-counter run, separated from both
        # RightRange600 (the actual stove) and BP_Kitchen10 (the sink).
        # Its measured top spans x=[4.0788,4.9788], y=[1.4592,2.3592].
        "top_z_m": 0.860,
        "safe_region": {
            "x_min": 4.23,
            "x_max": 4.83,
            "y_min": 1.62,
            "y_max": 2.20,
        },
        "single_anchor": (4.53, 1.91),
        "double_anchors": (),
        "max_clusters": 1,
    },
}
SEED_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


@dataclass(frozen=True)
class Asset:
    key: str
    path: str
    diameter: float
    height: float
    length: float = 0.0


@dataclass
class Item:
    name: str
    asset: str
    translate: tuple[float, float, float]
    rotate_xyz: tuple[float, float, float]
    role: str


PLATES = {
    "plate_001": Asset("plate_001", "plate_001/exports/plate_001.usda", 0.260, 0.024),
    "plate_002": Asset("plate_002", "plate_002/exports/plate_002.usda", 0.245, 0.020),
    "plate_003": Asset("plate_003", "plate_003/exports/plate_003.usda", 0.225, 0.022),
    "plate_004": Asset("plate_004", "plate_004/exports/plate_004.usda", 0.240, 0.038),
    "plate_005": Asset("plate_005", "plate_005/exports/plate_005.usda", 0.205, 0.019),
    "plate_006": Asset("plate_006", "plate_006/exports/plate_006.usda", 0.255, 0.026),
}
BOWLS = {
    "bowl_001": Asset("bowl_001", "bowl_001/exports/bowl_001.usda", 0.190, 0.088),
    "bowl_002": Asset("bowl_002", "bowl_002/exports/bowl_002.usda", 0.190, 0.095),
    "bowl_003": Asset("bowl_003", "bowl_003/exports/bowl_003.usda", 0.180, 0.067),
    "bowl_004": Asset("bowl_004", "bowl_004/exports/bowl_004.usda", 0.210, 0.055),
    "bowl_005": Asset("bowl_005", "bowl_005/exports/bowl_005.usda", 0.175, 0.088),
    "bowl_006": Asset("bowl_006", "bowl_006/exports/bowl_006.usda", 0.185, 0.074),
}
CUTLERY = {
    "chopstick_001": Asset(
        "chopstick_001",
        "cutlery_set_v1/chopstick_001/exports/chopstick_001.usda",
        0.007,
        0.007,
        0.240,
    ),
}
ASSETS = {**PLATES, **BOWLS, **CUTLERY}


def seeded_rng(seed: str) -> tuple[random.Random, int]:
    if not SEED_PATTERN.fullmatch(seed):
        raise ValueError("seed must match [A-Za-z0-9_.:-]{1,128}")
    seed_u64 = random.Random(f"{POLICY_VERSION}\0{seed}").getrandbits(64)
    return random.Random(seed_u64), seed_u64


def jitter(rng: random.Random, amount: float) -> float:
    return rng.uniform(-amount, amount)


def add(
    items: list[Item],
    asset: Asset,
    role: str,
    xyz: tuple[float, float, float],
    rxyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    index = 1 + sum(item.asset == asset.key for item in items)
    items.append(
        Item(
            name=f"{asset.key}_{index:02d}",
            asset=asset.key,
            translate=tuple(round(value, 6) for value in xyz),
            rotate_xyz=tuple(round(value, 5) for value in rxyz),
            role=role,
        )
    )


def compatible_plate_bowl(rng: random.Random) -> tuple[Asset, Asset]:
    candidates = [
        (plate, bowl)
        for plate in PLATES.values()
        for bowl in BOWLS.values()
        if bowl.diameter <= plate.diameter - 0.030
    ]
    return rng.choice(candidates)


def flat_cutlery(
    items: list[Item],
    asset: Asset,
    role: str,
    x: float,
    y: float,
    yaw: float,
) -> None:
    # Source cutlery is long along local Y.  +90deg aligns it with world X.
    add(items, asset, role, (x, y, TABLE_Z + 0.008), (0.0, 0.0, 90.0 + yaw))


def chopstick_pair_flat(
    items: list[Item], x: float, y: float, yaw: float, role: str
) -> None:
    spacing = 0.013
    flat_cutlery(items, CUTLERY["chopstick_001"], role, x, y - spacing / 2, yaw)
    flat_cutlery(items, CUTLERY["chopstick_001"], role, x, y + spacing / 2, yaw + 1.0)


def template_a(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items: list[Item] = []
    plate = rng.choice(list(PLATES.values()))
    add(items, plate, "place_setting_plate", (x, y, TABLE_Z + 0.005), (0, 0, rng.uniform(0, 360)))
    chopstick_pair_flat(items, x, y - 0.175, jitter(rng, 3), "chopsticks_right")
    return items


def template_b(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items: list[Item] = []
    plate, bowl = compatible_plate_bowl(rng)
    add(items, plate, "base_plate", (x, y, TABLE_Z + 0.005), (0, 0, rng.uniform(0, 360)))
    add(items, bowl, "bowl_on_plate", (x, y, TABLE_Z + plate.height + 0.011), (0, 0, rng.uniform(0, 360)))
    chopstick_pair_flat(items, x, y - 0.185, jitter(rng, 3), "chopsticks_right")
    return items


def template_c(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items: list[Item] = []
    bowl = rng.choice(list(BOWLS.values()))
    add(items, bowl, "soup_bowl", (x, y, TABLE_Z + 0.005), (0, 0, rng.uniform(0, 360)))
    chopstick_pair_flat(items, x, y - 0.175, jitter(rng, 3), "chopsticks_right")
    return items


def template_d(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items: list[Item] = []
    plate = PLATES["plate_005"] if rng.random() < 0.7 else PLATES["plate_003"]
    add(items, plate, "breakfast_plate", (x, y, TABLE_Z + 0.005), (0, 0, rng.uniform(0, 360)))
    chopstick_pair_flat(items, x, y - 0.165, jitter(rng, 3), "chopsticks_right")
    return items


def plate_stack(
    anchor: tuple[float, float], rng: random.Random, count: int
) -> tuple[list[Item], float]:
    x, y = anchor
    items: list[Item] = []
    if rng.random() < 0.7:
        chosen = [rng.choice(list(PLATES.values()))] * count
    else:
        chosen = sorted(rng.sample(list(PLATES.values()), count), key=lambda a: -a.diameter)
    z = TABLE_Z + 0.005
    for level, plate in enumerate(chosen):
        add(
            items,
            plate,
            f"plate_stack_level_{level}",
            (x + jitter(rng, 0.006), y + jitter(rng, 0.006), z),
            (0, 0, rng.uniform(0, 360)),
        )
        z += plate.height + 0.006
    return items, z


def bowl_stack(
    anchor: tuple[float, float], rng: random.Random, count: int
) -> tuple[list[Item], float]:
    x, y = anchor
    items: list[Item] = []
    if rng.random() < 0.7:
        chosen = [rng.choice(list(BOWLS.values()))] * count
    else:
        chosen = sorted(rng.sample(list(BOWLS.values()), count), key=lambda a: -a.diameter)
    z = TABLE_Z + 0.005
    for level, bowl in enumerate(chosen):
        add(
            items,
            bowl,
            f"bowl_stack_level_{level}",
            (x + jitter(rng, 0.004), y + jitter(rng, 0.004), z),
            (0, 0, rng.uniform(0, 360)),
        )
        z += bowl.height + 0.008
    return items, z


def template_e(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items = plate_stack(anchor, rng, rng.randint(2, 4))[0]
    chopstick_pair_flat(items, x, y - 0.175, jitter(rng, 4), "stack_chopsticks")
    return items


def template_f(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items = bowl_stack(anchor, rng, rng.randint(2, 3))[0]
    chopstick_pair_flat(items, x, y - 0.175, jitter(rng, 4), "stack_chopsticks")
    return items


def template_g(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    plate_count = rng.randint(1, 2)
    items, z = plate_stack(anchor, rng, plate_count)
    bottom_plate = max((ASSETS[item.asset] for item in items), key=lambda a: a.diameter)
    bowls = [b for b in BOWLS.values() if b.diameter <= bottom_plate.diameter - 0.030]
    bowl_count = rng.randint(1, 2)
    chosen = sorted((rng.choice(bowls) for _ in range(bowl_count)), key=lambda a: -a.diameter)
    for level, bowl in enumerate(chosen):
        add(items, bowl, f"mixed_bowl_level_{level}", (x, y, z), (0, 0, rng.uniform(0, 360)))
        z += bowl.height + 0.008
    chopstick_pair_flat(items, x, y - 0.18, jitter(rng, 4), "mixed_chopsticks")
    return items


def template_h(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items, _ = plate_stack((x, y + 0.055), rng, rng.randint(2, 3))
    chopstick_pair_flat(items, x - 0.02, y - 0.165, jitter(rng, 4), "spare_chopsticks")
    return items


def template_i(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    x, y = anchor
    items: list[Item] = []
    chosen = sorted(rng.sample(list(PLATES.values()), rng.randint(2, 3)), key=lambda a: -a.diameter)
    z = TABLE_Z + 0.005
    for level, plate in enumerate(chosen):
        add(
            items,
            plate,
            f"casual_plate_level_{level}",
            (x + jitter(rng, 0.012), y + jitter(rng, 0.012), z),
            (0, 0, rng.uniform(0, 360)),
        )
        z += plate.height + 0.006
    bowls = [b for b in BOWLS.values() if b.diameter <= chosen[0].diameter - 0.030]
    bowl = rng.choice(bowls)
    add(items, bowl, "casual_top_bowl", (x + jitter(rng, 0.008), y + jitter(rng, 0.008), z), (0, 0, rng.uniform(0, 360)))
    chopstick_pair_flat(items, x, y - 0.18, jitter(rng, 8), "casual_chopsticks")
    return items


def template_k(anchor: tuple[float, float], rng: random.Random) -> list[Item]:
    """A physical chopstick pair released just above a bowl rim."""
    x, y = anchor
    items: list[Item] = []
    bowl = rng.choice(list(BOWLS.values()))
    bowl_yaw = rng.uniform(0, 360)
    add(items, bowl, "bowl_support", (x, y, TABLE_Z + 0.005), (0, 0, bowl_yaw))
    pair_yaw = rng.uniform(0, 360)
    theta = math.radians(pair_yaw)
    # Offset the pair perpendicular to its long axis while keeping both roots
    # close to the bowl centre so each chopstick can contact both rim sides.
    # The source chopstick's long axis is local Y; at yaw=0 its world-width
    # axis is +X. Offset the pair across that width axis, not along its length.
    nx, ny = math.cos(theta), math.sin(theta)
    spacing = 0.013
    z = TABLE_Z + 0.005 + bowl.height + 0.008
    chop = CUTLERY["chopstick_001"]
    for index, sign in enumerate((-0.5, 0.5)):
        along = jitter(rng, 0.006)
        dx = sign * spacing * nx + along * math.cos(theta)
        dy = sign * spacing * ny + along * math.sin(theta)
        add(
            items,
            chop,
            f"rim_chopstick_{index + 1}",
            (x + dx, y + dy, z + index * 0.001),
            (0, 0, pair_yaw),
        )
    return items


TEMPLATES: dict[str, Callable[[tuple[float, float], random.Random], list[Item]]] = {
    "A": template_a,
    "B": template_b,
    "C": template_c,
    "D": template_d,
    "E": template_e,
    "F": template_f,
    "G": template_g,
    "H": template_h,
    "I": template_i,
    "K": template_k,
}
TEMPLATE_WEIGHTS = {
    "A": 0.16,
    "B": 0.16,
    "C": 0.10,
    "D": 0.08,
    "E": 0.10,
    "F": 0.10,
    "G": 0.08,
    "H": 0.07,
    "I": 0.05,
    "K": 0.10,
}


def choose_templates(rng: random.Random, count: int) -> list[str]:
    keys = list(TEMPLATE_WEIGHTS)
    weights = [TEMPLATE_WEIGHTS[key] for key in keys]
    chosen: list[str] = []
    while len(chosen) < count:
        candidate = rng.choices(keys, weights=weights, k=1)[0]
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen


def build_plan(seed: str, base_layer: str, requested: list[str] | None) -> dict[str, object]:
    rng, seed_u64 = seeded_rng(seed)
    support_id = rng.choice(sorted(SUPPORTS))
    support = SUPPORTS[support_id]
    count = 2 if support["max_clusters"] == 2 and rng.random() < 0.65 else 1
    templates = requested or choose_templates(rng, count)
    if not templates or len(templates) > int(support["max_clusters"]):
        raise ValueError(f"{support_id} supports at most {support['max_clusters']} clusters")
    unknown = sorted(set(templates) - set(TEMPLATES))
    if unknown:
        raise ValueError(f"unknown templates: {', '.join(unknown)}")

    anchors = (
        list(support["double_anchors"])
        if len(templates) == 2
        else [support["single_anchor"]]
    )
    z_shift = float(support["top_z_m"]) - TABLE_Z
    clusters = []
    for index, (template, nominal_anchor) in enumerate(zip(templates, anchors), 1):
        anchor = (
            nominal_anchor[0] + jitter(rng, 0.025),
            nominal_anchor[1] + jitter(rng, 0.025),
        )
        items = TEMPLATES[template](anchor, rng)
        for item in items:
            item.translate = (
                item.translate[0], item.translate[1],
                round(item.translate[2] + z_shift, 6),
            )
        clusters.append(
            {
                "index": index,
                "template": template,
                "anchor": [round(anchor[0], 6), round(anchor[1], 6), support["top_z_m"]],
                "items": [asdict(item) for item in items],
            }
        )

    return {
        "schema": "real2sim.kitchen_tableware_plan.v2",
        "policy_version": POLICY_VERSION,
        "seed": seed,
        "seed_u64": seed_u64,
        "base_layer": base_layer,
        "interference_level": "safe",
        "support_id": support_id,
        "support_prim_path": (
            "/World/Background/Shell/CounterTop_17"
            if support_id == "central_table"
            else "/World/Background/Shell/ConstructorKitchen/MetricScale/"
                 "BP_Kitchen7/Body/Worktop"
        ),
        "table_top_z_m": support["top_z_m"],
        "safe_region": support["safe_region"],
        "clusters": clusters,
        "invariants": {
            "rigid_bodies_disabled": False,
            "collisions_disabled": False,
            "kinematic_authored": False,
            "attachments_authored": False,
            "fixed_joints_authored": False,
            "physics_settle_required": True,
            "read_only_baseline_modified": False,
            "working_scene_adds_dynamic_props": True,
            "cups_allowed": False,
            "cup_asset_count": 0,
        },
    }


def render_usda(plan: dict[str, object]) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        "    subLayers = [",
        f"        @./{plan['base_layer']}@",
        "    ]",
        '    upAxis = "Z"',
        ")",
        "",
        f"# policy_version: {plan['policy_version']}",
        f"# seed: {plan['seed']}",
        "# All referenced props retain their authored dynamic rigid-body physics.",
        'over "World"',
        "{",
        '    def Xform "KitchenRandom"',
        "    {",
        '        def Xform "Tableware"',
        "        {",
    ]
    for cluster in plan["clusters"]:
        lines.extend(
            [
                f'            def Xform "Cluster_{cluster["index"]:02d}_{cluster["template"]}"',
                "            {",
            ]
        )
        for item in cluster["items"]:
            asset = ASSETS[item["asset"]]
            tx, ty, tz = item["translate"]
            rx, ry, rz = item["rotate_xyz"]
            lines.extend(
                [
                    f'                def Xform "{item["name"]}" (',
                    f"                    prepend references = @{ASSET_ROOT}/{asset.path}@",
                    "                    customData = {",
                    f'                        string randomizationRole = "{item["role"]}"',
                    f'                        string sourceAsset = "{asset.key}"',
                    "                    }",
                    "                )",
                    "                {",
                    f"                    float3 xformOp:rotateXYZ = ({rx:.5f}, {ry:.5f}, {rz:.5f})",
                    f"                    double3 xformOp:translate = ({tx:.6f}, {ty:.6f}, {tz:.6f})",
                    '                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]',
                    "                }",
                    "",
                ]
            )
        lines.extend(["            }", ""])
    lines.extend(["        }", "    }", "}", ""])
    return "\n".join(lines)


def validate_plan(plan: dict[str, object]) -> None:
    if plan.get("schema") != "real2sim.kitchen_tableware_plan.v2":
        raise ValueError("not a cup-free kitchen tableware v2 plan")
    safe = plan["safe_region"]
    top_z = float(plan["table_top_z_m"])
    for cluster in plan["clusters"]:
        names: set[str] = set()
        ax, ay, _ = cluster["anchor"]
        if not (safe["x_min"] <= ax <= safe["x_max"]):
            raise ValueError(f"cluster x outside safe region: {ax}")
        if not (safe["y_min"] <= ay <= safe["y_max"]):
            raise ValueError(f"cluster y outside safe region: {ay}")
        for item in cluster["items"]:
            if item["name"] in names:
                raise ValueError(f"duplicate prim name: {item['name']}")
            names.add(item["name"])
            if item["asset"] not in ASSETS:
                raise ValueError(f"unknown asset: {item['asset']}")
            if "cup" in str(item["asset"]).lower() or "cup" in str(item["role"]).lower():
                raise ValueError(f"cup asset/role is forbidden: {item}")
            x, y, z = item["translate"]
            if not (
                safe["x_min"] < x < safe["x_max"]
                and safe["y_min"] < y < safe["y_max"]
                and z >= top_z
            ):
                raise ValueError(f"item origin outside support volume: {item}")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_collision_free_plan(
    seed: str,
    base_layer: str,
    requested: list[str] | None = None,
    *,
    max_attempts: int = 64,
) -> tuple[dict[str, object], dict[str, object]]:
    """Rejection-sample complete footprints using the accepted 0806 audit."""
    try:
        from kitchen_randomization import validate_tableware_plan
    except ImportError:
        # Supports package imports during tests without changing production.
        from real2sim.production.kitchen_randomization import (
            validate_tableware_plan,
        )

    failures: list[str] = []
    for attempt in range(max_attempts):
        candidate_seed = f"{seed}.candidate-{attempt}"
        plan = build_plan(candidate_seed, base_layer, requested)
        validate_plan(plan)
        try:
            audit = validate_tableware_plan(plan, ASSETS)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        audit.update({
            "requested_seed": seed,
            "selected_candidate_seed": candidate_seed,
            "selection_attempt": attempt,
            "rejected_candidate_count": len(failures),
            "rejection_reasons": failures,
            "cups_present": False,
        })
        return plan, audit
    raise RuntimeError(
        f"no collision-free cup-free tableware plan after {max_attempts} "
        f"candidates; last failures: {failures[-3:]}"
    )


def _validate_working_scene(scene: Path, *, allow_test_scene: bool) -> None:
    if "baseline" in str(scene).lower():
        raise ValueError("refusing to modify a baseline scene")
    if not scene.is_file():
        raise FileNotFoundError(f"scene does not exist: {scene}")
    if allow_test_scene:
        return
    if scene.name != "scene_room.usda":
        raise ValueError(
            "production tableware randomization requires scene_room.usda"
        )
    sidecar = scene.with_suffix(scene.suffix + ".session.json")
    if not sidecar.is_file():
        raise ValueError("scene_room.usda has no open session sidecar")
    session = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        session.get("schema") != "real2sim.scene_session.v1"
        or session.get("status") != "open"
        or Path(str(session.get("scene", ""))).resolve() != scene.resolve()
    ):
        raise ValueError("scene_room.usda session sidecar is not a valid owner")


def _matrix_values(matrix: object) -> tuple[float, ...]:
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _asset_physics_audit(asset_root: Path) -> dict[str, object]:
    from pxr import Usd, UsdPhysics

    rows = []
    for key, asset in sorted(ASSETS.items()):
        path = (asset_root / asset.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing tableware asset: {path}")
        stage = Usd.Stage.Open(str(path))
        if not stage:
            raise ValueError(f"could not open tableware asset: {path}")
        rigid = []
        collision = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision.append(str(prim.GetPath()))
        if not rigid or not collision:
            raise ValueError(
                f"tableware asset lacks rigid-body/collision APIs: {key}"
            )
        rows.append({
            "asset": key,
            "path": str(path),
            "rigid_body_prims": rigid,
            "collision_prims": collision,
        })
    return {
        "schema": "real2sim.kitchen_tableware_asset_physics_audit.v1",
        "all_assets_dynamic_and_collidable": True,
        "cup_asset_count": 0,
        "assets": rows,
    }


def apply_plan_to_stage(
    *,
    scene_path: Path | str,
    plan: dict[str, object],
    collision_audit: dict[str, object],
    asset_root: Path | str,
    allow_test_scene: bool = False,
) -> dict[str, object]:
    """Add one physical, cup-free tableware root to the working scene."""
    from pxr import Usd, UsdGeom

    scene = Path(scene_path).resolve()
    assets = Path(asset_root).resolve()
    _validate_working_scene(scene, allow_test_scene=allow_test_scene)
    validate_plan(plan)
    if collision_audit.get("collision_free") is not True:
        raise ValueError("tableware plan is not collision-free")
    if collision_audit.get("cups_present") is not False:
        raise ValueError("tableware collision audit does not forbid cups")
    physics_audit = _asset_physics_audit(assets)
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"could not open scene: {scene}")

    root_path = "/World/EpisodeKitchenTableware"
    if stage.GetPrimAtPath(root_path).IsValid():
        raise ValueError("episode tableware already exists; refusing a second pass")
    support_paths = [
        "/World/Background/Shell/CounterTop_17",
        "/World/Background/Shell/ConstructorKitchen/MetricScale/"
        "BP_Kitchen7/Body/Worktop",
    ]
    before = {}
    for path in support_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"tableware support is missing: {path}")
        before[path] = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
        )

    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetCustomDataByKey("cupFree", True)
    root.GetPrim().SetCustomDataByKey("supportId", str(plan["support_id"]))
    authored = [root_path]
    for cluster in plan["clusters"]:
        cluster_path = f"{root_path}/Cluster_{int(cluster['index']):02d}_{cluster['template']}"
        UsdGeom.Xform.Define(stage, cluster_path)
        for item in cluster["items"]:
            prim_path = f"{cluster_path}/{item['name']}"
            xform = UsdGeom.Xform.Define(stage, prim_path)
            asset = ASSETS[str(item["asset"])]
            source = (assets / asset.path).resolve()
            reference = os.path.relpath(source, start=scene.parent)
            if not xform.GetPrim().GetReferences().AddReference(reference):
                raise ValueError(f"failed to reference {source}")
            tx, ty, tz = item["translate"]
            rx, ry, rz = item["rotate_xyz"]
            xform.AddTranslateOp().Set((float(tx), float(ty), float(tz)))
            xform.AddRotateXYZOp().Set((float(rx), float(ry), float(rz)))
            xform.GetPrim().SetCustomDataByKey(
                "randomizationRole", str(item["role"])
            )
            xform.GetPrim().SetCustomDataByKey(
                "sourceAsset", str(item["asset"])
            )
            authored.append(prim_path)

    for path, expected in before.items():
        observed = _matrix_values(
            UsdGeom.Xformable(stage.GetPrimAtPath(path))
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        if observed != expected:
            raise ValueError(f"tableware randomization moved support {path}")
    stage.GetRootLayer().Save()
    manifest = {
        **plan,
        "schema": "real2sim.kitchen_tableware.applied.v2",
        "scene_path": str(scene),
        "tableware_root_prim": root_path,
        "authored_prim_paths": sorted(authored),
        "collision_audit": collision_audit,
        "asset_physics_audit": physics_audit,
        "support_transforms_unchanged": True,
        "cups_present": False,
        "physics_effect": "dynamic_rigid_bodies_with_collision",
    }
    sidecar = scene.with_suffix(scene.suffix + ".kitchen_tableware.json")
    atomic_write(sidecar, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["sidecar_path"] = str(sidecar)
    return manifest


def _mesh311_python() -> Path:
    teleop = Path(__file__).resolve().parents[1] / "teleop"
    if str(teleop) not in sys.path:
        sys.path.insert(0, str(teleop))
    from machine_paths import MESH311_PYTHON

    return Path(MESH311_PYTHON)


def run_tableware_subprocess(
    *,
    scene_path: Path | str,
    task_id: str,
    episode_uid: str = "",
    seed_override: str | int | None = None,
    asset_root: Path | str,
    timeout: int = 180,
) -> dict[str, object]:
    seed = str(seed_override).strip() if seed_override is not None else ""
    if not seed:
        seed = str(episode_uid).strip() or f"manual-{task_id}"
    command = [
        str(_mesh311_python()), str(Path(__file__).resolve()),
        "--scene", str(scene_path),
        "--asset-root", str(asset_root),
        "--seed", seed,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "kitchen tableware randomizer returned invalid JSON: "
            + (result.stdout + result.stderr)[-1000:]
        ) from exc
    if result.returncode != 0 or payload.get("ok") is not True:
        raise RuntimeError(
            "kitchen tableware randomization failed: "
            + str(payload.get("reason", payload))
        )
    return payload["tableware"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--base-layer", default=DEFAULT_BASE_LAYER)
    parser.add_argument(
        "--templates",
        help="comma-separated subset of A-I or K; safe v1 accepts at most two",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-test-scene", action="store_true")
    args = parser.parse_args()

    requested = None
    if args.templates:
        requested = [value.strip().upper() for value in args.templates.split(",") if value.strip()]
    plan, collision_audit = build_collision_free_plan(
        args.seed, args.base_layer, requested
    )
    if args.scene:
        if args.output:
            raise ValueError("--scene and --output are mutually exclusive")
        asset_root = args.asset_root or (
            args.scene.resolve().parent / "kitchen_random/assets/tableware"
        )
        result = plan if args.dry_run else apply_plan_to_stage(
            scene_path=args.scene,
            plan=plan,
            collision_audit=collision_audit,
            asset_root=asset_root,
            allow_test_scene=args.allow_test_scene,
        )
        print(json.dumps({"ok": True, "tableware": result}, sort_keys=True))
        return
    output = args.output
    if output is None:
        output = Path(__file__).resolve().parent / (
            f"scene_kitchen_random_v1_tableware_{args.seed}.usda"
        )
    output = output.resolve()
    atomic_write(output, render_usda(plan))
    sidecar = output.with_suffix(output.suffix + ".tableware.json")
    atomic_write(sidecar, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(output)
    print(sidecar)


if __name__ == "__main__":
    main()
