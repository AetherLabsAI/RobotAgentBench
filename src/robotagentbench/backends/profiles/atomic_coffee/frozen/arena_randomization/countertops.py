"""Approved generated-counter height policy, applied before scene construction.

The vendored L1 catalog stays byte-identical. This changes effective generated
support dimensions in an independent copy, not original kitchen/robot assets.
"""
from __future__ import annotations

import copy
import math

POLICY = "house.generated_countertops.0p92.v1"
TOP_HEIGHT_M = 0.92
COUNTER_MODULES = frozenset(("microwave_cluster", "kitchen_counter_cluster",
                             "coffee_machine_cluster", "air_fryer_cluster"))


def configure(modules):
    """Set all four generated kitchen counters to 0.92 m, keeping floor and XY.

    Materialized appliances with explicit support-relative placement follow
    the height delta. Existing source-root appliances/robots are not moved.
    """
    effective = copy.deepcopy(modules)
    rows = []
    for name, module in effective.items():
        # Bag/basket support plinths are not countertops; keep their own height.
        if name not in COUNTER_MODULES:
            continue
        variants = (("support_materialization", "size_xyz_m"),
                    ("asset_materialization", "support_size_xyz_m"))
        for key, size_key in variants:
            spec = module.get(key)
            if spec is None or size_key not in spec:
                continue
            size = list(spec[size_key])
            if len(size) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                     or not math.isfinite(v) or v <= 0 for v in size):
                raise ValueError("Invalid generated counter dimensions: " + name)
            before = list(size)
            size[2] = TOP_HEIGHT_M
            spec[size_key] = size
            row = dict(module=name, path=spec["support_prim_path"],
                       source_size_xyz_m=before, effective_size_xyz_m=list(size),
                       floor_z_m=0.0, top_z_m=TOP_HEIGHT_M)
            if key == "asset_materialization":
                xyz = list(spec["asset_local_translate_m"])
                if len(xyz) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                       or not math.isfinite(v) for v in xyz):
                    raise ValueError("Invalid materialized appliance placement: " + name)
                original = list(xyz)
                xyz[2] += TOP_HEIGHT_M - before[2]
                spec["asset_local_translate_m"] = xyz
                row["appliance_placement"] = dict(path=spec["placement_prim_path"],
                    source_local_xyz_m=original, effective_local_xyz_m=list(xyz))
            rows.append(row)
    if {row['module'] for row in rows} != COUNTER_MODULES or len(rows) != len(COUNTER_MODULES):
        raise ValueError("Expected exactly the four declared kitchen countertops")
    return effective, dict(policy=POLICY, target_top_height_m=TOP_HEIGHT_M,
                           original_catalog_modified=False, supports=rows)


def validate(stage, plan):
    """Measure saved geometry/collision without changing the stage or physics."""
    from pxr import Gf, UsdGeom, UsdPhysics

    if plan["policy"] != POLICY:
        raise ValueError("Unknown generated-counter height policy")
    cache = UsdGeom.XformCache()
    rows = []
    for spec in plan["supports"]:
        prim = stage.GetPrimAtPath(spec["path"])
        if not prim or not prim.IsA(UsdGeom.Cube):
            raise ValueError("Expected generated counter cube: " + spec["path"])
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute("physics:collisionEnabled").Get() is False:
            raise ValueError("Generated counter has no enabled collider: " + spec["path"])
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            raise ValueError("Generated counter is invisible: " + spec["path"])
        tf = cache.GetLocalToWorldTransform(prim)
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        half = size / 2
        corners = [tf.Transform(Gf.Vec3d(x,y,z))
                   for x in (-half,half) for y in (-half,half) for z in (-half,half)]
        bottom, top = min(p[2] for p in corners), max(p[2] for p in corners)
        # Axis lengths, not world AABB dimensions: layouts may rotate the table.
        dimensions = [size * math.sqrt(sum(tf[i][j]**2 for j in range(3))) for i in range(3)]
        if (abs(bottom)>1e-6 or abs(top-TOP_HEIGHT_M)>1e-6 or
                any(abs(a-b)>1e-6 for a,b in zip(dimensions,spec["effective_size_xyz_m"]))):
            raise ValueError("Generated counter height/size/floor mismatch: " + spec["path"])
        rows.append(dict(path=spec["path"], bottom_z_m=bottom, top_z_m=top,
                         measured_size_xyz_m=dimensions, visible_and_collision_same_cube=True))
    return dict(policy=POLICY, passed=True, physics_validated=False, supports=rows)
