"""Bridge the current House USD collision meshes to the Newton cloth probes.

Task definitions still come from the Sheet catalog. This numeric sidecar is
geometry, not a second task registry or an assertion of physical equivalence.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO / "real2sim/teleop", REPO / "real2sim/production"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tablecloth_basket_contract import BASKET_COLLISION_PATH, BASKET_RISER_PATH

SCHEMA = "real2sim.tablecloth_scene_collision.v1"
ROOTS = {"table": "/World/Assets/CoffeeTable", "basket": "/World/Assets/TableclothBasket"}


def task_environment(task_id):
    from benchmark_eval_registry import resolve

    task = resolve(task_id)["task"]
    if task["runtime"]["environment"]["robot_station"]["key"] != "tablecloth":
        raise ValueError(f"{task_id} does not use the tablecloth station")
    return task["runtime"]["environment"]


def station_xyyaw(task_id):
    station = task_environment(task_id)["robot_station"]
    return np.asarray([*station["xy_m"], station["yaw_deg"]], dtype=np.float64)


def _triangles(counts, indices):
    faces, offset = [], 0
    for count in counts:
        if count not in (3, 4):
            raise ValueError(f"Unsupported USD polygon size {count}; triangulate the source asset")
        face = indices[offset:offset + count]
        faces.extend((face[0], face[j], face[j + 1]) for j in range(1, count - 1))
        offset += count
    if offset != len(indices):
        raise ValueError("Malformed USD face indices")
    return np.asarray(faces, dtype=np.int32)


def export_scene_collisions(stage):
    """Export enabled table/basket colliders in world metres, including riser."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("House collision export requires a metre stage")
    meshes, payload = [], {}
    roots = dict(ROOTS)
    if stage.GetPrimAtPath('/World/Assets/TableclothLaundryBasket'):
        roots['laundry_basket'] = '/World/Assets/TableclothLaundryBasket'
    for group, root in roots.items():
        prim = stage.GetPrimAtPath(root)
        if not prim:
            raise ValueError(f"Missing current House prim {root}")
        for child in Usd.PrimRange(prim):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is False:
                continue
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                faces = _triangles(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
                left_handed = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
            elif child.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(child).GetSizeAttr().Get()) / 2
                points = np.asarray([(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)])
                faces = _triangles([4] * 6, [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3])
                left_handed = False
            else:
                raise ValueError(f"Unsupported collider {child.GetPath()}: {child.GetTypeName()}")
            matrix = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            points = np.asarray([matrix.Transform(Gf.Vec3d(*point)) for point in points], dtype=np.float32)
            if left_handed != (matrix.GetDeterminant() < 0):
                faces = faces[:, [0, 2, 1]]
            key = str(len(meshes))
            payload[f"scene_points_{key}"] = points
            payload[f"scene_triangles_{key}"] = faces
            meshes.append({"key": key, "group": group, "prim_path": str(child.GetPath()), "vertices": len(points), "triangles": len(faces)})
    basket_paths = sorted(m["prim_path"] for m in meshes if m["group"] == "basket")
    if basket_paths != sorted([BASKET_COLLISION_PATH, BASKET_RISER_PATH]):
        raise ValueError(f"Stale basket geometry: {basket_paths}")
    payload["scene_collision_json"] = np.asarray(json.dumps({"schema": SCHEMA, "meshes": meshes}, sort_keys=True))
    return payload


def load_scene_collisions(source):
    if "scene_collision_json" not in source:
        raise ValueError("Missing current House collision export; remeasure v5 pad tracks")
    metadata = json.loads(str(source["scene_collision_json"].item()))
    if metadata["schema"] != SCHEMA:
        raise ValueError("Unsupported scene collision schema")
    meshes = []
    for entry in metadata["meshes"]:
        key = entry["key"]
        points = np.asarray(source[f"scene_points_{key}"], dtype=np.float32)
        triangles = np.asarray(source[f"scene_triangles_{key}"], dtype=np.int32)
        if (points.shape != (entry["vertices"], 3) or triangles.shape != (entry["triangles"], 3)
                or not len(points) or not len(triangles) or not np.isfinite(points).all()
                or triangles.min() < 0 or triangles.max() >= len(points)):
            raise ValueError(f"Invalid current collider {entry['prim_path']}")
        meshes.append((entry, points, triangles))
    if sorted(e["prim_path"] for e, _, _ in meshes if e["group"] == "basket") != sorted([BASKET_COLLISION_PATH, BASKET_RISER_PATH]):
        raise ValueError("Legacy basket collision sidecar")
    if not any(e["group"] == "table" for e, _, _ in meshes):
        raise ValueError("Missing table collision meshes")
    return meshes


def add_newton_collisions(builder, meshes, group):
    import newton

    start = builder.shape_count
    for entry, points, faces in meshes:
        if entry["group"] == group:
            # Keep the open basket concave. Convex-hull conversion closes its mouth.
            mesh = newton.Mesh(points, faces.reshape(-1), is_solid=False, compute_inertia=False)
            builder.add_shape_mesh(body=-1, mesh=mesh)
    return start, builder.shape_count


def verify_scene_collisions(stage, source):
    """Compare contents at render time; matching AABBs do not establish identity."""
    current = export_scene_collisions(stage)
    for key, value in current.items():
        if key not in source or not np.array_equal(value, source[key]):
            raise ValueError(f"Scene differs from measured collision geometry: {key}")


def source_record():
    return {"repository": str(REPO), "commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(), "scene": "arena_scene/v1_2/demo_sim/scene_room_baseline.usda", "robot": "arena_scene/v1_2/source/geniesim/assets/robot/G1_omnipicker/robot.usda"}


def main():
    import argparse
    from pxr import Usd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a fresh geometry output: {output}")
    stage = Usd.Stage.Open(args.scene)
    if not stage:
        raise ValueError(f"Cannot open House: {args.scene}")
    payload = export_scene_collisions(stage)
    payload["source_record_json"] = np.asarray(json.dumps(source_record()))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"Exported {len(load_scene_collisions(payload))} current House colliders to {output}")


if __name__ == "__main__":
    main()
