"""Conservative visible-geometry checks for supplemental render dressing.

These checks never modify captured bodies or assert physical collision safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent / "teleop"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

SOFA_PATH = "/World/Assets/Sofa"
SUPPORT_GAP_M = 0.001
SUPPORT_CONTACT_TOLERANCE_M = 0.015


def triangles_intersect_box(triangles, bounds) -> np.ndarray:
    """Triangle/AABB SAT, including crossings with no contained vertices."""
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    box = np.asarray(bounds, dtype=float).reshape(2, 3)
    if not np.isfinite(triangles).all() or not np.isfinite(box).all():
        raise ValueError("nonfinite visible geometry")
    if np.any(box[1] < box[0]):
        raise ValueError("inverted visual box")
    center, half = box.mean(axis=0), (box[1] - box[0]) / 2
    vertices = triangles - center
    hit = np.all(vertices.min(axis=1) <= half + 1e-10, axis=1)
    hit &= np.all(vertices.max(axis=1) >= -half - 1e-10, axis=1)
    indices = np.flatnonzero(hit)
    v = vertices[indices]
    if not len(v):
        return hit
    edges = np.roll(v, -1, axis=1) - v
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    axes.extend(np.cross(edges[:, edge], axis) for edge in range(3)
                for axis in np.eye(3))
    overlaps = np.ones(len(v), dtype=bool)
    for axis in axes:
        projection = np.einsum("nij,nj->ni", v, axis)
        radius = np.abs(axis) @ half
        overlaps &= (projection.min(axis=1) <= radius + 1e-10)
        overlaps &= (projection.max(axis=1) >= -radius - 1e-10)
    hit[indices] = overlaps
    return hit


def visible_mesh_triangles(stage, path: str) -> np.ndarray:
    """Triangulate visible render meshes in their actual world transforms."""
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsActive():
        return np.empty((0, 3, 3))
    parts = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if (imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
                or imageable.ComputePurpose() not in
                (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points.ndim != 2 or points.shape[1] != 3 or counts is None or indices is None:
            raise ValueError(f"unreadable visible mesh: {prim.GetPath()}")
        transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=float)
        world = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        indices = np.asarray(indices, dtype=int)
        if len(indices) != sum(counts) or np.any(indices < 0) or np.any(indices >= len(points)):
            raise ValueError(f"invalid visible mesh topology: {prim.GetPath()}")
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        cursor = 0
        for face, count in enumerate(counts):
            ids = indices[cursor:cursor + count]
            if face not in holes:
                parts.extend(world[[ids[0], ids[j], ids[j + 1]]]
                             for j in range(1, count - 1))
            cursor += count
    return np.asarray(parts, dtype=float).reshape(-1, 3, 3)


def visible_bounds(stage, path: str) -> list[float] | None:
    """Use mesh points, never authored mesh extent boxes, for mesh bounds."""
    from pxr import Usd, UsdGeom
    triangles = visible_mesh_triangles(stage, path)
    points = [triangles.reshape(-1, 3)] if len(triangles) else []
    root = stage.GetPrimAtPath(path)
    if not root or not root.IsActive():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Boundable) or prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if (imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
                or imageable.ComputePurpose() not in
                (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if not box.IsEmpty():
            points.append(np.asarray([box.GetMin(), box.GetMax()], dtype=float))
    if not points:
        return None
    points = np.concatenate(points)
    if not np.isfinite(points).all():
        raise ValueError(f"nonfinite visible geometry: {path}")
    return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]


def _clip_polygon(polygon, axis, limit, keep_above):
    output = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        a, b = start[axis] - limit, end[axis] - limit
        inside_a = a >= 0 if keep_above else a <= 0
        inside_b = b >= 0 if keep_above else b <= 0
        if inside_a:
            output.append(start)
        if inside_a != inside_b:
            output.append(start + (-a / (b - a)) * (end - start))
    return np.asarray(output, dtype=float).reshape(-1, 3)


def sofa_support_height(triangles, bounds, nominal_z):
    """Maximum visible seat height inside the actual prop footprint.

    Exclude triangles rising into arms/backrests before clipping; a large
    whole-sofa box would otherwise place every prop on top of the backrest.
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    box = np.asarray(bounds, dtype=float).reshape(2, 3)
    seat = triangles[(triangles[:, :, 2].max(axis=1) <= nominal_z + 0.025)
                     & (triangles[:, :, 2].max(axis=1) >= nominal_z - 0.025)]
    heights = []
    for triangle in seat:
        polygon = triangle
        for axis in (0, 1):
            polygon = _clip_polygon(polygon, axis, box[0, axis], True)
            if not len(polygon):
                break
            polygon = _clip_polygon(polygon, axis, box[1, axis], False)
            if not len(polygon):
                break
        if len(polygon):
            heights.append(float(polygon[:, 2].max()))
    if not heights:
        raise ValueError("supplemental prop footprint has no visible sofa seat support")
    return max(heights)


def sofa_clear(triangles, bounds) -> bool:
    """Reject raised sofa geometry inside the prop, allowing seat contact."""
    box = np.asarray(bounds, dtype=float).reshape(2, 3).copy()
    box[0, 2] += SUPPORT_CONTACT_TOLERANCE_M
    if box[0, 2] > box[1, 2]:
        return True
    box[0, :2] -= 0.01
    box[1, :2] += 0.01
    return not bool(triangles_intersect_box(triangles, box).any())


def set_render_pose(stage, item, demo_root, *, sofa_triangles, authored_rest):
    """Apply the existing planner pose, then seat only this supplemental prop."""
    from pxr import Gf, UsdGeom
    import episode_scene_randomization as clutter
    clutter._set_pose(stage, item, demo_root)
    prim = stage.GetPrimAtPath(item["prim_path"])
    orient = clutter._find_xform_op(prim, UsdGeom.XformOp.TypeOrient)
    if orient is not None and item.get("rest_mode") == "flat" and authored_rest is not None:
        yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(item["yaw_deg"])).GetQuat()
        rest = Gf.Quatd(authored_rest)
        quaternion = yaw * rest
        old = orient.Get()
        orient.Set(type(old)(quaternion.GetReal(), type(old.GetImaginary())(*quaternion.GetImaginary())))
    bounds = visible_bounds(stage, item["prim_path"])
    if bounds is None:
        raise ValueError("supplemental prop has no visible geometry")
    support = float(item["surface_z_m"])
    if item["surface"] == "sofa":
        support = sofa_support_height(sofa_triangles, bounds, support)
    shift = support + SUPPORT_GAP_M - bounds[2]
    translate = clutter._find_xform_op(prim, UsdGeom.XformOp.TypeTranslate)
    current = translate.Get()
    translate.Set(type(current)(current[0], current[1], current[2] + shift))
    bounds = visible_bounds(stage, item["prim_path"])
    if item["surface"] == "sofa":
        sofa = clutter.load_policy()["surfaces"]["sofa"]
        xmin, xmax, ymin, ymax = item.get("placement_bounds_xy_m", sofa["bounds_xy_m"])
        margin = float(item.get("placement_edge_margin_m", sofa["edge_margin_m"]))
        if not (xmin + margin <= bounds[0] and bounds[3] <= xmax - margin
                and ymin + margin <= bounds[1] and bounds[4] <= ymax - margin):
            raise ValueError("sofa prop final orientation crosses middle two seats")
    return bounds, {"method": "visible_mesh_support", "support_z_m": support,
        "visible_bottom_z_m": bounds[2], "vertical_correction_m": shift,
        "authored_flat_rest_preserved": orient is not None and authored_rest is not None
                                      and item.get("rest_mode") == "flat"}


def audit_final_scene(stage, objects, decor_paths) -> dict:
    """Reject geometric conflicts after layout and decoration have moved."""
    from visual_scene_randomization import bounds_overlap
    sofa = visible_mesh_triangles(stage, SOFA_PATH)
    decor = {path: visible_bounds(stage, path) for path in decor_paths}
    from physical_distractors import ensure_scene_distractor_collisions
    collision_rows = ensure_scene_distractor_collisions(
        stage, [item["prim_path"] for item in objects] + [p for p in decor_paths
            if visible_bounds(stage, p) is not None], author=False)
    observed, failures = {}, []
    for item in objects:
        path = item["prim_path"]
        bounds = visible_bounds(stage, path)
        if bounds is None:
            failures.append({"prim_path": path, "reason": "missing_visible_geometry"})
            continue
        observed[path] = bounds
        if not sofa_clear(sofa, bounds):
            failures.append({"prim_path": path, "reason": "raised_sofa_geometry"})
        for other, box in decor.items():
            if box is not None and bounds_overlap(bounds, box, clearance_m=0.005):
                failures.append({"prim_path": path, "other_prim": other,
                                 "reason": "final_decoration_bounds_overlap"})
        for other, box in observed.items():
            if other != path and bounds_overlap(bounds, box, clearance_m=0.005):
                failures.append({"prim_path": path, "other_prim": other,
                                 "reason": "final_supplemental_bounds_overlap"})
    return {"schema": "real2sim.visual_scene_geometry.v1", "ok": not failures,
        "objects_checked": len(objects), "collision_geometry": collision_rows, "visible_world_bounds": observed,
        "failures": failures, "method": "visible_mesh_and_conservative_decoration_aabb",
        "physics_collision_validation": "not_performed", "scene_modified": False}


def prune_motion_conflicts(stage, manifest, sweep, state, body_sweep=None):
    """Remove declared extras and nonphysical decor from protected sweeps."""
    import copy
    from pxr import Usd, UsdPhysics
    from house_topology_visual import DECOR_SLOT_ROOTS
    from visual_scene_randomization import related_path, swept_bounds_clear
    paths = state['rigid_body_paths'].astype(str).tolist()
    protected = [*paths, '/World/Robot']
    objects = manifest['objects']
    if len({x['prim_path'] for x in objects}) != len(objects):
        raise ValueError('duplicate supplemental prim identity')
    if any(related_path(item['prim_path'], path) for item in objects for path in protected):
        raise ValueError('refusing to prune captured or robot geometry')
    radii = [manifest['captured_body_clearance_radii_m'][p] for p in paths]
    removed = []
    for item in objects:
        box = visible_bounds(stage, item['prim_path'])
        if box is None:
            raise ValueError('missing declared supplemental geometry')
        robot = sweep.check(box)
        body_clear = (body_sweep.clear(box) if body_sweep is not None
                      else swept_bounds_clear(box, state['prop_xyz_m'], radii))
        if not robot['passed'] or not body_clear:
            removed.append(dict(prim_path=item['prim_path'],
                reason='post_layout_motion_exclusion', robot=robot,
                captured_body_sweep_clear=body_clear, final_world_bounds=box))
    excluded = {x['prim_path'] for x in removed}
    if len(objects) - len(excluded) < 2:
        raise ValueError('insufficient motion-safe distractors after layout')
    decor_removed = []
    for slots in DECOR_SLOT_ROOTS.values():
        for path in slots:
            prim = stage.GetPrimAtPath(path)
            box = visible_bounds(stage, path)
            if box is None:
                continue
            robot = sweep.check(box)
            body_clear = (body_sweep.clear(box) if body_sweep is not None
                          else swept_bounds_clear(box, state['prop_xyz_m'], radii))
            if robot['passed'] and body_clear:
                continue
            if any(related_path(path, captured) for captured in protected):
                raise ValueError('refusing to prune captured decoration')
            marker = prim.GetAttribute('real2sim:renderOnlyDecoration')
            if not marker or marker.Get() is not True:
                continue  # It remains visible and must fail the final audit.
            physical = any(
                (p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False)
                or (p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False)
                for p in Usd.PrimRange(prim))
            if physical:
                continue  # Never remove actual physics to make a video pass.
            decor_removed.append(dict(prim_path=path,
                reason='render_only_decoration_motion_exclusion', robot=robot,
                captured_body_sweep_clear=body_clear, final_world_bounds=box,
                render_only_marker_verified=True, enabled_physics_absent=True))
    updated = copy.deepcopy(manifest)
    for path in excluded | {x['prim_path'] for x in decor_removed}:
        stage.GetPrimAtPath(path).SetActive(False)
    updated['objects'] = [x for x in updated['objects'] if x['prim_path'] not in excluded]
    for item in updated['objects']:
        item['pre_layout_world_bounds'] = item['world_bounds']
        item['world_bounds'] = visible_bounds(stage, item['prim_path'])
        item['robot_motion_clearance'] = sweep.check(item['world_bounds'])
    updated['applied_count'] = len(updated['objects'])
    updated['extra_prim_frames'] = {p:f for p,f in updated['extra_prim_frames'].items() if p not in excluded}
    updated['rejected'] = [*updated.get('rejected', []), *removed, *decor_removed]
    updated['render_only_decoration_motion_exclusions'] = decor_removed
    updated['post_layout_motion_exclusions'] = [*removed, *decor_removed]
    return updated


def main():
    from pxr import Usd
    from house_topology_visual import DECOR_SLOT_ROOTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--randomization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--prune-motion-conflicts", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.randomization.read_text())
    stage = Usd.Stage.Open(str(args.scene))
    if not stage or manifest.get("ok") is not True or len(manifest.get("objects", [])) < 2:
        raise ValueError("missing prepared scene or usable randomization manifest")
    from robot_visual_clearance import RobotSweep
    from visual_scene_randomization import density_report, render_policy, swept_bounds_clear, shelf_count_report, CapturedBodySweep
    with np.load(args.state, allow_pickle=False) as archive:
        state = {k: archive[k] for k in archive.files}
    if manifest.get("additional_props_mode") == "captured_dynamic_physics":
        from physical_distractors import read_capture, require_same_static_scene
        capture = read_capture(state)
        require_same_static_scene(stage, capture.get("static_collision_transforms"))
        from physical_distractors import ensure_scene_distractor_collisions
        ensure_scene_distractor_collisions(stage,
            [item["prim_path"] for item in manifest["objects"]] +
            [p for group in DECOR_SLOT_ROOTS.values() for p in group
             if visible_bounds(stage, p) is not None], author=False)
        # Never prune a collider because the recorded robot touched it.
        missing = [o["prim_path"] for o in manifest["objects"]
                   if visible_bounds(stage, o["prim_path"]) is None]
        result = dict(ok=not missing, scene_modified=False, missing_objects=missing,
                      physics_collision_validation="recorded_live_physx",
                      final_randomization_manifest=manifest)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return 0 if result["ok"] else 1
    sweep = RobotSweep(stage, state)
    body_sweep = CapturedBodySweep(stage, state)
    original = manifest
    if args.prune_motion_conflicts:
        manifest = prune_motion_conflicts(stage, manifest, sweep, state, body_sweep)
        if manifest['post_layout_motion_exclusions']:
            if not stage.GetRootLayer().Save():
                raise RuntimeError('cannot persist supplemental motion exclusions')
            stage = Usd.Stage.Open(str(args.scene))
            if any(stage.GetPrimAtPath(x['prim_path']).IsActive()
                   for x in manifest['post_layout_motion_exclusions']):
                raise RuntimeError('motion exclusions did not persist on reopen')
    result = audit_final_scene(stage, manifest["objects"],
                              [p for paths in DECOR_SLOT_ROOTS.values() for p in paths])
    motion_bounds = dict(result['visible_world_bounds'])
    for paths in DECOR_SLOT_ROOTS.values():
        for path in paths:
            box = visible_bounds(stage, path)
            if box is not None:
                motion_bounds[path] = box
    robot_checks = {p: sweep.check(box) for p, box in motion_bounds.items()}
    paths = state['rigid_body_paths'].astype(str).tolist()
    radii = [manifest['captured_body_clearance_radii_m'][p] for p in paths]
    body_checks = {p: body_sweep.clear(box)
                   for p, box in motion_bounds.items()}
    objects = [dict(item, world_bounds=result['visible_world_bounds'][item['prim_path']])
               for item in manifest['objects'] if item['prim_path'] in result['visible_world_bounds']]
    result['density'] = density_report(objects, render_policy(manifest['task_key']))
    result['shelf_counts'] = shelf_count_report(objects)
    result['robot_motion'] = dict(sweep.summary, checks=robot_checks,
                                  passed=all(x['passed'] for x in robot_checks.values()))
    result['captured_body_motion'] = dict(body_sweep.summary, checks=body_checks, passed=all(body_checks.values()))
    result['ok'] = (result['ok'] and result['robot_motion']['passed']
                    and result['captured_body_motion']['passed'] and result['density']['passed']
                    and result['shelf_counts']['passed'])
    result['scene_modified'] = bool(manifest.get('post_layout_motion_exclusions'))
    if args.prune_motion_conflicts:
        manifest['density'] = result['density']
        manifest['shelf_counts'] = result['shelf_counts']
        manifest['robot_full_motion_sweeps_clear'] = result['robot_motion']['passed']
        manifest['captured_body_sweeps_clear'] = result['captured_body_motion']['passed']
        args.randomization.with_name('scene_randomization_before_final_motion.json').write_text(json.dumps(original, indent=2)+'\n')
        args.randomization.write_text(json.dumps(manifest, indent=2)+'\n')
        result['final_randomization_manifest'] = manifest
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
