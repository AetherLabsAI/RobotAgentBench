#!/usr/bin/env python3
"""Prepare dynamic clutter before physics, or restore captured physical clutter.

Rendering never samples new physical objects. The explicit --prepare-physics
entrypoint must be followed by the owner action runtime and fresh state capture.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TELEOP = REPO / "real2sim/teleop"
for directory in (REPO, HERE, TELEOP):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

SCHEMA = "real2sim.visual_scene_randomization.v1"
MAX_LAYOUT_ATTEMPTS = 8


def related_path(left: str, right: str) -> bool:
    return (left == right or left.startswith(right.rstrip("/") + "/")
            or right.startswith(left.rstrip("/") + "/"))


def swept_bounds_clear(bounds, positions, radii) -> bool:
    """Conservative captured-body sweep, including linear inter-frame motion."""
    box = np.asarray(bounds, dtype=float).reshape(2, 3)
    points = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if not np.isfinite(points).all() or not np.isfinite(box).all() or not np.isfinite(radii).all():
        return False
    step = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    padding = np.maximum(np.concatenate((step, np.zeros_like(points[:1, :, 0]))),
                         np.concatenate((np.zeros_like(points[:1, :, 0]), step)))
    delta = np.maximum(np.maximum(box[0] - points, points - box[1]), 0.0)
    return bool(np.all(np.linalg.norm(delta, axis=-1) > radii + padding))


def swept_body_boxes(corners, positions, quaternions, clearance_m=.025):
    """Conservative mesh-box sweep with translation and inter-frame rotation."""
    from scipy.spatial.transform import Rotation
    corners = np.asarray(corners, dtype=float).reshape(8, 3)
    positions = np.asarray(positions, dtype=float)
    q = np.asarray(quaternions, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2
            or q.shape != (len(positions), 4) or not np.isfinite(corners).all()
            or not np.isfinite(positions).all() or not np.isfinite(q).all()
            or np.any(np.linalg.norm(q, axis=1) < 1e-8)):
        raise ValueError('invalid captured-body sweep geometry or poses')
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    world = np.einsum('fij,kj->fki', rotations.as_matrix(), corners) + positions[:, None, :]
    low, high = world.min(axis=1), world.max(axis=1)
    angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
    # Endpoint unions cover linear translation; radius*angle also bounds the
    # curved path of every corner between the recorded orientations.
    padding = float(clearance_m) + np.max(np.linalg.norm(corners, axis=1)) * angles
    return np.stack((np.minimum(low[:-1], low[1:]) - padding[:, None],
                     np.maximum(high[:-1], high[1:]) + padding[:, None]), axis=1)


class CapturedBodySweep:
    """Match the renderer's prepared-world-shape and recorded rigid poses."""
    def __init__(self, stage, state):
        import itertools
        from pxr import Gf, UsdGeom
        import visual_scene_geometry as geometry
        self.boxes = []
        self.paths = state['rigid_body_paths'].astype(str).tolist()
        for index, path in enumerate(self.paths):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():
                raise ValueError('missing captured body geometry: ' + path)
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            rotation = np.asarray(Gf.Matrix4d(1.).SetRotate(
                matrix.RemoveScaleShear().ExtractRotationQuat()))[:3, :3]
            vertices = geometry.visible_mesh_triangles(stage, path).reshape(-1, 3)
            if not len(vertices):
                bound = geometry.visible_bounds(stage, path)
                if bound is None:
                    raise ValueError('missing visible captured body: ' + path)
                vertices = np.array(list(itertools.product(*zip(bound[:3], bound[3:]))))
            # Remove only rigid orientation, retaining scale/shear exactly as
            # WorldAffineRigidPoseDriver does; the origin need not be central.
            residual = (vertices - np.asarray(matrix.ExtractTranslation())) @ rotation.T
            corners = np.array(list(itertools.product(*zip(residual.min(0), residual.max(0)))))
            self.boxes.append(swept_body_boxes(corners, state['prop_xyz_m'][:, index],
                                                state['prop_wxyz'][:, index]))
        self.boxes = np.stack(self.boxes, axis=1)
        self.summary = dict(method='visible_mesh_box_recorded_pose_sweep_with_rotation_bound',
                            clearance_m=.025, frames=len(state['prop_xyz_m']),
                            body_paths=self.paths, shape_model='initial_world_affine_residual',
                            physical_collision_rollout=False)

    def clear(self, bounds):
        box = np.asarray(bounds, dtype=float).reshape(2, 3)
        if not np.isfinite(box).all() or np.any(box[1] < box[0]):
            raise ValueError('invalid supplemental geometry')
        overlap = np.all((self.boxes[:, :, 0] < box[1]) &
                         (self.boxes[:, :, 1] > box[0]), axis=-1)
        return not bool(overlap.any())


def bounds_overlap(left, right, clearance_m: float = 0.02) -> bool:
    """Whether two visual AABBs intersect within the supplemental clearance."""
    a = np.asarray(left, dtype=float).reshape(2, 3)
    b = np.asarray(right, dtype=float).reshape(2, 3)
    return bool(np.all(a[0] < b[1] + clearance_m)
                and np.all(b[0] < a[1] + clearance_m))


def density_report(objects, policy):
    """Bound actual dressing per support; safety exclusions need not be refilled."""
    surfaces = {}
    for surface, config in policy['surfaces'].items():
        x0, x1, y0, y1 = config['bounds_xy_m']
        surfaces[surface] = dict(area_m2=(x1-x0)*(y1-y0),
            maximum_count=config['desired_count'], maximum_coverage_fraction=0.30)
    shelf = policy['shelf']
    width = np.diff(shelf['inner_x_bounds_m']).item()
    for surface, groups in [('shelf_beverage', ('beverage_foreground', 'beverage_background')),
                            ('shelf_snack', ('snack_foreground', 'snack_background')),
                            ('shelf_top', ('top_background',))]:
        maximum = (int(shelf['render_distractors_per_layer']) if surface != 'shelf_top'
                   else sum(sum(item.get('production_distractor') is not False for item in shelf[g]) for g in groups))
        surfaces[surface] = dict(area_m2=width*0.40, maximum_count=maximum,
                                maximum_coverage_fraction=0.70)
    for surface, row in surfaces.items():
        items = [x for x in objects if x['surface'] == surface]
        row['count'] = len(items)
        row['footprint_area_m2'] = sum((x['world_bounds'][3]-x['world_bounds'][0]) *
                                     (x['world_bounds'][4]-x['world_bounds'][1]) for x in items)
        row['coverage_fraction'] = row['footprint_area_m2']/row['area_m2']
        row['passed'] = row['count'] <= row['maximum_count'] and row['coverage_fraction'] <= row['maximum_coverage_fraction']
    unknown = sorted({x['surface'] for x in objects}-set(surfaces))
    return dict(passed=not unknown and all(x['passed'] for x in surfaces.values()),
        surfaces=surfaces, unknown_surfaces=unknown,
        policy='Existing surface count caps; at most 30% loose-surface and 70% stocked-shelf conservative footprint coverage. Never refill a motion exclusion to meet a count.')


def shelf_count_report(objects, policy=None):
    """Count supplemental stock separately from the recorded target bodies."""
    if policy is None:
        import episode_scene_randomization as clutter
        policy = clutter.load_policy()
    expected = int(policy['shelf']['render_distractors_per_layer'])
    counts = {surface: sum(x.get('surface') == surface for x in objects)
              for surface in ('shelf_snack', 'shelf_beverage')}
    paths = [x.get('prim_path') for x in objects]
    return dict(passed=all(n == expected for n in counts.values())
                and len(paths) == len(set(paths)), expected_per_layer=expected,
                counts=counts, target_included=False)


def shelf_position_candidates(bounds, board, rng, count=96):
    """Translate a measured, yawed footprint inside the visible support face."""
    box = np.asarray(bounds, dtype=float).reshape(2, 3)
    lower = np.asarray(board['xy_min_m']) + .035
    upper = np.asarray(board['xy_max_m']) - .035
    half = (box[1, :2] - box[0, :2]) / 2
    lo, hi = lower + half, upper - half
    if np.any(lo > hi):
        return
    # Cover edges and both depth rows as well as random interior positions.
    centers = [np.array([x, y]) for x in np.linspace(lo[0], hi[0], 9)
               for y in np.linspace(lo[1], hi[1], 5)]
    centers.extend(np.array([rng.uniform(lo[i], hi[i]) for i in range(2)])
                   for _ in range(count))
    rng.shuffle(centers)
    for center in centers:
        delta = center - box.mean(axis=0)[:2]
        moved = box.copy()
        moved[:, :2] += delta
        yield delta, moved.reshape(-1).tolist()


def shelf_distribution_cost(bounds, board, accepted, slot):
    """Prefer evenly spaced interior positions instead of first-fit edges."""
    low, high = np.asarray(board['xy_min_m']), np.asarray(board['xy_max_m'])
    size = high - low
    center = np.asarray(bounds, dtype=float).reshape(2, 3).mean(0)[:2]
    x_slots = [.15, .325, .5, .675, .85]
    index = [0, 4, 2, 1, 3][slot]
    goal = low + size * [x_slots[index], .35 if index % 2 == 0 else .68]
    cost = float(np.sum(((center - goal) / size) ** 2))
    if accepted:
        centers = np.array([np.asarray(x['world_bounds']).reshape(2, 3).mean(0)[:2]
                            for x in accepted])
        spacing = np.linalg.norm(centers - center, axis=1)
        cost += float(np.sum(np.maximum(.18 - spacing, 0.) ** 2)) * 8
    return cost


def _place_shelf_distractors(stage, scene, policy, protected, xyz, radii,
                            robot_sweep, sofa_triangles, seed, body_sweep):
    """Reuse approved stock and pose authoring, searching both layers safely."""
    from collections import Counter
    from pxr import Usd, UsdGeom, UsdPhysics
    import episode_scene_randomization as clutter
    import visual_scene_geometry as geometry
    from shelf_support_geometry import measure_board_faces, SURFACES
    from shelf_asset_policy import asset_has_approval, verify_refined_bundle

    shelf = policy['shelf']
    expected = int(shelf['render_distractors_per_layer'])
    boards = measure_board_faces(stage)
    rng = random.Random(f'shelf_stock:{seed}')
    placed, evidence, approvals = [], [], {}
    for surface, groups in [('shelf_snack', ('snack_foreground', 'snack_background')),
                            ('shelf_beverage', ('beverage_foreground', 'beverage_background'))]:
        board = boards[SURFACES[surface][0]]
        pool = []
        for group in groups:
            for raw in shelf[group]:
                path = raw['prim_path']
                if raw.get('production_distractor') is False or any(
                        related_path(path, p) or (raw.get('source_prim_path') and
                        related_path(raw['source_prim_path'], p)) for p in protected):
                    continue
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsActive():
                    continue  # Preserve any owner contribution.
                asset = raw.get('asset_id')
                if asset:
                    approved, reason, revision = asset_has_approval(asset)
                    valid, bundle_reason, bundle_revision = verify_refined_bundle(asset, scene.parent)
                    if not approved or not valid:
                        raise ValueError(f'unavailable distractor asset {asset}: {reason}; {bundle_reason}')
                    approvals[asset] = dict(approval_revision=revision, bundle_revision=bundle_revision)
                pool.append(dict(raw))
        accepted = []
        for attempt in range(MAX_LAYOUT_ATTEMPTS):
            for item in accepted:
                stage.GetPrimAtPath(item['prim_path']).SetActive(False)
            accepted = []
            order = list(pool)
            rng.shuffle(order)
            rejected = Counter()
            for raw in order:
                if len(accepted) == expected:
                    break
                path = raw['prim_path']
                found = None
                best = None
                for _ in range(12):
                    yaw = rng.choice((-1, 1)) * rng.uniform(*shelf['render_yaw_abs_range_deg'])
                    item = dict(raw, surface=surface, surface_z_m=board['z_m'],
                                x_m=0., y_m=0., yaw_deg=yaw, dynamic=True,
                                role='interactive_distractor')
                    item.pop('coordinate_frame_prim', None)
                    item.pop('world_y_offset_m', None)
                    bounds, support = geometry.set_render_pose(stage, item, scene.parent,
                        sofa_triangles=sofa_triangles, authored_rest=None)
                    candidates = sorted(shelf_position_candidates(bounds, board, rng),
                        key=lambda pair: shelf_distribution_cost(pair[1], board, accepted, len(accepted)))
                    for delta, candidate in candidates:
                        cost = shelf_distribution_cost(candidate, board, accepted, len(accepted))
                        if best is not None and cost >= best[0]:
                            break
                        if not body_sweep.clear(candidate):
                            rejected['captured_body_swept_clearance'] += 1
                            continue
                        if any(bounds_overlap(candidate, old['world_bounds']) for old in [*placed, *accepted]):
                            rejected['supplemental_prop_overlap'] += 1
                            continue
                        robot = robot_sweep.check(candidate)
                        if not robot['passed']:
                            rejected['robot_full_motion_clearance'] += 1
                            continue
                        if not geometry.sofa_clear(sofa_triangles, candidate):
                            rejected['raised_sofa_geometry'] += 1
                            continue
                        best = (cost, item, delta, candidate, robot)
                        break
                if best is not None:
                    cost, item, delta, candidate, robot = best
                    _, support = geometry.set_render_pose(stage, item, scene.parent,
                        sofa_triangles=sofa_triangles, authored_rest=None)
                    prim = stage.GetPrimAtPath(path)
                    parent_matrix = np.asarray(UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0))
                    local_delta = (np.r_[delta, 0., 0.] @ np.linalg.inv(parent_matrix))[:3]
                    translate = clutter._find_xform_op(prim, UsdGeom.XformOp.TypeTranslate)
                    current = translate.Get()
                    translate.Set(type(current)(*(np.asarray(current) + local_delta)))
                    realized = geometry.visible_bounds(stage, path)
                    if not np.allclose(realized, candidate, atol=1e-6):
                        raise ValueError('shelf candidate world/local placement mismatch')
                    from physical_distractors import validate_bodies
                    validate_bodies(stage, [path])
                    found = dict(item, x_m=float(translate.Get()[0]), y_m=float(translate.Get()[1]),
                                 world_bounds=realized, support_geometry=support,
                                 robot_motion_clearance=robot, planner_attempt=attempt,
                                 planner_seed=seed, placement_method='measured_shelf_distributed_search',
                                 distribution_cost=cost)
                if found:
                    accepted.append(found)
                else:
                    stage.GetPrimAtPath(path).SetActive(False)
            evidence.append(dict(surface=surface, attempt=attempt, applied_count=len(accepted),
                                 rejected_candidate_counts=dict(rejected), visible_support=board))
            if len(accepted) == expected:
                break
        if len(accepted) != expected:
            raise ValueError(f'insufficient safe {surface} stock: {len(accepted)}/{expected}; {evidence[-1]}')
        placed.extend(accepted)
    return placed, evidence, approvals


def render_policy(task_key: str) -> dict:
    import benchmark_eval_registry
    import episode_scene_randomization as clutter

    task_id = task_key.removeprefix("benchmark_").replace("_", ".")
    benchmark_eval_registry.resolve(task_id)
    policy = copy.deepcopy(clutter.load_policy())
    # This is a call-scoped planner input, not another task registry or a
    # translation back to a legacy robot template ID.
    policy["eligible_production_task_ids"] = [task_key]
    return policy


def prepare_for_physics(*, scene: Path, core: Path, task_key: str, seed: int,
          output: Path) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics
    import episode_scene_randomization as clutter
    from git_provenance import source_revision
    from session_scene import extend_session_activation
    from shelf_asset_policy import asset_has_approval, verify_refined_bundle
    import visual_scene_geometry as geometry

    sidecar = scene.with_suffix(scene.suffix + ".session.json")
    session = json.loads(sidecar.read_text(encoding="utf-8"))
    if "scene_randomization_seed" in session:
        raise ValueError(
            "distractors are fixed for this House session; sample once before "
            "executing its tasks, with all task motion/object exclusions")

    from prerender_acceptance import evaluate_core
    acceptance = evaluate_core(task_key, core, require_physical_distractors=False)
    admission_evidence = output.with_name(output.stem + "_core_acceptance.json")
    admission_evidence.parent.mkdir(parents=True, exist_ok=True)
    admission_evidence.write_text(json.dumps(acceptance, indent=2))
    if acceptance.get("passed") is not True:
        raise ValueError(
            "render input failed current task semantic/action-state checks; "
            f"see {admission_evidence}")
    with np.load(core / "full_state.npz", allow_pickle=False) as archive:
        state = {k: np.asarray(archive[k]).copy() for k in archive.files}
    paths = state["rigid_body_paths"].astype(str).tolist()
    xyz = np.asarray(state["prop_xyz_m"], dtype=float)
    target = str(state["target_prim"].item())
    if target not in paths or xyz.shape[1:] != (len(paths), 3):
        raise ValueError("captured target/body arrays do not correspond")
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"cannot open render scene {scene}")
    # Reuse the owner's reviewed materializer for loose props deliberately
    # absent from the shared House root. Do not substitute an unrelated can.
    from init_family7_code_control import ensure_session_assets
    materialization = ensure_session_assets(stage)
    if materialization["empty_can_materialized"]:
        stage.GetPrimAtPath(materialization["empty_can_path"]).SetActive(False)
    policy = render_policy(task_key)
    from robot_visual_clearance import RobotSweep
    robot_sweep = RobotSweep(stage, state)
    body_sweep = CapturedBodySweep(stage, state)
    protected = list(dict.fromkeys([*paths, "/World/Robot"]))
    before = {}
    radii = []
    for path in protected:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError(f"captured/protected prim is not active: {path}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        before[path] = (prim.IsActive(), tuple(map(tuple, transform)))
        if path in paths:
            bound = clutter._world_bound(stage, path)
            if bound is None:
                raise ValueError(f"captured body has no finite bounds: {path}")
            origin = np.asarray(transform.ExtractTranslation())
            corners = np.asarray(bound).reshape(2, 3)
            extent = np.maximum(abs(corners[0] - origin), abs(corners[1] - origin))
            # Includes the open fingers around a held object; all recorded
            # positions are used, not just the authored frame-zero location.
            radii.append(float(np.linalg.norm(extent)) + 0.08)

    descriptor = clutter._target_descriptor(stage, target, policy)
    if descriptor is None:
        raise ValueError("target has no placement descriptor")
    target_origin = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(target))
                               .ComputeLocalToWorldTransform(0).ExtractTranslation())
    offset = xyz[0, paths.index(target)] - target_origin
    for axis, delta in zip(("x_m", "y_m", "z_m"), offset):
        descriptor[axis] += float(delta)
    # Shelf foreground is authored below a translated parent. Background
    # clones must use that same frame rather than the old unshifted world row.
    shelf = stage.GetPrimAtPath("/World/Assets/ShelfItems")
    shelf_matrix = UsdGeom.Xformable(shelf).ComputeLocalToWorldTransform(0)
    shelf_shift = np.asarray(shelf_matrix.ExtractTranslation(), dtype=float)
    if not np.allclose(np.asarray(shelf_matrix)[:3, :3], np.eye(3)):
        raise ValueError("clutter preparation requires the canonical shelf frame")

    applied, rejected, frames = [], [], {}
    sofa_triangles = geometry.visible_mesh_triangles(stage, geometry.SOFA_PATH)
    authored_rest = {}
    approved_assets = {}
    shelf_objects, shelf_search, shelf_approvals = _place_shelf_distractors(
        stage, scene, policy, protected, xyz, radii, robot_sweep, sofa_triangles, seed, body_sweep)
    applied.extend(shelf_objects)
    frames.update({item['prim_path']: item['surface'] for item in shelf_objects})
    approved_assets.update(shelf_approvals)
    planning_attempts = []
    retry_rng = random.Random(f"visual_scene_supplemental_retry:{seed}")
    for attempt in range(MAX_LAYOUT_ATTEMPTS):
        planner_seed = seed if attempt == 0 else retry_rng.randrange(1, 2**63)
        plan = clutter.build_layout_plan(
            task_id=task_key, seed_override=planner_seed, target=descriptor,
            protected_prim_paths=protected, policy=policy)
        if not plan.get("enabled") or not plan.get("objects"):
            raise ValueError("scene randomization was silently disabled or empty")
        attempt_evidence = {"attempt": attempt, "planner_seed": planner_seed,
                            "planned_count": len(plan["objects"]),
                            "applied_before": len(applied)}
        planning_attempts.append(attempt_evidence)
        for original in plan["objects"]:
            if original['surface'] in ('shelf_snack', 'shelf_beverage'):
                continue  # Both layers already have their separately verified stock.
            item = dict(original)
            path = str(item["prim_path"])
            source = str(item.get("source_prim_path", ""))
            if any(related_path(path, p) or (source and related_path(source, p))
                   for p in protected):
                rejected.append({"prim_path": path, "reason": "captured_body_or_source"})
                continue
            existing = stage.GetPrimAtPath(path)
            if existing and existing.IsActive():
                # An owner may already restore captured distractors or supports.
                # Never replace that owner contribution with supplemental dressing.
                rejected.append({"prim_path": path, "reason": "existing_owner_object"})
                continue
            asset_id = item.get("asset_id")
            if asset_id and asset_id not in approved_assets:
                approved, reason, revision = asset_has_approval(asset_id)
                valid, bundle_reason, bundle_revision = verify_refined_bundle(asset_id, scene.parent)
                if not approved or not valid:
                    raise ValueError(f"unavailable distractor asset {asset_id}: {reason}; {bundle_reason}")
                approved_assets[asset_id] = {"approval_revision": revision,
                                             "bundle_revision": bundle_revision}
            if item["surface"].startswith("shelf_"):
                parent = stage.GetPrimAtPath(str(Path(path).parent))
                parent_shift = (np.asarray(UsdGeom.Xformable(parent)
                                .ComputeLocalToWorldTransform(0).ExtractTranslation())
                                if parent else np.zeros(3))
                item["x_m"] += float(shelf_shift[0] - parent_shift[0])
                item["y_m"] += float(shelf_shift[1] - parent_shift[1])
            item["dynamic"] = True
            rest_key = source or path
            if rest_key not in authored_rest:
                rest_prim = stage.GetPrimAtPath(rest_key)
                orient = (clutter._find_xform_op(rest_prim, UsdGeom.XformOp.TypeOrient)
                          if rest_prim else None)
                authored_rest[rest_key] = orient.Get() if orient is not None else None
            try:
                bound, support = geometry.set_render_pose(
                    stage, item, scene.parent, sofa_triangles=sofa_triangles,
                    authored_rest=authored_rest[rest_key])
            except ValueError as exc:
                prim = stage.GetPrimAtPath(path)
                if prim:
                    prim.SetActive(False)
                rejected.append({"prim_path": path, "reason": "visible_support_unavailable",
                                 "detail": str(exc)})
                continue
            if not geometry.sofa_clear(sofa_triangles, bound):
                stage.GetPrimAtPath(path).SetActive(False)
                rejected.append({"prim_path": path, "reason": "raised_sofa_geometry"})
                continue
            if not body_sweep.clear(bound):
                stage.GetPrimAtPath(path).SetActive(False)
                rejected.append({"prim_path": path, "reason": "captured_body_swept_clearance"})
                continue
            robot_check = robot_sweep.check(bound)
            if not robot_check['passed']:
                stage.GetPrimAtPath(path).SetActive(False)
                rejected.append({'prim_path': path, 'reason': 'robot_full_motion_clearance',
                                 **robot_check})
                continue
            # Later candidate plans did not see earlier accepted extras.
            # Keep their actual visual boxes separated as well.
            if any(bounds_overlap(bound, earlier["world_bounds"])
                   for earlier in applied):
                stage.GetPrimAtPath(path).SetActive(False)
                rejected.append({"prim_path": path, "reason": "supplemental_prop_overlap"})
                continue
            prim = stage.GetPrimAtPath(path)
            from physical_distractors import validate_bodies
            validate_bodies(stage, [path])
            frames[path] = str(item["surface"])
            applied.append({**item, "world_bounds": list(bound), "support_geometry": support,
                            "robot_motion_clearance": robot_check,
                            "planner_attempt": attempt, "planner_seed": planner_seed})

        attempt_evidence["applied_after"] = len(applied)
        if len(applied) >= 2:
            break

    if len(applied) < 2:
        output.write_text(json.dumps({
            "schema": SCHEMA, "ok": False, "task_key": task_key, "seed": seed,
            "applied_count": len(applied), "objects": applied, "rejected": rejected,
            "planning_attempts": planning_attempts,
            "reason": "insufficient_trajectory_safe_distractors",
        }, indent=2) + "\n")
        raise ValueError("randomization did not produce multiple usable distractors after bounded retries")
    density = density_report(applied, policy)
    if not density['passed']:
        raise ValueError('actual distractor density exceeds support limits: ' + json.dumps(density))
    for path, expected in before.items():
        prim = stage.GetPrimAtPath(path)
        actual = (prim.IsActive(), tuple(map(tuple, UsdGeom.Xformable(prim)
                  .ComputeLocalToWorldTransform(0))))
        if actual != expected:
            raise ValueError(f"scene randomization modified captured object {path}")
    from house_topology_visual import DECOR_SLOT_ROOTS
    from physical_distractors import ensure_scene_distractor_collisions
    ensure_scene_distractor_collisions(stage,
        [p for group in DECOR_SLOT_ROOTS.values() for p in group
         if geometry.visible_bounds(stage, p) is not None])
    if not stage.GetRootLayer().Save():
        raise RuntimeError("cannot save randomized visual scene")
    extension = extend_session_activation(frames, authored_poses_replaced=True)
    reopened = Usd.Stage.Open(str(scene))
    if any(not reopened.GetPrimAtPath(p).IsActive() for p in frames):
        raise RuntimeError("randomized objects did not persist on reopen")
    manifest = {
        "schema": SCHEMA, "ok": True, "task_key": task_key, "seed": seed,
        "policy_version": clutter.POLICY_VERSION,
        "source_revision": source_revision((Path(__file__), Path(geometry.__file__), HERE / "robot_visual_clearance.py", Path(clutter.__file__), clutter.POLICY_PATH)),
        "planner_enabled": True,
        "planned_count": sum(row["planned_count"] for row in planning_attempts),
        "planning_attempts": planning_attempts,
        "shelf_placement_search": shelf_search,
        "shelf_counts": shelf_count_report(applied, policy),
        "applied_count": len(applied), "objects": applied, "rejected": rejected,
        "extra_prim_frames": frames, "verified_background_assets": approved_assets,
        "session_asset_materialization": materialization,
        "session_activation_extension": extension,
        "captured_bodies_unchanged": True, "captured_body_sweeps_clear": True,
        "captured_body_motion_evidence": body_sweep.summary,
        "robot_full_motion_sweeps_clear": True,
        "robot_motion_evidence": robot_sweep.summary,
        "density": density,
        "captured_body_clearance_radii_m": dict(zip(paths, radii)),
        "source_action_state_modified": False, "physics_rollout_performed": False,
        "additional_props_mode": "dynamic_pending_physics_rollout",
        "physics_collision_validation": "not_performed",
    }
    manifest["applied_prim_paths"] = sorted(frames)
    scene.with_suffix(scene.suffix + ".episode_randomization.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # extend_session_activation writes the same sidecar; retain its activation
    # changes and only add the once-per-environment sampling marker.
    from session_scene import _atomic_sidecar
    session = json.loads(sidecar.read_text(encoding="utf-8"))
    session["scene_randomization_seed"] = int(seed)
    _atomic_sidecar(session)
    return manifest


def apply(*, scene: Path, core: Path, task_key: str, seed: int,
          output: Path) -> dict:
    """Restore only the clutter actually present during the recorded physics.

    The caller's seed is an appearance seed. It cannot resample physical props
    after actions have already run. Preparation for new actions is explicit.
    """
    from pxr import Gf, Usd, UsdGeom
    from physical_distractors import read_capture, require_same_static_scene
    import episode_scene_randomization as clutter
    import visual_scene_geometry as geometry
    with np.load(core / "full_state.npz", allow_pickle=False) as archive:
        state = {k: archive[k] for k in archive.files}
    captured = read_capture(state)
    plan = captured["plan"]
    by_path = {item["prim_path"]: item for item in plan.get("objects", [])}
    paths = [item["prim_path"] for item in captured["objects"]]
    if set(paths) != set(by_path):
        raise ValueError("physical capture must contain the exact randomized object plan")
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"cannot open {scene}")
    require_same_static_scene(stage, captured.get("static_collision_transforms"))
    body_paths = state["rigid_body_paths"].astype(str).tolist()
    objects = []
    for path in paths:
        item = dict(by_path[path])
        # Materialize the same asset; never choose another object or location.
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            clutter._set_pose(stage, item, scene.parent)
            prim = stage.GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)
        world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation = Gf.Matrix4d(1).SetRotate(world.RemoveScaleShear().ExtractRotationQuat())
        shape = Gf.Matrix4d(world)
        shape.SetTranslateOnly(Gf.Vec3d(0))
        shape = shape * rotation.GetInverse()
        index = body_paths.index(path)
        q = state["prop_wxyz"][0, index]
        desired = shape * Gf.Matrix4d(1).SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
        desired.SetTranslateOnly(Gf.Vec3d(*map(float, state["prop_xyz_m"][0, index])))
        local = desired * xform.ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        op = xform.MakeMatrixXform()
        op.Set(local)
        objects.append({**item, "world_bounds": geometry.visible_bounds(stage, path)})
    stage.GetRootLayer().Save()
    result = dict(schema=SCHEMA, ok=True, task_key=task_key, seed=seed,
        physical_seed=plan.get("seed_u64", plan.get("seed")),
        applied_count=len(objects), objects=objects, extra_prim_frames={},
        additional_props_mode="captured_dynamic_physics",
        physics_rollout_performed=True, physics_collision_validation="recorded_live_physx",
        physical_capture=captured, source_action_state_modified=False,
        resampled_after_physics=False, captured_bodies_unchanged=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-physics", action="store_true",
                        help="Prepare dynamic clutter before owner action replay; never render this setup as accepted data")
    args = parser.parse_args()
    operation = prepare_for_physics if args.prepare_physics else apply
    result = operation(scene=args.scene, core=args.core, task_key=args.task_key,
                   seed=args.seed, output=args.output)
    print(json.dumps({"ok": result["ok"], "applied_count": result["applied_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
