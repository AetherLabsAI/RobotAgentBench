"""One approved robot-station rectangle for all four sofa-to-table targets.

Pure proposal generation only. The caller must supply a verified, floor-preserving
row-vector rigid transform from the calibrated sofa to the actual layout.
No object transforms, joints, physics or scoring are changed here.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np


def sample_reset_candidates(seed, delta, context, count=100):
    if context.get('asset') == 'remote_tv':
        ranges=posture_policy()['ranges']
        digest=hashlib.sha256(f'sofa-remote-data-posture-v1:{seed}'.encode()).digest()
        rng=random.Random(int.from_bytes(digest,'big'))
        for row in sample_candidates(seed, np.asarray(delta).tolist(), context, count=count):
            targets={name:rng.uniform(*limits) for name,limits in ranges.items()}
            yield dict(row,targets_rad=targets,posture_source='remote_task119_first_state_continuous_ranges')
        return
    from openend.tasks.layout_repair_v1.ready_envelope import REFERENCE
    ready=json.loads((REFERENCE/'navigation_result.json').read_text())
    if ready['status']!='native_action_sequence_completed':
        raise ValueError('Measured folded-ready reference required')
    targets=ready['navigation_posture']['targets_rad']
    for row in sample_candidates(seed, np.asarray(delta).tolist(), context, count=count):
        yield dict(row,targets_rad=dict(targets),posture_source='approved_generic_folded_ready_not_dataset_fit')

POLICY = Path(__file__).with_name("initial_station_policy.json")
POSTURE_POLICY = Path(__file__).with_name('remote_posture_ranges.json')


def posture_policy():
    return json.loads(POSTURE_POLICY.read_text())


def policy():
    return json.loads(POLICY.read_text())


def rigid_delta(value):
    try:
        a = [list(row) for row in value]
    except TypeError as exc:
        raise ValueError("A finite 4x4 sofa rigid transform is required") from exc
    if len(a) != 4 or any(len(row) != 4 for row in a):
        raise ValueError("A finite 4x4 sofa rigid transform is required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in a for v in row):
        raise ValueError("Finite numeric transform required")
    tol = 1e-7
    expected = {(0,2):0, (1,2):0, (2,0):0, (2,1):0, (2,2):1,
                (0,3):0, (1,3):0, (2,3):0, (3,2):0, (3,3):1}
    if any(abs(a[i][j]-v)>tol for (i,j),v in expected.items()):
        raise ValueError("Transform must be upright and floor-height preserving")
    for i in range(2):
        for j in range(2):
            if abs(sum(a[i][k]*a[j][k] for k in range(2))-(i==j))>tol:
                raise ValueError("Sofa transform must be rigid; no scale or shear")
    if abs(a[0][0]*a[1][1]-a[0][1]*a[1][0]-1)>tol:
        raise ValueError("Sofa reflection is not a valid layout rotation")
    return a


def sample_candidates(seed, delta, context, *, count):
    """Same seed gives the same base proposals for every supported asset.

    Proposals are not a collision, reachability or native settling certificate.
    Reject infeasible candidates downstream without widening this rectangle.
    """
    p = policy()
    if context.get("asset") not in p["assets"] or context.get("source") != "sofa":
        raise ValueError("Supported sofa target required")
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError("Positive uint32 seed required")
    if type(count) is not int or count < 1:
        raise ValueError("Positive candidate count required")
    a = rigid_delta(delta)
    digest = hashlib.sha256(f"sofa-to-table-shared-station-v1:{seed}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    result = []
    for i in range(count):
        distance = rng.uniform(*p["front_distance_m"])
        right = rng.uniform(*p["right_offset_m"])
        angle = rng.uniform(*p["yaw_deviation_deg"])
        local = [p["canonical_seat_front_x_m"]-distance,
                 p["canonical_sofa_centerline_y_m"]-right, 0., 1.]
        world = [sum(local[k]*a[k][j] for k in range(4)) for j in range(3)]
        yaw = math.atan2(a[0][1], a[0][0])+math.radians(angle)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        result.append(dict(candidate_index=i,
            sampling=p["sampling"], initial_base_xyyaw=[world[0],world[1],yaw],
            canonical_base_xyz=local[:3], edge_distance_m=distance,
            right_offset_m=right, yaw_deviation_deg=angle,
            robot_room_geometry_checked=False, native_physics_validated=False))
    return result

def sofa_context(stage, manifest):
    from pxr import Usd, UsdGeom
    from short_tasks.agent_probe_v1.target_spawn import ROOT
    baseline = ROOT / 'layout_task_activation_v1/calibration_drafts_layout01.json'
    rows = [r for r in json.loads(baseline.read_text())['entries'] if r['id'] == 'sofa']
    if len(rows) != 1:
        raise ValueError('One calibrated sofa reference required')
    ref = rows[0]['target']
    prim = stage.GetPrimAtPath(ref['anchor'])
    if not prim or not prim.IsActive():
        raise ValueError('Actual sofa anchor missing')
    cache = UsdGeom.XformCache()
    reference_scene = Path(manifest['source_layout']).parent / 'ID_01_LIVING_STRAIGHT.usdc'
    original = Usd.Stage.Open(str(reference_scene))
    if not original:
        raise ValueError('Source sofa reference scene missing')
    def meshes(scene):
        transform = UsdGeom.XformCache()
        result = {}
        for child in Usd.PrimRange(scene.GetPrimAtPath(ref['anchor']), Usd.TraverseInstanceProxies()):
            if child.IsA(UsdGeom.Mesh) and child.IsActive():
                points = np.asarray(child.GetAttribute('points').Get(), dtype=float)
                if points.size:
                    matrix = np.asarray(transform.GetLocalToWorldTransform(child))
                    result[str(child.GetPath())] = (np.c_[points, np.ones(len(points))] @ matrix)[:,:3]
        return result
    aa, bb = meshes(original), meshes(stage)
    if not aa or set(aa) != set(bb) or any(aa[k].shape != bb[k].shape for k in aa):
        raise ValueError('Sofa mesh topology differs from source')
    a, b = (np.concatenate([m[k] for k in sorted(m)]) for m in (aa, bb))
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    rotation = u @ vt
    delta = np.eye(4)
    delta[:3,:3] = rotation
    delta[3,:3] = bc-ac@rotation
    delta = np.asarray(rigid_delta(delta.tolist()))
    if not np.allclose(a@rotation+delta[3,:3], b, atol=1e-5, rtol=0):
        raise ValueError('Sofa vertex transfer is not rigid')
    bbox = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'], False, True)
    # Root transformation alone is insufficient if someone edited children.
    measured = np.array([a.min(0), a.max(0)])
    expected = np.array([ref['bounds']['min'], ref['bounds']['max']])
    if not np.isfinite(measured).all() or not np.allclose(measured, expected, atol=1e-5, rtol=0):
        raise ValueError('Sofa child geometry differs from the source frame')
    owned = [r for r in manifest['objects'] if r.get('task_id') == 'sofa_to_coffee_table']
    if len(owned)!=1 or owned[0].get('source')!='sofa' or owned[0].get('id') not in policy()['assets']:
        raise ValueError('No approved sofa-to-table station for these bindings')
    row = owned[0]
    target = stage.GetPrimAtPath(row['path'])
    if not target or not target.IsActive():
        raise ValueError('Scene-owned sofa-to-table target missing')
    bounds = bbox.ComputeWorldBound(target).ComputeAlignedRange()
    centre = (np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
    return delta, dict(asset=row['id'], source=row['source'], target_xy=centre[:2].tolist(),
        target_path=row['path'], sofa_path=ref['anchor'],
        reference_files=[str(baseline),str(reference_scene)]), baseline
