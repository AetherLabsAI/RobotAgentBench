"""Geometric side-cup proposal, not a forged physical early-release record.

Before-run pose authoring is task initialization. Native settling, visibility,
contact/support and any claimed grasp/lift/release history remain separate gates.
The same cup, bin binding, room, mass, colliders, drives and score are preserved.
"""
import argparse
import json
import math
import os
from pathlib import Path
from ..common import CURRENT, HERE, digest, owned, physics, finish, write_json
from .prepare_base import NAMESPACE


def load_base(key, expected):
    if not key.replace('_', '').isalnum():
        raise ValueError('Simple existing preparation key required')
    base = NAMESPACE/key/'base/native'
    launch = json.loads((base/'launch.json').read_text())
    manifest = json.loads((base/'manifest.json').read_text())
    bound = json.loads((base/'bindings.json').read_text())
    ready = json.loads((base/'ready_pose.json').read_text())
    from short_tasks import registry
    from short_tasks.scene import verify_manifest
    verify_manifest(manifest)
    if bound['task_name'] != expected or launch['task_name'] != expected:
        raise ValueError('Wrong original full-tree task')
    registry.verify_sources(bound)
    if not Path(manifest['scene']).resolve().is_relative_to(NAMESPACE):
        raise ValueError('Only isolated fresh full-tree scenes')
    return base, launch, manifest, bound, ready


def build(key, output):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdGeom, UsdPhysics
    from ..procedural_grasp import collider_points, complete_closed_reset
    from ..poses import world, place
    from short_tasks.tasks.clear_table_trash.initial_station import table_context
    base, launch, manifest, bound, ready = load_base(key, 'clear_table_trash')
    declarations = bound['runtime_components']
    if len(declarations) != 1 or declarations[0]['parameters']['category'] != 'cup':
        raise ValueError('Exactly one explicit used-cup component required')
    binding = declarations[0]['parameters']
    target = binding['target_path']
    rows = [o for o in manifest['objects'] if o['path'] == target]
    if len(rows) != 1 or rows[0]['id'] != 'paper_cup' or rows[0]['trash_category'] != 'cup':
        raise ValueError('Exact scene-owned paper cup required')
    out = owned(output, fresh=True); out.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(out/'initial.usda'))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(manifest['scene'], out)]
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    before = physics(stage)
    prim = stage.GetPrimAtPath(target)
    bodies = [UsdPhysics.RigidBodyAPI(p) for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies())
              if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies)!=1 or not bodies[0].GetRigidBodyEnabledAttr().Get() or bodies[0].GetKinematicEnabledAttr().Get():
        raise ValueError('Original cup must contain exactly one unchanged dynamic body')
    delta, table, reference = table_context(stage, manifest)
    old = world(stage, target)
    points = np.concatenate([r[3] for r in collider_points(stage, target)])
    # A ninety-degree sideways state follows the requested geometry, not a
    # new random angle range. Retain original sampled planar origin and yaw.
    axis = np.array(delta[0, :3], dtype=float)
    turn = Rotation.from_rotvec(axis*(math.pi/2)).as_matrix()
    origin = old[:3, 3].copy()
    rotated = (points-origin)@turn.T+origin
    old_gap = float(points[:,2].min()-table[1,2])
    proposed = old.copy(); proposed[:3,:3] = turn@old[:3,:3]
    proposed[2,3] += float(table[1,2]+old_gap-rotated[:,2].min())
    place(stage, target, proposed)
    actual = np.concatenate([r[3] for r in collider_points(stage, target)])
    canonical = np.c_[actual, np.ones(len(actual))]@np.linalg.inv(delta)
    table_fit = bool(np.all(canonical[:,:2] >= table[0,:2]) and
                     np.all(canonical[:,:2] <= table[1,:2]))
    # Restore unchanged task base station, not the asset's authored default.
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    base_receipt=initialize_base_pose(stage,launch['initial_base_xyyaw'])
    issued = dict(positions=ready['targets_rad'], hold_targets=ready['targets_rad'])
    initialization, linkage = complete_closed_reset(stage, issued, 'r')
    initialization['positions'].update(linkage['source_coordinates'])
    for n in initialization['hold_targets']:
        initialization['hold_targets'][n] = initialization['positions'][n]
    record = dict(variant='table_waste_tipped_cup', seed=manifest['seed'],
        failure_kind='early_release_tipped_cup', target=target,
        destination=binding['destination_path'], category='cup', original_request=launch['request'],
        dynamic_body=str(bodies[0].GetPrim().GetPath()),
        original_bindings=bound, original_launch=launch,
        robot_initialization=initialization, robot_base_policy='unchanged_original_task_station',
        robot_base_initialization=base_receipt,
        maximum_score=bound['score_policy']['maximum'],
        completion_policy='same_cup_released_inside_correct_bin',
        original_target_world=old.tolist(), proposed_target_world=proposed.tolist(),
        cup_side_pose_authored=True, original_planar_origin_preserved=True,
        support_gap_preserved_m=old_gap, whole_collider_envelope_on_table=table_fit,
        source_motion_read=False, source_grasp_lift_release_replayed=False,
        physical_failure_generation_verified=False, preparation_accepted=False,
        visible_after_settle_verified=False, released_after_settle_verified=False,
        supported_side_rest_verified=False,
        code_sources={str(Path(__file__).resolve()):digest(__file__),str(reference):digest(reference)},
        base_input_hashes={str(base/n):digest(base/n) for n in
            ('launch.json','manifest.json','bindings.json','ready_pose.json')})
    result = finish(stage, out, record, before)
    result['layout_scope'] = 'full_original_short_task_random_room'
    result['status'] = 'geometric_candidate_only_not_accepted_failure_start'
    write_json(out/'initial.json', result)
    if not table_fit:
        raise ValueError('Sideways cup does not wholly fit original table; no planar relocation or seed retry')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-key', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    value=build(a.base_key,a.output)
    print(json.dumps({k:value[k] for k in ('variant','seed','whole_collider_envelope_on_table',
        'physics_unchanged','physical_settle_verified','status','scene_sha256')},ensure_ascii=False))


if __name__ == '__main__': main()
