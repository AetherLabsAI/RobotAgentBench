"""Read saved native poses to explain a rejected route without a new GPU run."""
import json
from pathlib import Path
import numpy as np
from .geometry import collision_shapes, box, swept, intersect
from .moving_frame import pose_matrix


def inspect(output, initial):
    from pxr import Usd, UsdGeom, UsdPhysics
    from evaluation.tasks.coffee_machine.specs import MACHINE_BODY, BAG, COUNTER
    from evaluation.tasks.coffee_machine.isaac_evidence import body_colliders
    from arena_api import l1_scene
    output, initial = Path(output).resolve(), Path(initial).resolve()
    here = Path(__file__).resolve().parent
    if not output.is_relative_to(here) or not initial.is_relative_to(here):
        raise ValueError('Task-owned saved diagnostic only')
    report = json.loads((output / 'report.json').read_text())
    with (output / 'physical_evidence/physical_facts.jsonl').open() as stream:
        last = None
        for line in stream:
            last = json.loads(line)
    if last is None:
        raise ValueError('No saved native frame')
    stage = Usd.Stage.Open(str(initial))
    stage.SetEditTarget(stage.GetSessionLayer())
    l1_scene.apply(stage)  # Same source masks, session-only; never saved.
    cache = UsdGeom.XformCache()

    def native_vertices(path):
        points = np.concatenate([points for points, _ in body_colliders(stage, path).values() if points is not None])
        pose = last['poses'][path]
        matrix = pose_matrix(pose['position_m'], pose['quaternion_wxyz'])
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    proposal = report['route_admission'].get('proposal')
    if proposal is None:
        from .push_motion import proposal as sample
        plan = json.loads((initial.parent / 'initial.json').read_text())
        body = last['poses'][MACHINE_BODY]
        # Existing asset Body is identity inside Machine; assert, don't assume.
        body_local, _ = cache.GetLocalTransformation(stage.GetPrimAtPath(MACHINE_BODY))
        if not np.allclose(np.asarray(body_local), np.eye(4), atol=1e-9, rtol=0):
            raise ValueError('Nonidentity body-to-root needs the stored moving-frame contract')
        proposal = sample(plan['seed'], pose_matrix(body['position_m'], body['quaternion_wxyz']))
    delta = np.array(proposal['direction_world_xy_unit']) * proposal['distance_m']
    machine_route = swept(box(native_vertices(MACHINE_BODY)), delta)
    bag = box(native_vertices(BAG))
    fixture_candidates = []
    bounds = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'])
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Background')):
        path = str(prim.GetPath())
        if path == COUNTER or path.startswith(COUNTER + '/'):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False:
            r = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            if not r.IsEmpty() and intersect(machine_route, np.array([r.GetMin(), r.GetMax()])):
                fixture_candidates.append(path)
    result = dict(saved_native_time_s=last['time_s'], machine_swept_box_m=machine_route.tolist(),
                  held_bag_now=last['contacts'][BAG]['grasped'], bag_box_m=bag.tolist(),
                  bag_in_machine_sweep=intersect(machine_route, bag),
                  authored_background_candidates=fixture_candidates,
                  robot_full_geometry_not_reconstructed=True,
                  no_new_physics_run=True, no_source_or_scene_files_modified=True)
    (output / 'route_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    inspect(sys.argv[1], sys.argv[2])
