"""Read-only actual-room dependencies; never a reset or initial-pose gate."""
import hashlib
import json
from pathlib import Path
from pxr import Usd,UsdPhysics
from evaluation.tasks.microwave_put.physical_geometry import PLATE_BODY,STAND,BODY,HINGE
from evaluation.tasks.microwave_put.payload_binding import PATHS
from evaluation.tasks.refrigerator_take.variants import VARIANTS

ROOT=Path('/home/lingjun/robot_agent_arena/layout_task_activation_v1/task_scene_registration_v1')
OUT=ROOT/'fridge_to_microwave_v1'
FR='/World/Background/Shell/RefrigeratorRecovered'


def audit(path):
    manifest=json.loads(path.read_text());stage=Usd.Stage.Open(manifest['scene'])
    tracked={str(l.realPath):hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
             for l in stage.GetUsedLayers() if l.realPath}
    def present(p):return bool(stage.GetPrimAtPath(p) and stage.GetPrimAtPath(p).IsActive())
    candidates=[]
    for p in stage.Traverse():
        name=str(p.GetPath())
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and any(x in name.lower() for x in ('beef','food','salad','plate','microwave','refrigerat')):
            candidates.append(dict(path=name,enabled=p.GetAttribute('physics:rigidBodyEnabled').Get(),
                                   kinematic=p.GetAttribute('physics:kinematicEnabled').Get()))
    canonical={p:present(p) for p in [FR,FR+'/left_door_hinge',FR+'/right_door_hinge',
                                    BODY,HINGE,PLATE_BODY,STAND,*PATHS.values(),
                                    VARIANTS['26-1'].body,VARIANTS['26-2'].body]}
    food_manifest=[o for o in manifest['objects'] if any(t in o['id'].lower() for t in ('beef','leftover','foodbox'))]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in tracked.items())
    return dict(manifest=str(path),scene=manifest['scene'],canonical_paths=canonical,
                manifest_food=food_manifest,all_candidate_rigid_bodies=candidates,
                sources_unchanged=True,reset_performed=False,physical_acceptance=False)


if __name__=='__main__':
    rows=[audit(p) for p in [ROOT/'room_demo_v6/composition.json',
           ROOT/'coffee_initialized_rooms_v1/layout03/composition.json',
           ROOT/'coffee_initialized_rooms_v1/layout04/composition.json']]
    with (OUT/'room_dependencies_v1.json').open('x') as f:json.dump(rows,f,indent=2)
    for r in rows:
        print(r['scene'],json.dumps(r['canonical_paths']), 'food_manifest_count',len(r['manifest_food']))
