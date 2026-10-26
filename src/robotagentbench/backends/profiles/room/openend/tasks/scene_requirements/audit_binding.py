"""CPU USD checks of actual source geometry, no simulated facts."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[2]),str(HERE.parents[2]/'api-protocol')]
from pxr import Usd,UsdPhysics
from openend.room import prepare_registration
from evaluation.tasks.find_object.isaac_evidence import body_colliders
from openend.tasks.glasses_transfer.native_geometry import SceneGeometry as Glasses
from openend.tasks.wallet_transfer.native_geometry import SceneGeometry as Wallet
from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.scene import Scene
folder=Path(sys.argv[1]).resolve()
assert folder.parent==HERE/'generated_rooms'
manifest=json.loads((folder/'composition.json').read_text());stage=Usd.Stage.Open(manifest['scene'])
result={'registration':prepare_registration(json.loads((folder/'runtime_request.json').read_text()),manifest),'native_physics_validated':False,'drawer':[]}
for obj in manifest['objects']:
    if obj.get('source')=='coffee_table.drawer':
        g=(Glasses if obj['id']=='glasses_case' else Wallet)(stage,manifest)
        walls=body_colliders(stage,obj['drawer_path'])
        assert any('Bottom' not in name for name in walls)
        result['drawer'].append(dict(target=g.target,body=g.body,drawer=obj['drawer_path'],colliders=list(walls),rigid_dynamic=True))
for task in manifest['tasks']:
    if task['type']=='fridge_to_microwave':
        s=Scene(stage,manifest,task['parameters']['payload_variant'])
        result['fridge']={'target':s.target,'variant':s.variant,'source_paths':s.source_paths}
(folder/'binding_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
