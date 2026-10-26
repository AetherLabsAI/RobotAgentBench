"""Guarded task-local deployment; preserves original host and room manifests."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
EXPECTED={
 'multi_task_room_v2.py':'7571fbda6b40c6e9134f3e04a4bfe8010e2a39e6670c29518f8792ddc48f0321',
 'build_room_demo_v1.py':'1ed79ded0695fd584dca47aadbbef5cbfdc96dc39d5e891c6812af331cc62f0f',
 'build_other_layouts_v1.py':'e95e388452c634be4b561293ce8ade6a52ff0fd3f691e4ef3c3f2298b7e4c3fa',
 'initialize_room_fridge_boxes_v1.py':'197fb988dd54db4a31d8ff051e832c3955732e915043c20cd3940946927c1ae3',
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in EXPECTED.items():
    if sha(ROOT/name)!=expected:raise RuntimeError('Concurrent change: '+name)
    compile((HERE/'integration'/('candidate_'+name)).read_text(),name,'exec')
for name in EXPECTED:
    backup=HERE/'integration'/('before_'+name)
    with backup.open('xb') as f:f.write((ROOT/name).read_bytes())
for name in EXPECTED:
    tmp=ROOT/(name+'.task9.tmp')
    with tmp.open('xb') as f:f.write((HERE/'integration'/('candidate_'+name)).read_bytes())
    tmp.replace(ROOT/name)
from fridge_to_microwave_v1.support import extend
manifests=[ROOT/p/'composition.json' for p in ('room_demo_v6','coffee_initialized_rooms_v1/layout03','coffee_initialized_rooms_v1/layout04')]
for m in manifests:extend(m)
receipt=dict(status='deployed',scope='task_owned_unified_room_host_not_shared_HTTP_registry',
             files={name:sha(ROOT/name) for name in EXPECTED},
             manifests={str(m):sha(m) for m in manifests},request=str(ROOT/'multi_task_episode_task9_v1.json'),
             full_trajectory_verified=False)
(HERE/'deployment_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
