"""Read-only checks on existing 2/4-product rooms; no scene or render writes."""
import hashlib
import json
from pathlib import Path
from pxr import Usd
from openend.room import prepare_registration
from .shelf_binding import SceneBinding
from .shelf_instances import KIND, peer_bodies

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    for episode in ('room_0592566103f4', 'room_03356b86b5fc'):
        folder=HERE/'generated_rooms'/episode
        mpath=folder/'composition.json'; request_path=folder/'runtime_request.json'
        manifest=json.loads(mpath.read_text()); request=json.loads(request_path.read_text())
        protected=[mpath,request_path,Path(manifest['scene'])]
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        registration=prepare_registration(request,manifest)
        # Verification only. Never relabel old generated-room acceptance records.
        manifest['registration']=registration
        stage=Usd.Stage.Open(manifest['scene'])
        checks=[]
        for task in request['tasks']:
            if task['type']!=KIND: continue
            b=SceneBinding(stage,manifest,task['parameters']['product'],episode,task['instance_id'])
            peers=peer_bodies(manifest,task['instance_id'],task['parameters']['product'],b.body_by_wrapper)
            checks.append(dict(instance_id=task['instance_id'],target_bodies=list(b.identity.candidates),
                               peer_bodies=sorted(peers),maximum_score=7))
        assert len(checks) in (2,4) and all(len(c['peer_bodies'])==len(checks)-1 for c in checks)
        assert len({c['target_bodies'][0] for c in checks})==len(checks)
        assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        rows.append(dict(episode=episode,tasks=checks,source_hashes=before,unchanged=True))
        print(episode, len(checks), 'independent USD bindings passed', flush=True)
    report=dict(scope='actual_cpu_usd_binding_not_native_manipulation',rooms=rows,
                physics_started=False,render_started=False,existing_rooms_modified=False)
    with (HERE/'shelf_instances_usd_verification_20261006.json').open('x') as f:
        json.dump(report,f,indent=2)


if __name__=='__main__':main()
