"""Deploy only the three approved files after exact before-hash checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STAGED=HERE/'integration/reserve_first_v1'
FILES=('openend/tasks/scene_requirements/build_shelf.py',
       'openend/tasks/scene_requirements/build_selected_room.py','short_tasks/scene_sources.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    receipt=STAGED/'deployment.json'
    if receipt.exists():raise ValueError('Deployment already recorded')
    if not json.loads((STAGED/'audit.json').read_text())['tests']['success']:raise ValueError('Tests not passed')
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[p for p in processes.splitlines() if 'python' in p and any(token in p for token in
        ('build_selected_room.py','generate_reserved_room.py','short_tasks.agent_probe_v1.prepare_task','short_tasks.agent_probe_v1.native_owner'))]
    if active:raise ValueError('Preparation/runtime active; defer deployment: '+str(active))
    before=json.loads((STAGED/'before/scene_sources.json').read_text())
    after=json.loads((STAGED/'candidate/scene_sources.json').read_text())
    changed=[k for k in before['sources'] if before['sources'][k]!=after['sources'].get(k)]
    assert set(changed)==set(FILES[:2]) and before.keys()==after.keys()
    check=json.loads(json.dumps(after));check['sources']=before['sources'];assert check==before
    rows={}
    for name in FILES:
        target=ROOT/name;backup=STAGED/'before'/target.name;source=STAGED/'candidate'/target.name
        if target.is_symlink() or sha(target)!=sha(backup):raise ValueError('Shared file drift: '+name)
        if target.suffix=='.py':
            compile(source.read_text(),name,'exec');assert sha(source)==after['sources'][name]
        rows[name]=dict(before=sha(target),after=sha(source))
    done=[]
    try:
        for name in FILES:
            shutil.copyfile(STAGED/'candidate'/Path(name).name,ROOT/name);done.append(name)
        from short_tasks.scene import verify_scene_sources
        verified=verify_scene_sources()
    except BaseException:
        for name in reversed(done):shutil.copyfile(STAGED/'before'/Path(name).name,ROOT/name)
        raise
    receipt.write_text(json.dumps(dict(files=rows,source_verification=verified,
        task_policy_only=True,native_runtime_available=False),indent=2)+'\n')
    print(receipt.read_text())

if __name__=='__main__':main()
