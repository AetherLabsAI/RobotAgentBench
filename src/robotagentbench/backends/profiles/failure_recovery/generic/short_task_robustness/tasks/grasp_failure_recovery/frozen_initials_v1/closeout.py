"""Freeze the task-owned fast-initial implementation and honest CPU evidence."""
import json
from pathlib import Path
import shutil
from ..common import HERE,owned,digest,write_json


def main():
    root=HERE/'frozen_initials_v1';bank=root/'configurations/bank_v1.json'
    pins=json.loads((HERE/'public_component_sources.json').read_text())['task_sources']
    drift=[n for n,h in pins.items() if digest(HERE/n)!=h]
    batch=json.loads((root/'samples/cup_random10_v1/summary.json').read_text())
    smoke=HERE/'expansion_v2/candidates/cup_frozen_default_smoke_v1/initial.json'
    v=json.loads(smoke.read_text())
    if not v['preparation_accepted'] or v.get('online_ik_calls')!=0:raise ValueError('Default dispatch did not use accepted frozen path')
    out=owned(root/'versions/v1',fresh=True);out.mkdir(parents=True)
    files=[*root.glob('*.py'),root/'README.md',bank,HERE/'expansion_v2/candidate_runner.py']
    hashes={}
    for path in files:
        if path.is_symlink():raise ValueError('No alias in source snapshot')
        name=path.relative_to(HERE);dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);hashes[str(name)]=digest(dst)
    report=dict(source_files=hashes,bank_bytes=bank.stat().st_size,bank_sha256=digest(bank),
        published_v1_task_sources=len(pins),published_v1_drift=drift,shared_entries_edited=False,
        default_task_local_cup_dispatch_accepted=True,default_smoke_sha256=digest(smoke),
        cpu_random_draws=batch['requested'],cpu_accepted=batch['accepted'],hands=batch['hands'],
        distinct_target_poses=batch['distinct_target_poses'],optimizer_calls=batch['optimizer_calls'],
        timing_seconds=batch['timing_seconds'],whole_rooms_generated=batch['fresh_whole_rooms_generated'],
        native_physics_tested=False,model_calls=0)
    write_json(out/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_files'}))
    if drift:raise ValueError('Published source drift: do not update pins')


if __name__=='__main__':main()
