"""Freeze current task-only operating-initial development source as evidence."""
import hashlib
import json
from pathlib import Path
import shutil
from ..common import HERE,owned,write_json


def main():
    target=owned(HERE/'expansion_v2/versions/operating_v8',fresh=True)
    target.mkdir(parents=True)
    files=['catalog.py','common.py','poses.py','procedural_grasp.py','shelf_checkpoint.py',
        'render_initials.py','render_ready_fleet.py','sim2_runner.py']
    files += ['expansion_v2/'+n for n in ('__init__.py','catalog.py','operating_station.py',
        'empty_grasp.py','ik_branches.py','tipped_initial.py','prepare_base.py',
        'runner.py','candidate_runner.py','preview_runner.py','check_gate.py','audit_cameras.py',
        'freeze_directions.py','paired_direction.py','test_catalog.py','test_empty_grasp.py',
        'test_operating_station.py','test_ik_branches.py','README.md')]
    files += ['expansion_v2/configurations/'+n for n in
        ('sofa_directions_v1.json','cup_directions_held_profile_v2.json')]
    hashes={}
    for name in files:
        source=HERE/name
        if source.is_symlink():raise ValueError('Source snapshots must be real files')
        out=target/name;out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,out)
        hashes[name]=hashlib.sha256(out.read_bytes()).hexdigest()
    pins=json.loads((HERE/'public_component_sources.json').read_text())['task_sources']
    drift=[n for n,h in pins.items() if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h]
    report=dict(schema='failure_recovery.operating_source_snapshot.v1',files=hashes,
        published_v1_task_source_count=len(pins),published_v1_task_source_drift=drift,
        shared_sources_edited=False,task_code_snapshot_only=True,
        cpu_accepted_cases=['sofa_empty401_operating_v8'],native_physics_accepted_cases=[],
        model_calls=0)
    write_json(target/'snapshot.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))
    if drift:raise ValueError('Published v1 task files drifted; do not update their pins')


if __name__=='__main__':main()
