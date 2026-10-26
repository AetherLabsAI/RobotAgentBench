"""Immutable task-only source record for the approved arm-first v4 pilot."""
import json
from pathlib import Path
import shutil
from ..common import HERE,owned,digest,write_json


def main():
    out=owned(HERE/'expansion_v2/versions/arm_first_v4',fresh=True);out.mkdir(parents=True)
    files=['common.py','poses.py','procedural_grasp.py','render_ready_fleet.py','sim2_runner.py']
    files+=['expansion_v2/'+n for n in (
        'catalog.py','empty_grasp.py','arm_first_cup.py','arm_first_visible_cup.py',
        'arm_first_visible_cup_v4.py','camera_projection.py','camera_projection_complete.py',
        'geometric_direction.py','operating_station.py','ik_branches.py','tipped_initial.py',
        'prepare_base.py','runner.py','candidate_runner.py','preview_runner.py','test_arm_first_cup.py',
        'test_catalog.py','test_empty_grasp.py','test_operating_station.py','test_ik_branches.py',
        'test_geometric_direction.py','README.md','freeze_arm_first.py',
        'configurations/cup_directions_geometry_v5.json',
        'candidates/cup_armfirst401_visible_v4/initial.json')]
    hashes={}
    for name in files:
        src=HERE/name
        if src.is_symlink():raise ValueError('No source aliases in immutable task snapshot')
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        hashes[name]=digest(dst)
    pins=json.loads((HERE/'public_component_sources.json').read_text())['task_sources']
    drift=[n for n,h in pins.items() if digest(HERE/n)!=h]
    write_json(out/'snapshot.json',dict(files=hashes,published_v1_task_source_count=len(pins),
        published_v1_task_source_drift=drift,shared_sources_edited=False,
        cpu_accepted_initial='cup_armfirst401_visible_v4',native_physics_verified=False,
        model_calls=0,scoring_changed=False))
    print(json.dumps(dict(files=len(hashes),published_v1_drift=drift)))
    if drift:raise ValueError('Published v1 drift; do not rewrite fingerprints')


if __name__=='__main__':main()
