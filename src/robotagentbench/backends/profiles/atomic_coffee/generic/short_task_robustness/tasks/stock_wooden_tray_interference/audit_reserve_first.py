"""Collect bounded CPU evidence; never deploy shared files."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
I=HERE/'integration'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=I/'reserve_first_v1/audit.json'
    if out.exists():raise ValueError('Audit output already exists')
    tests=read(I/'reserve_first_checks_004/result.json')
    assert tests['success'] and tests['skips']==0
    old=read(I/'reserve_first_ordinary_before/composition.json')
    new=read(I/'reserve_first_ordinary_after/composition.json')
    ordinary={key:old[key]==new[key] for key in ('objects','tasks','shelf_initialization')}
    assert all(ordinary.values())
    shared={name:sha(ROOT/'openend/tasks/scene_requirements'/name)==sha(I/'reserve_first_v1/before'/name)
            for name in ('build_shelf.py','build_selected_room.py')}
    assert all(shared.values())
    rooms=[]
    for directory in ('reserve_first_room_006','reserve_first_room_007'):
        report=read(I/directory/'report.json');m=read(I/directory/'composition.json')
        assert report['success'] and report['source_unchanged']
        targets=[o for o in m['objects'] if o['id']=='gatorade_t']
        assert len(targets)==1
        r=m['environment_reservations'][0]
        stock=[o for o in m['objects'] if o.get('source')=='snack_shelf']
        count=sum(m['shelf_initialization']['metrics']['layer_counts'].values())
        assert len(stock)==count
        rooms.append(dict(seed=m['seed'],target_count=1,shelf_stock_count=count,
            destination_xy=r['destination_xy'],target_bounds=r['initial_target_bounds'],
            frozen_recheck_passed=True,source_unchanged=True))
    receipt=dict(tests=tests,ordinary_comparison=ordinary,shared_files_unchanged=shared,rooms=rooms,
        different_seed_destinations=rooms[0]['destination_xy']!=rooms[1]['destination_xy'],
        deployed=False,source_pin_update_permission_pending=True,
        full_room_entry_validated=False,native_physics_validated=False,gpu_started=False,
        task_code_hashes={name:sha(HERE/name) for name in ('generation.py','shelf_geometry.py',
            'initialization.py','generate_reserved_room.py','check_reserve_first.py')},
        candidate_hashes={name:sha(I/'reserve_first_v1/candidate'/name)
            for name in ('build_shelf.py','build_selected_room.py')})
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
