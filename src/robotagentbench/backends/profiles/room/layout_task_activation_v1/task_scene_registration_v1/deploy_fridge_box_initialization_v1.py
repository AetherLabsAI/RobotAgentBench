"""Hash-guarded task-owned deployment; never modifies shared assets."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
STAGED=ROOT/'fridge_box_initialization_deploy_v1'
EXPECTED={'build_room_demo_v1.py':'2d6410aabc3a67a0454fa21a5584fc371988d9957ad7b315d74115b7d1cebbcc',
          'build_other_layouts_v1.py':'6695171e95cee272c3a592e1d7f4e1af0f9aa4dd75e2be42ec615d99f4135581'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    for name,digest in EXPECTED.items():
        assert sha(ROOT/name)==digest,('Concurrent edit',name)
        assert not (STAGED/(name+'.before')).exists()
    for name in EXPECTED:
        shutil.copy2(ROOT/name,STAGED/(name+'.before'))
    for name in [*EXPECTED,'initialize_room_fridge_boxes_v1.py']:
        shutil.copy2(STAGED/name,ROOT/name)
    report=dict(deployed_sha256={n:sha(ROOT/n) for n in [*EXPECTED,'initialize_room_fridge_boxes_v1.py']},
        before_sha256=EXPECTED,scope=str(ROOT),shared_assets_modified=False,
        cpu_tests=dict(passed=8,sampler_seeds=1000,slot_assignments=24),
        native_probe=str(ROOT/'fridge_box_settle_v1.S5ekWP/result.json'),
        native_scope='one_seed_45_layout03_isolated_fridge_natural_initialization_settle_2.2667s_not_full_task',
        initial_pose_writes_only=True,runtime_docking=False,
        current_asset_comparison=dict(beef='243mm nominal length retained; current standard collision structure',
            food='same L1 source hash; existing 0.887 scale baked into geometry',
            salad='same L1 source hash; post-shortened-rim approximately 192.5mm, not stale 220mm comment'),
        source_initialization='L1 runs/fridge-repair-20260910/producer_data/mw_studio/layout_gen.py and ep_run.py; four slots, 3cm lateral, 6mm depth, outward yaw 0..20deg, lean 14deg plus original small tilt/roll perturbations',
        pending=['General random-seed native coverage','Door-open and full take/put trajectories'])
    (STAGED/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
