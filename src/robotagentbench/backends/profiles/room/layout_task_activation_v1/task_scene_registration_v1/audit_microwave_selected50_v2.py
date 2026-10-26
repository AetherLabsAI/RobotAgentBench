"""Four bounded CPU readers: selected 9.2 initial scenes and door motion only."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_microwave_placement_v1 import measure, sha, HERE, LAYOUT, local_visible_bounds, DOOR, MICRO, corners, bounds

INDEX = Path('/mnt/agent-bench-data/lingjun/arena-data-review-20260920/datasets/task9_selected50_20261001/task_9.2_selected50.json')

def audit(item):
    original = Path(item['sim_info_dir']); source = original
    if not (source/'initial_scene.usda').exists():
        copies = {'lerobot_beef':'microwave_beef','lerobot_beef_grasp20':'microwave_beef_grasp'}
        source = INDEX.parents[1]/copies[Path(item['dataset']).name]/'sim_info'/source.name
        a=json.loads((source/'randomization_audit.json').read_text());f=item['features']
        for j,v in enumerate(a['base_requested']['xy_delta_m']): assert abs(v-f[f'base_requested.xy_delta_m.{j}'])<1e-12
        assert abs(a['base_requested']['yaw_delta_deg']-f['base_requested.yaw_delta_deg'])<1e-12
        for j,v in enumerate(a['arm_initial_delta_rad']): assert abs(v-f[f'arm_initial_delta_rad.{j}'])<1e-12
    scene=source/'initial_scene.usda'; control=source/'control_state_60hz.npz'
    scene_hash,control_hash=sha(scene),sha(control)
    stage,cache,matrix,result=measure(scene)
    door=local_visible_bounds(stage,cache,matrix,DOOR)
    with np.load(control,allow_pickle=False) as data:
        poses=data['body_pose'][:,list(data['rigid_body_paths']).index(DOOR),:]
    assert np.isfinite(poses).all()
    transforms=np.broadcast_to(np.eye(4),(len(poses),4,4)).copy()
    transforms[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix();transforms[:,:3,3]=poses[:,:3]
    swept=np.einsum('nij,kj->nki',np.linalg.inv(matrix(MICRO))@transforms,corners(door['min'],door['max']))[:,:,:3]
    assert sha(scene)==scene_hash and sha(control)==control_hash
    return dict(source=str(source),original_source=str(original),scene_sha256=scene_hash,control_sha256=control_hash,
        source_copy_identity='17 selected randomization features checked when using review-site copy',
        door_frames=len(poses),door_sweep_in_microwave=bounds(swept),**result)

def main():
    selection=json.loads(INDEX.read_text());assert selection['selected_count']==len(selection['selected'])==50
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(audit,selection['selected']):
            rows.append(result)
            print(len(rows),result['metrics']['front_to_operator_edge_m'],flush=True)
    summary={k:dict(min=min(r['metrics'][k] for r in rows),max=max(r['metrics'][k] for r in rows),
        min_source=min(rows,key=lambda r:r['metrics'][k])['source'],max_source=max(rows,key=lambda r:r['metrics'][k])['source']) for k in rows[0]['metrics']}
    _,_,_,current=measure(LAYOUT)
    union=bounds([v for r in rows for v in (r['door_sweep_in_microwave']['min'],r['door_sweep_in_microwave']['max'])])
    m=np.array(current['microwave_matrix'])
    current['put_only_door_sweep_projection']=bounds((m@corners(union['min'],union['max']).T).T[:,:3])
    report=dict(schema='arena.microwave.selected50_placement.v2',task='9.2',selected_count=50,measured_count=len(rows),
        index=str(INDEX),index_sha256=sha(INDEX),summary=summary,records=rows,current_layout=current,
        door_sweep_in_microwave=union,door_control_frames=sum(r['door_frames'] for r in rows),
        limitations=['All 50 selected 9.2 source recordings; not fifty validated API successes.',
            'Initial visual AABBs measured from west/operator table edge, not collision geometry.',
            'Door envelope computed from visible-geometry local box transformed by all recorded 60Hz body poses; no 240Hz or inter-frame certificate.',
            'Door-only projection into current layout, not a validated trajectory retarget.',
            'Arms, carried-box motion, leftover task 9.4 and take task 10.1 NOT included. No final microwave purple exclusion box or frozen distance policy.'])
    (HERE/'microwave_selected50_placement_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)
    print('DOOR',json.dumps(current['put_only_door_sweep_projection']),flush=True)

if __name__=='__main__':main()
