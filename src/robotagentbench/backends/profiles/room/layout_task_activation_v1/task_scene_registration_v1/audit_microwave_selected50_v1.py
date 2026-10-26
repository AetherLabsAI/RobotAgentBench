"""Current 9.2 selected-50 initial placement audit, read-only source geometry."""
import json
from pathlib import Path
from audit_microwave_placement_v1 import measure, sha, HERE, LAYOUT

index = Path('/mnt/agent-bench-data/lingjun/arena-data-review-20260920/datasets/task9_selected50_20261001/task_9.2_selected50.json')
selection = json.loads(index.read_text())
assert selection['selected_count'] == len(selection['selected']) == 50
rows = []
for i, item in enumerate(selection['selected']):
    source = Path(item['sim_info_dir'])
    original_source = str(source)
    if not (source/'initial_scene.usda').exists():
        copies = {'lerobot_beef':'microwave_beef', 'lerobot_beef_grasp20':'microwave_beef_grasp'}
        name = Path(item['dataset']).name
        source = index.parents[1]/copies[name]/'sim_info'/source.name
        # Verify copied episode identity against 17 selected numeric features,
        # not just the reused episode number in two different datasets.
        audit = json.loads((source/'randomization_audit.json').read_text())
        features = item['features']
        for j, v in enumerate(audit['base_requested']['xy_delta_m']):
            assert abs(v-features[f'base_requested.xy_delta_m.{j}']) < 1e-12
        assert abs(audit['base_requested']['yaw_delta_deg']-features['base_requested.yaw_delta_deg']) < 1e-12
        for j, v in enumerate(audit['arm_initial_delta_rad']):
            assert abs(v-features[f'arm_initial_delta_rad.{j}']) < 1e-12
    scene = source/'initial_scene.usda'
    if not scene.exists():
        rows.append(dict(source=str(source),status='missing_scene'))
        print(i+1,'MISSING',source,flush=True)
        continue
    _,_,_,result = measure(scene)
    rows.append(dict(source=str(source),original_source=original_source,scene_sha256=sha(scene),status='measured',**result))
    print(i+1,result['metrics']['front_to_operator_edge_m'],flush=True)
valid=[r for r in rows if r['status']=='measured']
assert valid, 'No measurable initial scenes; do not report an empty range'
summary={k:dict(min=min(r['metrics'][k] for r in valid),max=max(r['metrics'][k] for r in valid),
                   min_source=min(valid,key=lambda r:r['metrics'][k])['source'],
                   max_source=max(valid,key=lambda r:r['metrics'][k])['source']) for k in valid[0]['metrics']}
_,_,_,current=measure(LAYOUT)
report=dict(schema='arena.microwave.selected50_placement.v1',task='9.2',index=str(index),index_sha256=sha(index),
            selected_count=50,measured_count=len(valid),summary=summary,records=rows,current_layout=current,
            limitations=['Default-time visual AABBs, not contact geometry; operator edge is west/-X in these source layouts.',
                         'Source recordings, not a claim of new physical validation. No door/arms motion range in this initial-state report.',
                         'Does not include 9.4 leftover-box put or 10.1 take. Not a frozen shared microwave placement rule.'])
(HERE/'microwave_selected50_placement_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('SUMMARY',json.dumps(summary),flush=True)
