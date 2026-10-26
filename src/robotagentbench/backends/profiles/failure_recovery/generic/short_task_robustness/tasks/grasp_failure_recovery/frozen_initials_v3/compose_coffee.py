"""Data-only contexts from the existing pinned original atomic coffee sampler."""
import argparse
import json
from ..common import HERE,owned,finish,write_json


if __name__=='__main__':
    from ..procedural_coffee import compose
    from ..procedural_grasp import complete_closed_reset
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args()
    out=owned(a.output,fresh=True);out.mkdir(parents=True);rows=[]
    cfg=json.loads((HERE/'procedural_config.json').read_text());cfg['config_path']=str(HERE/'procedural_config.json')
    for seed in a.seeds:
        stage,record,before=compose(out/str(seed),seed,cfg)
        record['robot_initialization'],record['gripper_closure']=complete_closed_reset(stage,record['robot_initialization'],'r')
        record.update(preparation_accepted=False,status='original_atomic_object_context_only_not_failure_pose',
            online_ik_calls=0,source_motion_read=False)
        value=finish(stage,out/str(seed),record,before)
        rows.append(dict(seed=seed,scene_sha256=value['scene_sha256'],sampler='unchanged_existing_atomic_1_2'))
    write_json(out/'contexts.json',dict(rows=rows,models_called=False,physics_run=False))
