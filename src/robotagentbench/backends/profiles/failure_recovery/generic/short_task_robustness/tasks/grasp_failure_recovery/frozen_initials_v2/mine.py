"""Offline-only extraction of real training rows, not an episode-time dependency."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from ..common import owned,write_json,digest


def mine(root,output):
    root=Path(root).resolve(strict=True);out=owned(output,fresh=True);out.mkdir(parents=True)
    meta=root/'meta';info=json.loads((meta/'info.json').read_text())
    names=info['features']['observation.state']['names']
    action_names=info['features']['action']['names']
    if info['fps']!=20 or len(names)!=23 or len(action_names)!=22 or names[:20]!=action_names[:20]:
        raise ValueError('Expected unchanged native House state23/action22 channel contract')
    tasks=[json.loads(s) for s in (meta/'tasks.jsonl').read_text().splitlines()]
    selected={t['task_index']:t for t in tasks if t.get('task_kind')=='atomic' and t.get('benchmark_task_id') in ('6.11','7.10')}
    provenance=[json.loads(s) for s in (meta/'episode_provenance.jsonl').read_text().splitlines()]
    candidates=[];episodes=[]
    for pr in provenance:
        if pr.get('task_index') not in selected:continue
        index=pr['episode_index'];path=root/info['data_path'].format(episode_chunk=index//info['chunks_size'],episode_index=index)
        table=pq.read_table(path,columns=['observation.state','action','frame_index','task_index'])
        state=np.asarray(table['observation.state'].to_pylist(),float);action=np.asarray(table['action'].to_pylist(),float)
        frames=np.asarray(table['frame_index']);ti=np.asarray(table['task_index'])
        if state.shape!=(len(frames),23) or action.shape!=(len(frames),22) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('Invalid numeric training data')
        if not np.array_equal(frames,np.arange(len(frames))) or not np.all(ti==pr['task_index']):raise ValueError('Training row alignment mismatch')
        grip=action[:,18:20];span=np.ptp(grip,axis=0);active=int(np.argmax(span));hand=('l','r')[active]
        if span[active]<=0 or span[1-active]>span[active]:raise ValueError('Cannot identify source operating gripper')
        # First minimum after the largest commanded close transition. These
        # are command-phase candidates, never claimed to prove physical grasp.
        close=int(np.argmin(np.diff(grip[:,active])))+1
        tail=grip[close:,active];first_min=close+int(np.argmin(tail))
        frame_ids=sorted(set((close,first_min,min(len(frames)-1,first_min+5),min(len(frames)-1,first_min+10))))
        receipt=dict(episode_index=index,task=selected[pr['task_index']],parquet_relative=str(path.relative_to(root)),
            parquet_sha256=digest(path),source_id=pr.get('source_id'),source_path=pr.get('source_path'),
            provenance_sha256=None,active_hand=hand,command_gripper_span_rad=span.tolist(),
            close_transition_frame=close,first_minimum_frame=first_min,frames_examined=frame_ids,
            original_source_approval_not_inferred=True,base_pose_not_per_frame_telemetry=pr.get('mapping',{}).get('base_projection_requires_review'))
        write_json(out/f'provenance_{index}.json',pr);receipt['provenance_sha256']=digest(out/f'provenance_{index}.json')
        episodes.append(receipt)
        for frame in frame_ids:
            candidates.append(dict(id=f'train_ep{index}_f{frame}_{hand}',active_hand=hand,episode_index=index,frame=frame,
                benchmark_task_id=selected[pr['task_index']]['benchmark_task_id'],source_row=receipt,
                measured_state=dict(zip(names,map(float,state[frame]))),command=dict(zip(action_names,map(float,action[frame]))),
                source_joint_interpolation=False,source_action_clipped=False,physical_grasp_proven=False))
    result=dict(schema='failure_recovery.training_cup_candidates.v2',dataset_root_at_extraction=str(root),
        dataset_metadata_sha256={p.name:digest(p) for p in (meta/'info.json',meta/'tasks.jsonl',meta/'episode_provenance.jsonl')},
        runtime_dataset_required=False,episodes=episodes,candidates=candidates,
        counts={h:sum(e['active_hand']==h for e in episodes) for h in ('l','r')},fps=20)
    write_json(out/'inventory.json',result)
    print(json.dumps(dict(episodes=len(episodes),candidates=len(candidates),hands=result['counts'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args();mine(a.dataset,a.output)
