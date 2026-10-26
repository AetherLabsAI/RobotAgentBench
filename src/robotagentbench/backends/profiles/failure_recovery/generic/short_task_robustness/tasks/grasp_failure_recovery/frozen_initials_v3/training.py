"""Read-only offline training catalogue/miner. No runtime trajectory dependency."""
import argparse
from collections import Counter
import json
from pathlib import Path
from ..common import owned,write_json,digest


def catalogue(root,out):
    root=Path(root).resolve(strict=True);out=owned(out,fresh=True);out.mkdir(parents=True)
    meta=root/'meta';info=json.loads((meta/'info.json').read_text())
    tasks=[json.loads(s) for s in (meta/'tasks.jsonl').read_text().splitlines()]
    episodes=[json.loads(s) for s in (meta/'episode_provenance.jsonl').read_text().splitlines()]
    counts=Counter(p.get('task_index') for p in episodes)
    value=dict(info=info,tasks=[dict(t,actual_episode_count=counts[t['task_index']]) for t in tasks],
        metadata_sha256={p.name:digest(p) for p in (meta/'info.json',meta/'tasks.jsonl',meta/'episode_provenance.jsonl')},
        dataset_root_at_extraction=str(root),runtime_dataset_required=False)
    write_json(out/'catalogue.json',value)
    for task in value['tasks']:print(json.dumps(task,ensure_ascii=False),flush=True)


def mine(root,out,indices,cover_all_phases=False):
    import numpy as np
    import pyarrow.parquet as pq
    root=Path(root).resolve(strict=True);out=owned(out,fresh=True);out.mkdir(parents=True)
    meta=root/'meta';info=json.loads((meta/'info.json').read_text())
    names=info['features']['observation.state']['names'];actions=info['features']['action']['names']
    if info['fps']!=20 or len(names)!=23 or len(actions)!=22 or names[:20]!=actions[:20]:raise ValueError('Original state23/action22 required')
    tasks={t['task_index']:t for t in map(json.loads,(meta/'tasks.jsonl').read_text().splitlines()) if t['task_index'] in indices}
    if set(tasks)!=set(indices):raise ValueError('Unknown exact training task index')
    provenance=list(map(json.loads,(meta/'episode_provenance.jsonl').read_text().splitlines()))
    candidates=[];episodes=[]
    for pr in provenance:
        if pr.get('task_index') not in tasks:continue
        index=pr['episode_index'];path=root/info['data_path'].format(episode_chunk=index//info['chunks_size'],episode_index=index)
        table=pq.read_table(path,columns=['observation.state','action','frame_index','task_index'])
        state=np.asarray(table['observation.state'].to_pylist(),float);action=np.asarray(table['action'].to_pylist(),float)
        frames=np.asarray(table['frame_index']);ti=np.asarray(table['task_index'])
        if state.shape!=(len(frames),23) or action.shape!=(len(frames),22) or not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('Malformed measured training rows')
        if not np.array_equal(frames,np.arange(len(frames))) or not np.all(ti==pr['task_index']):raise ValueError('Training row mismatch')
        grips=action[:,18:20];span=np.ptp(grips,axis=0);active=int(np.argmax(span));hand=('l','r')[active]
        if span[active]<=0:continue
        transition=int(np.argmin(np.diff(grips[:,active])))+1
        minimum=transition+int(np.argmin(grips[transition:,active]))
        # Source phase candidates, not proof of successful physical grasp.
        ids=sorted(set(max(0,min(len(frames)-1,f)) for anchor in (transition,minimum)
            for f in (anchor-10,anchor-5,anchor,anchor+5,anchor+10)))
        if cover_all_phases:ids=sorted(set(ids)|set(map(int,np.linspace(0,len(frames)-1,min(len(frames),64),dtype=int))))
        receipt=dict(episode_index=index,task=tasks[pr['task_index']],active_hand=hand,
            parquet_relative=str(path.relative_to(root)),parquet_sha256=digest(path),
            source_id=pr.get('source_id'),source_path=pr.get('source_path'),
            mapping=pr.get('mapping'),close_transition_frame=transition,first_minimum_frame=minimum,
            frames_examined=ids,source_success_not_inferred=True,all_phase_real_frame_coverage=cover_all_phases)
        write_json(out/f'provenance_{index}.json',pr);receipt['provenance_sha256']=digest(out/f'provenance_{index}.json');episodes.append(receipt)
        for frame in ids:
            candidates.append(dict(id=f'train_ep{index}_f{frame}_{hand}',active_hand=hand,episode_index=index,frame=frame,
                benchmark_task_id=tasks[pr['task_index']].get('benchmark_task_id'),task_index=pr['task_index'],source_row=receipt,
                measured_state=dict(zip(names,map(float,state[frame]))),command=dict(zip(actions,map(float,action[frame]))),
                active_arm_copied_exactly=True,source_joint_interpolation=False,physical_grasp_proven=False))
    value=dict(schema='failure_recovery.training_candidates.v3',tasks=tasks,episodes=episodes,candidates=candidates,
        runtime_dataset_required=False,metadata_sha256={p.name:digest(p) for p in (meta/'info.json',meta/'tasks.jsonl',meta/'episode_provenance.jsonl')})
    write_json(out/'inventory.json',value);print(json.dumps(dict(episodes=len(episodes),candidates=len(candidates),tasks=list(tasks))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('catalogue','mine'));p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--task-indices',type=int,nargs='+');p.add_argument('--cover-all-phases',action='store_true');a=p.parse_args()
    if a.mode=='catalogue':catalogue(a.dataset,a.output)
    else:
        if not a.task_indices:raise ValueError('Explicit reviewed training task indices required')
        mine(a.dataset,a.output,a.task_indices,a.cover_all_phases)
