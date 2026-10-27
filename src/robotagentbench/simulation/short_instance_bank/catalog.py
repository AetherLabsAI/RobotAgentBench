"""Aggregate saved preparations, preserving rejected candidates and real seeds.

This is a read-only artifact check and immutable catalogue export. It does not
call a sampler, prepare a scene, start Isaac or interpret a process score.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .bank import TASKS, SCHEMA, read, sha, semantic_fingerprint

CATALOG='robotagentbench.short_instance_catalog.v31'


def collect(indexes,*,translate=lambda p:Path(p)):
    candidates={};history=[];sources=[]
    for path in map(Path,indexes):
        value=read(path)
        if value.get('schema')!=SCHEMA:raise ValueError('Original prepared short index required')
        sources.append(dict(path=str(path),sha256=sha(path)))
        for original in value['instances']:
            row=deepcopy(original)
            if row.get('task_name')not in TASKS or type(row.get('seed'))is not int or not 1<=row['seed']<=90:
                raise ValueError('Invalid saved task or original candidate seed')
            row['source_index']=str(path)
            if row['status']!='prepared_cpu_not_native_accepted':
                history.append(row);continue
            for source,expected in row['output_hashes'].items():
                if sha(translate(source))!=expected:raise ValueError('Saved preparation changed: '+source)
            row['semantic_fingerprint']=semantic_fingerprint(row['physical_configuration'])
            row['read_only_reload_passed']=True
            key=(row['task_name'],row['seed'])
            if key in candidates:
                old=candidates[key]
                if (old['launch_sha256'],old['semantic_fingerprint'])!=(row['launch_sha256'],row['semantic_fingerprint']):
                    raise ValueError('Different successful preparations claim the same task/seed')
            else:candidates[key]=row
    rows=[];counts={}
    for task in TASKS:
        distinct=set();selected=[]
        all_rows=sorted((r for r in candidates.values()if r['task_name']==task),key=lambda r:r['seed'])
        for row in all_rows:
            signature=row['semantic_fingerprint']
            if signature in distinct:continue
            distinct.add(signature)
            if len(selected)<30:
                row['episode_ordinal']=len(selected)+1;row['actual_seed']=row['seed']
                selected.append(row)
        rows.extend(selected)
        counts[task]=dict(selected=len(selected),distinct_available=len(distinct),
            rejected_or_infrastructure_records=sum(r['task_name']==task for r in history))
    return dict(schema=CATALOG,source_indexes=sources,counts=counts,instances=rows,
        preserved_failed_preparations=history,all_30_distinct_valid=all(c['selected']==30 for c in counts.values()),
        native_accepted=False,model_called=False,resampled=False)


def load(path,task,ordinal,*,translate=lambda p:Path(p)):
    value=read(path)
    if value.get('schema')!=CATALOG or task not in TASKS or type(ordinal)is not int or not 1<=ordinal<=30:
        raise ValueError('Saved task and episode ordinal1..30 required')
    rows=[r for r in value['instances']if r['task_name']==task and r['episode_ordinal']==ordinal]
    if len(rows)!=1:raise ValueError('No saved prepared instance for this task/ordinal')
    row=rows[0]
    for source,expected in row['output_hashes'].items():
        if sha(translate(source))!=expected:raise ValueError('Saved preparation changed: '+source)
    return deepcopy(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    value=collect(args.index)
    with args.output.open('x')as out:json.dump(value,out,indent=2);out.write('\n')
    print(json.dumps(dict(counts=value['counts'],all_30_distinct_valid=value['all_30_distinct_valid'])))


if __name__=='__main__':main()
