"""Resolve an existing cloth instance without generating or resampling it."""
import argparse
import json
from pathlib import Path

from . import randomization as reset


def load(index,seed):
    reset._seed(seed)
    path=Path(index).resolve(strict=True)
    value=json.loads(path.read_text())
    if value.get('schema')!='robotagentbench.cloth_instance_bank.v1':
        raise ValueError('Prepared cloth index required')
    rows=[r for r in value['instances'] if r['seed']==seed]
    if len(rows)!=1:raise ValueError('Seed is absent or duplicated in the prepared bank')
    row=rows[0];launch=Path(row['launch'])
    if not launch.is_absolute() or launch.resolve(strict=True)!=launch or not launch.is_relative_to(path.parent):
        raise ValueError('Instance launch must remain inside its immutable bank')
    if reset.digest(launch)!=row['launch_sha256']:raise ValueError('Saved instance launch changed')
    envelope=json.loads(launch.read_text());sidecar=envelope['variation']
    record=Path(sidecar['record']);initial=Path(sidecar['initial'])
    for source in (record,initial):
        if source.resolve(strict=True)!=source or source.parent!=launch.parent:
            raise ValueError('Only the saved per-instance reset sidecars are allowed')
    if reset.digest(record)!=sidecar['record_sha256'] or reset.digest(initial)!=sidecar['initial_sha256']:
        raise ValueError('Saved cloth instance changed')
    data=reset.verify_record(json.loads(record.read_text()),task=value['task_name'])
    if data['seed']!=seed or envelope['task_name']!=value['task_name']:
        raise ValueError('Prepared instance identity mismatch')
    if sidecar['initial_sha256']!=row['initial_sha256'] or data['initial_sha256']!=row['initial_sha256']:
        raise ValueError('Prepared physical initial-state identity mismatch')
    return launch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index',type=Path);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(dict(launch=str(load(args.index,args.seed)),resampled=False,native_started=False)))


if __name__=='__main__':main()
