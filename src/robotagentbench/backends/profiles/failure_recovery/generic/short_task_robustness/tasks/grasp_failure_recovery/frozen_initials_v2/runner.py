"""CPU-only full-source launcher. Training is mounted read-only only for mining."""
import argparse
import socket
import subprocess
import time
from .. import sim2_runner as io
from ..expansion_v2.runner import command

DATASET='/workspace/alienware/baselines/datasets/lerobot-all-atomic-short-20261010-reconcile.staging/revisions/handoff_recovered_20261010_v2/dataset'


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('mine','inspect','compile','batch'))
    p.add_argument('--key',required=True);p.add_argument('--inventory-key');p.add_argument('--base-key',default='tipped_cup_base401_r3')
    p.add_argument('--bank-key');p.add_argument('--seeds',type=int,nargs='+')
    choice=p.add_mutually_exclusive_group();choice.add_argument('--paired',action='store_true');choice.add_argument('--stationed',action='store_true');choice.add_argument('--adapt',action='store_true');a=p.parse_args()
    if a.mode=='batch' and a.bank_key is None:a.bank_key='cup_training_exact8_v6'
    if a.mode=='batch' and a.bank_key!='cup_training_exact8_v6':raise ValueError('Current cup pool is exact-eight training only')
    if io.HERE!=io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):raise ValueError('Only this task on sim1/2')
    for key in (a.key,a.inventory_key,a.base_key,a.bank_key):
        if key and not key.replace('_','').isalnum():raise ValueError('Simple immutable key required')
    run=io.HOST/'frozen_initials_v2/commands'/a.key;run.mkdir(parents=True,exist_ok=False)
    cmd=command('sofa',401,'frozen2_'+a.key,sofa_asset='remote_tv')
    args=['--output',str(io.GUEST/'frozen_initials_v2'/a.mode/a.key)]
    if a.mode=='mine':
        cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')]=['--mount',f'type=bind,src={DATASET},dst=/training_dataset,readonly']
        args+=['--dataset','/training_dataset']
    else:
        args+=['--base-key',a.base_key]
        if a.inventory_key:
            branch,name=('inspect','inspection.json') if a.mode=='compile' else ('mine','inventory.json')
            args+=['--inventory',str(io.GUEST/'frozen_initials_v2'/branch/a.inventory_key/name)]
        if a.bank_key:args+=['--bank',str(io.GUEST/'frozen_initials_v2/compile'/a.bank_key/'bank_v2.json')]
        if a.seeds:args+=['--seeds',*map(str,a.seeds)]
    if (a.paired or a.stationed or a.adapt) and a.mode!='compile':raise ValueError('Offline checks are compile stages only')
    module='adapt' if a.adapt else 'stationed' if a.stationed else 'paired' if a.paired else a.mode
    if a.mode=='batch' and a.bank_key=='cup_training_exact8_v6':module='batch_exact8'
    cmd[cmd.index('-m')+1:]=['grasp_failure_recovery.frozen_initials_v2.'+module,*args]
    start=time.time();io.save(run/'command.json',dict(command=cmd,started=start,source_tree='full_only',gpu_used=False,
        dataset_mounted=a.mode=='mine',training_dataset_readonly=True))
    with (run/'stdout.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    io.save(run/'exit.json',dict(returncode=rc,ended=time.time(),wall_seconds=time.time()-start));raise SystemExit(rc)


if __name__=='__main__':main()
