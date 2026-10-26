"""CPU source preparation on sim1/2, with exact output-only bind isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
from .. import sim2_runner as io


def command(variant, seed, key, *, sofa_asset=None, categories=None, gpu_uuid=None):
    if not key.replace('_', '').isalnum():
        raise ValueError('Simple unique key required')
    arguments = ['--variant', variant, '--seed', str(seed), '--key', key]
    if sofa_asset:
        arguments += ['--sofa-asset', sofa_asset]
    for c in categories or []:
        arguments += ['--waste-category', c]
    cmd = io.command('sofa', 'expansion_v2.prepare_base', arguments, 'expansion_'+key,gpu=gpu_uuid is not None)
    if gpu_uuid is not None:
        cmd[cmd.index('--gpus')+1]='device='+gpu_uuid
        cmd[2:2]=['--hostname',socket.gethostname()]
    # These are output namespaces, not overlaid Python modules or source pins.
    # The full source entry and its original directory/source checks still run.
    root = io.HOST/'expansion_v2'
    mounts = []
    if gpu_uuid is not None:
        for name,target in (('kit_cache','/isaac-sim/kit/cache'),('kit_data','/isaac-sim/kit/data')):
            directory=io.HOST/'execution'/('expansion_'+key)/name
            directory.mkdir(parents=True,exist_ok=True)
            mounts+=['--mount',f'type=bind,src={directory},dst={target}']
    node = root/'tools/node/bin/node'
    receipt = json.loads((root/'tools/node/receipt.json').read_text())
    if node.is_symlink() or hashlib.sha256(node.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Task-owned Node runtime differs from its recorded input')
    mounts += ['--mount', f'type=bind,src={node.parent},dst=/arena_node,readonly',
               '-e', 'PATH=/arena_node:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin']
    for host, guest in (
        (root/'base_outputs', io.CURRENT+'/short_tasks/agent_probe_v1/runs'),
        (root/'generated_rooms', io.CURRENT+'/openend/tasks/scene_requirements/generated_rooms')):
        host.mkdir(parents=True, exist_ok=True)
        if host.is_symlink() or not host.resolve().is_relative_to(io.HOST):
            raise ValueError('Only task-owned output storage may be writable')
        mounts += ['--mount', f'type=bind,src={host},dst={guest}']
    cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')] = mounts
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('sofa', 'table_waste_tipped_cup', 'table_waste_empty_grasp'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--sofa-asset')
    ap.add_argument('--waste-category', action='append')
    a = ap.parse_args()
    if io.HERE != io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu', 'VM-0-10-ubuntu'):
        raise ValueError('Only this exact task directory on sim1/2')
    cmd = command(a.variant, a.seed, a.key, sofa_asset=a.sofa_asset, categories=a.waste_category)
    run = io.HOST/'expansion_v2/commands'/a.key
    run.mkdir(parents=True, exist_ok=False)
    io.save(run/'command.json', dict(command=cmd, started=time.time(), source='full_only',
        output_mounts_only=True, gpu_used=False, model_called=False))
    with (run/'stdout.log').open('x') as log:
        rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
    io.save(run/'exit.json', dict(returncode=rc, ended=time.time()))
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
