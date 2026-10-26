"""Task-only CPU candidate execution; reuses unchanged full-source mounts."""
import argparse
from pathlib import Path
import socket
import subprocess
import time
from .runner import command
from .. import sim2_runner as io


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-key', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--stage', choices=('tipped','freeze','mirror','geometry','empty','arm_first'), default='tipped')
    p.add_argument('--variant', choices=('sofa','table_waste_empty_grasp'))
    p.add_argument('--configuration-key')
    a = p.parse_args()
    if io.HERE != io.HOST or socket.gethostname() not in ('VM-0-8-ubuntu','VM-0-10-ubuntu'):
        raise ValueError('Only this exact task on sim1/2')
    if any(not k.replace('_','').isalnum() for k in (a.key,a.base_key)):
        raise ValueError('Simple distinct keys required')
    run = io.HOST/'expansion_v2/commands'/a.key
    run.mkdir(parents=True,exist_ok=False)
    out = io.GUEST/'expansion_v2/candidates'/a.key
    cmd = command('table_waste_tipped_cup', 401, a.key, categories=['cup'])
    ix = cmd.index('-m')
    if a.stage=='tipped':
        module='tipped_initial';arguments=['--base-key',a.base_key,'--output',str(out)]
    else:
        if a.variant is None:raise ValueError('Explicit new empty-grasp variant required')
        arguments=['--base-key',a.base_key,'--variant',a.variant,'--output',str(out)]
        if a.stage in ('freeze','mirror','geometry'):
            module='freeze_directions'
            out=io.GUEST/'expansion_v2/configurations'/(a.key+'.json')
            arguments[-1]=str(out)
            if a.stage=='geometry':
                if a.variant!='table_waste_empty_grasp':raise ValueError('Only the approved natural cup side-grasp calibration')
                module='geometric_direction'
                arguments=['--base-key',a.base_key,'--output',str(out)]
            if a.stage=='mirror':
                if a.variant!='table_waste_empty_grasp':raise ValueError('Only the cup paired-frame calibration')
                k=a.configuration_key
                if not k or not k.replace('_','').isalnum():raise ValueError('Right-only fixed configuration key required')
                module='paired_direction'
                arguments=['--base-key',a.base_key,'--output',str(out),'--configuration',
                    str(io.GUEST/'expansion_v2/configurations'/(k+'.json'))]
        elif a.stage=='empty' and a.variant=='table_waste_empty_grasp':
            if a.configuration_key is not None:
                raise ValueError('Current cup pool is exact-eight training only; omit historical calibration key')
            module='grasp_failure_recovery.frozen_initials_v2.exact8'
            arguments=['--base-key',a.base_key,'--output',str(out)]
        else:
            module='empty_grasp'
            k=a.configuration_key
            if not k or not k.replace('_','').isalnum():raise ValueError('Explicit fixed configuration key required')
            arguments+=['--configuration',str(io.GUEST/'expansion_v2/configurations'/(k+'.json'))]
            if a.stage=='arm_first':
                if a.variant!='table_waste_empty_grasp':raise ValueError('Arm-first placement approved only for the cup')
                module='arm_first_visible_cup_v4'
                arguments=['--base-key',a.base_key,'--output',str(out),'--configuration',
                    str(io.GUEST/'expansion_v2/configurations'/(k+'.json'))]
    qualified=module if module.startswith('grasp_failure_recovery.') else 'grasp_failure_recovery.expansion_v2.'+module
    cmd[ix+1:] = [qualified,*arguments]
    io.save(run/'command.json',dict(command=cmd,started=time.time(),
        phase=a.stage,implementation=module,native_accepted=False,model_called=False))
    with (run/'stdout.log').open('x') as log:
        rc = subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
    io.save(run/'exit.json',dict(returncode=rc,ended=time.time()))
    raise SystemExit(rc)


if __name__ == '__main__':main()
