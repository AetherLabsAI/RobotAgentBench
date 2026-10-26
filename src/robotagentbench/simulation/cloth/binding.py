"""Trusted NativeClothBinding injection, not a global registration change."""
import json
import os
from pathlib import Path
import sys
from . import config


def admit(path,output,*,task_name,options,workspace):
    from robotagentbench.simulation.admission import AdmittedLaunch
    path,output=workspace.input(path),workspace.output(output)
    value=config.validate(config.read(path),workspace,mounted=True)
    if value['task_name']!=task_name:raise ValueError('Exact cloth task required')
    if any(getattr(options,name)for name in('smoke','coding_policy','collision_guard_audit','allow_shared_gpu','allow_idle_utilization')):
        raise ValueError('Formal source owner requires exclusive raw robot22; no diagnostic fallback')
    hashes=value['owner']['hashes']
    pins=tuple((Path(p),h)for p,h in hashes.items())+((path,config.sha(path)),)
    plan=dict(task_name=task_name,execution_mode='native_cloth_scored.v1',available=True,scoring_available=True)
    return AdmittedLaunch(path,output,value,{'task_name':task_name},{'source_hashes':hashes},plan,'native_cloth',pins)


def binding(environment):
    from robotagentbench.simulation.cloth_binding import NativeClothBinding
    def execute(admitted,*,options,workspace):
        environment.probe();admitted.verify_inputs()
        config.validate(admitted.launch,workspace,mounted=True)
        boundary=dict(input_roots=list(map(str,workspace.input_roots)),
            output_root=str(workspace.output_root),lock_root=str(workspace.lock_root))
        argv=[str(environment.launcher),'-I','-S','-B','-c',config.module_bootstrap('supervisor'),
            str(admitted.launch_path),str(admitted.output),str(options.gpu),
            json.dumps(boundary),config.sha(admitted.launch_path)]
        os.execve(environment.launcher,argv,environment.environment(os.environ))
        raise RuntimeError('Fresh cloth handoff unexpectedly returned')
    return NativeClothBinding(admit,execute)
