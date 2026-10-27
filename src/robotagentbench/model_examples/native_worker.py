"""Thin raw-action entry for an existing prepared room launch.

This is the public example's original source/environment recipe with the saved
launch made explicit. No task factory, scorer, controller or simulator setting
is implemented here. Dedicated cloth/specialized owners retain their own entry.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def configuration(runtime_path,launch,output,lock_root,gpu,*,input_roots=(),public_input_binding=None):
    from robotagentbench.example import runtime
    base,root,value=runtime(runtime_path)
    launch=Path(launch);output=Path(output);lock=Path(lock_root)
    if type(gpu)is not int or gpu<0:raise ValueError('Explicit exclusive GPU index required')
    inputs=tuple(map(Path,input_roots)) or (root,)
    if (len(inputs)!=len(set(inputs)) or any(not p.is_absolute() or p.resolve()!=p or not p.is_dir() for p in inputs)
            or launch.resolve(strict=True)!=launch or not launch.is_file()
            or not any(launch.is_relative_to(p) for p in inputs)):
        raise ValueError('Use an unchanged prepared launch inside explicit trusted read-only input roots')
    if output.resolve()!=output or output.exists() or not output.is_relative_to(root) or output==root:
        raise ValueError('Fresh native output inside the configured source workspace required')
    if lock.resolve(strict=True)!=lock or not lock.is_dir():raise ValueError('Existing shared host lock directory required')
    data=json.loads(launch.read_text())
    if data.get('schema')not in ('openend.short_task.launch.v1','openend.coding_room.launch.v1'):
        raise ValueError('This entry handles original room/short raw-action owners only')
    if public_input_binding is not None:
        if data['schema']!='openend.coding_room.launch.v1':raise ValueError('Saved public text requires the room owner')
        from robotagentbench.simulation.public_input import load
        from robotagentbench.simulation.workspace import NativeWorkspace
        load(public_input_binding,launch,workspace=NativeWorkspace(inputs,root,lock))
    return base,root,value


def execute(runtime_path,sdk_site,launch,output,lock_root,gpu,*,input_roots=(),public_input_binding=None):
    from robotagentbench.example import runtime,sdk_paths,verify_native_profile,bind_collision_library
    base,root,value=configuration(runtime_path,launch,output,lock_root,gpu,input_roots=input_roots,
                                  public_input_binding=public_input_binding)
    site,usd,physx=sdk_paths(sdk_site)
    # Establish the same original native-library loader before any SDK import.
    marker='ROBOTAGENTBENCH_MODEL_EXAMPLE_NATIVE_HANDOFF'
    if os.environ.get(marker)!=str(site):
        env=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(key,None)
        env.update({marker:str(site),'PYTHONDONTWRITEBYTECODE':'1',
            'LD_LIBRARY_PATH':str(usd/'bin')+os.pathsep+str(physx/'bin'),
            'PXR_AR_DEFAULT_SEARCH_PATH':str(site/'isaacsim/kit/mdl/core/Base')})
        package=str(Path(__file__).resolve().parents[2])
        code=('import runpy,sys;sys.path.insert(0,'+repr(package)+');'
              'sys.argv=["model-example-native-worker",*sys.argv[1:]];'
              'runpy.run_module("robotagentbench.model_examples.native_worker",run_name="__main__")')
        args=['--runtime',str(base),'--sdk-site',str(site),'--launch',str(launch),
              '--output',str(output),'--lock-root',str(lock_root),'--gpu',str(gpu),'--execute']
        for item in input_roots:args+=['--input-root',str(item)]
        if public_input_binding is not None:args+=['--public-input-binding',str(public_input_binding)]
        os.execve(sys.executable,[sys.executable,'-I','-S','-B','-c',code,*args],env)
        raise RuntimeError('Dedicated native handoff unexpectedly returned')
    if sys.version_info[:2]!=(3,11):raise ValueError('The original native profile requires CPython3.11')
    sys.path[:0]=[str(usd),str(physx),str(site)]
    os.environ['PYTHONPATH']=os.pathsep.join(map(str,(usd,physx,site,root)))
    verify_native_profile(site)
    pins=dict(value['source_files']);native=bind_collision_library(root,site,pins)
    from robotagentbench.simulation.source_tree import NativeSourceTree
    from robotagentbench.simulation.original_backend import run_original_worker
    from robotagentbench.simulation.worker import NativeRunOptions
    from robotagentbench.simulation.workspace import NativeWorkspace
    sources=NativeSourceTree(root,pins,groups={'native':native},
        fcl_namespace='short_tasks/agent_probe_v1/collision_guard/vendor/fcl')
    workspace=NativeWorkspace(tuple(map(Path,input_roots)) or (root,),root,Path(lock_root))
    extra={}if public_input_binding is None else dict(public_input_binding=public_input_binding)
    return run_original_worker(launch,output,options=NativeRunOptions(gpu),sources=sources,workspace=workspace,**extra)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--sdk-site',type=Path,required=True)
    p.add_argument('--launch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock-root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--input-root',type=Path,action='append',default=[],help='Explicit trusted saved-input root, mounted read-only by the operator')
    p.add_argument('--public-input-binding',type=Path,help='Trusted saved room instruction/context binding; never a model/RPC argument')
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    configuration(a.runtime,a.launch,a.output,a.lock_root,a.gpu,input_roots=a.input_root,
                  public_input_binding=a.public_input_binding)
    if not a.execute:
        print(json.dumps(dict(mode='describe',launch=str(a.launch),output=str(a.output),
            native_started=False,model_called=False,original_task_owner=True,
            specialized_cloth_or_coding_owner_bound=False),indent=2));return 0
    extra={}if a.public_input_binding is None else dict(public_input_binding=a.public_input_binding)
    return execute(a.runtime,a.sdk_site,a.launch,a.output,a.lock_root,a.gpu,input_roots=a.input_root,**extra)


if __name__=='__main__':raise SystemExit(main())
