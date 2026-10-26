"""Prepare and run the included coffee example using a local room workspace."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from .backends import sources as embedded
from .simulation.source_tree import NativeSourceTree


def runtime(path):
    base=Path(path).resolve(strict=True)
    value=json.loads((base/'runtime.json').read_text())
    if value.get('schema')!='robotagentbench.local_installation.v1' or value.get('profile')!='room':
        raise ValueError('The coffee example requires a configured room workspace')
    old=Path(value['runtime'])
    relative=Path(value['source_root']).relative_to(old)
    root=base/relative
    if root.resolve()!=root or not root.is_dir():raise ValueError('Canonical local source directory required')
    return base,root,value


def sdk_paths(site):
    site=Path(site).resolve(strict=True)
    cache=site/'isaacsim/extscache'
    usd=list(cache.glob('omni.usd.libs-*'));physx=list(cache.glob('omni.usd.schema.physx-*'))
    if len(usd)!=1 or len(physx)!=1:raise ValueError('A single Isaac Sim 5.1 USD/PhysX library profile is required')
    return site,usd[0],physx[0]


def native_process(site,argv):
    site,usd,physx=sdk_paths(site)
    marker='ROBOTAGENTBENCH_EXAMPLE_LIBRARY_HANDOFF'
    if os.environ.get(marker)==str(site):return
    # Establish native loader paths before the fresh interpreter loads a shared
    # library. No simulator, scene, scoring or control parameters are set here.
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):
        env.pop(key,None)
    env.update({marker:str(site),'PYTHONDONTWRITEBYTECODE':'1',
        'LD_LIBRARY_PATH':str(usd/'bin')+os.pathsep+str(physx/'bin'),
        'PXR_AR_DEFAULT_SEARCH_PATH':str(site/'isaacsim/kit/mdl/core/Base')})
    code=('import runpy,sys;sys.path.insert(0,'+repr(str(Path(__file__).resolve().parent.parent))+');'
          'sys.argv=["robotagentbench.example",*sys.argv[1:]];'
          'runpy.run_module("robotagentbench.example",run_name="__main__")')
    os.execve(sys.executable,[sys.executable,'-I','-S','-B','-c',code,*argv],env)


def bind_collision_library(root,site,pins):
    specification=json.loads((embedded.HERE/'native-files.json').read_text())
    names=[]
    for relative,row in specification['files'].items():
        source=site/row['site_relative']
        if source.resolve()!=source or not source.is_file():raise ValueError('Required python-fcl file is absent')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Collision library differs from the pinned Linux/CPython 3.11 distribution')
        destination=root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Existing collision library changed')
        else:shutil.copyfile(source,destination)
        pins[relative]=row['sha256'];names.append(relative)
    return names


def verify_native_profile(site):
    specification=json.loads((embedded.HERE/'native-files.json').read_text())
    if (list(sys.version_info[:2])!=specification['python'] or sys.platform!=specification['platform']
            or platform.machine()!=specification['machine']):
        raise ValueError('Native interpreter differs from the selected platform profile')
    versions={}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name=distribution.metadata['Name'].lower().replace('_','-').replace('.','-')
        if name in specification['packages']:
            if name in versions:raise ValueError('Ambiguous native package metadata')
            versions[name]=distribution.version
    if versions!=specification['packages']:
        raise ValueError('Isaac Sim and native package versions differ from the selected profile')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--name',help='Episode directory name; required for prepare/run')
    parser.add_argument('--sdk-site',type=Path,help='Isaac Sim Python site-packages directory; prepare/run only')
    parser.add_argument('--gpu',type=int,help='Exclusive GPU index; run only')
    parser.add_argument('--lock-root',type=Path,help='Shared host GPU-lock directory mounted into the job; run only')
    args=parser.parse_args(argv)
    if not args.plan and (not args.name or not re.fullmatch('[A-Za-z0-9_-]{1,80}',args.name)):
        parser.error('prepare/run requires a simple episode --name')
    if not args.plan and not args.sdk_site:parser.error('prepare/run requires --sdk-site')
    if args.run and (args.gpu is None or args.gpu<0 or args.lock_root is None):
        parser.error('run requires an exclusive --gpu and shared --lock-root')
    if not args.plan:
        native_process(args.sdk_site,list(argv) if argv is not None else sys.argv[1:])
    base,root,config=runtime(args.runtime)
    pins=dict(config['source_files'])
    if not args.plan:
        site,usd,physx=sdk_paths(args.sdk_site)
        sys.path[:0]=[str(usd),str(physx),str(site)]
        os.environ['PYTHONPATH']=os.pathsep.join(map(str,(usd,physx,site,root)))
        if sys.version_info[:2]!=(3,11):raise ValueError('Native example requires CPython 3.11')
        verify_native_profile(site)
    native_group=bind_collision_library(root,site,pins)if args.run else None
    source=NativeSourceTree(root,pins,groups={'native':native_group}if args.run else None,
        fcl_namespace='short_tasks/agent_probe_v1/collision_guard/vendor/fcl'if args.run else None)
    output=root/'short_tasks/agent_probe_v1/runs'/args.name if args.name else None
    if args.run:
        from .simulation.original_backend import run_original_worker
        from .simulation.worker import NativeRunOptions
        from .simulation.workspace import NativeWorkspace
        preparation=json.loads((output/'preparation.json').read_text())
        launch=Path(preparation['launch'])
        if not launch.is_relative_to(output) or not launch.is_file():
            raise ValueError('Prepared example launch is missing or outside its episode')
        workspace=NativeWorkspace((root,),output,args.lock_root.resolve(strict=True))
        return run_original_worker(launch,output/'evaluation',options=NativeRunOptions(args.gpu),sources=source,workspace=workspace)
    with source.activated():
        request=json.loads((root/'short_tasks/tasks/prepare_coffee/request_v2.json').read_text())
        registry=source.import_module('short_tasks.registry')
        plan=registry.resolve(request);registry.verify_sources(plan)
        if args.plan:
            result=source.import_module('short_tasks.scene').plan_scene(request)
            print(json.dumps(dict(task=plan['task_name'],instruction=plan['instruction'],
                score_policy=plan['score_policy'],layout=result.get('actual_layout_id'),
                scene_status=result['scene_status']),indent=2))
            return 0
        if output.exists():raise ValueError('Episode already exists; choose a fresh name')
        sensor=base/'example-sensor.json'
        original=(Path(__file__).resolve().parent/'example_data/sensor.json').read_bytes()
        if sensor.exists():
            if sensor.read_bytes()!=original:raise ValueError('Example sensor calibration changed')
        else:
            with sensor.open('xb')as stream:stream.write(original)
        ready=root/'layout_task_activation_v1/task_scene_registration_v1/recorded_ready_pose_042_v1.json'
        prepare=source.import_module('short_tasks.agent_probe_v1.prepare_task').prepare
        result=prepare(request,ready,sensor,output)
        print(json.dumps(dict(task=plan['task_name'],launch=result['launch']),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
