"""Explicit task-local entry into the existing original coding-policy worker.

Run inside the operator's already configured native namespace. No simulator,
Docker, or model starts when importing or describing this entry.
"""
import argparse,ast,hashlib,json,os,runpy,sys
from pathlib import Path

POLICY_IMAGE='sha256:fa7785fb56f0d87ef35f4eb786a56c4d246e30e413f65fa3e7edd3d82c3dfea6'
SERVER_POLICY_IMAGE='sha256:4ccf7e950d48f9b5477292bbc123b676fdbad039c6d51573d969008d8ffd5dc6'
CODING_PREFIX='short_tasks/agent_probe_v1/cap_v1/native_tools/'

def selected_image(root,pins):
    relative=CODING_PREFIX+'sandbox_launcher.py';raw=(root/relative).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pins.get(relative):raise ValueError('Selected original sandbox launcher identity differs')
    assignments=[n for n in ast.parse(raw).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='IMAGE'for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Original policy image declaration must be explicit')
    image=ast.literal_eval(assignments[0].value)
    if image not in (POLICY_IMAGE,SERVER_POLICY_IMAGE):raise ValueError('Unreviewed policy-image recipe')
    return image

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    value=json.loads(a.config.read_text())
    required={'schema','runtime','sdk_site','launch','output','lock_root','gpu','coding_files','policy_image','docker_host'}
    if set(value)-{'input_roots','public_input_binding'}!=required or value['schema']!='robotagentbench.codex_worker.v1':raise ValueError('Explicit trusted coding worker configuration required')
    from robotagentbench.example import runtime
    _,root,config=runtime(value['runtime'])
    image=selected_image(root,config['source_files'])
    if value['policy_image']!=image:raise ValueError('Configured image differs from selected original launcher; no fallback')
    from .server_profile import FILES
    if image==SERVER_POLICY_IMAGE:
        for relative,expected in FILES.items():
            if config['source_files'].get(CODING_PREFIX+relative)!=expected:
                raise ValueError('Original server coding starter/launcher closure is not bound: '+relative)
    if not isinstance(value['docker_host'],str)or not value['docker_host'].startswith('unix://'):
        raise ValueError('Explicit local operator Docker endpoint required')
    if type(value['gpu'])is not int or value['gpu']<0:raise ValueError('Explicit exclusive GPU required')
    if value.get('public_input_binding')is not None:
        from robotagentbench.simulation.public_input import load
        from robotagentbench.simulation.workspace import NativeWorkspace
        launch=Path(value['launch'])
        if json.loads(launch.read_text()).get('schema')!='openend.coding_room.launch.v1':
            raise ValueError('Saved public text requires the room owner')
        inputs=tuple(Path(item)for item in value.get('input_roots',[str(root)]))
        load(value['public_input_binding'],launch,
             workspace=NativeWorkspace(inputs,root,Path(value['lock_root'])))
    if not a.execute:
        print(json.dumps(dict(entry='original coding worker',execute=False,policy_image=image,
            native_started=False,requires='original source pins, resources, namespace, SDK and exclusive GPU')));return 0
    from robotagentbench.example import runtime,sdk_paths,verify_native_profile,bind_collision_library
    site,usd,physx=sdk_paths(value['sdk_site'])
    marker='ROBOTAGENTBENCH_CODEX_NATIVE_HANDOFF'
    if os.environ.get(marker)!=str(site):
        env=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(key,None)
        env.update({marker:str(site),'PYTHONDONTWRITEBYTECODE':'1',
            'LD_LIBRARY_PATH':str(usd/'bin')+os.pathsep+str(physx/'bin'),
            'PXR_AR_DEFAULT_SEARCH_PATH':str(site/'isaacsim/kit/mdl/core/Base')})
        package=str(Path(__file__).resolve().parents[3])
        code='import runpy,sys;sys.path.insert(0,'+repr(package)+');sys.argv=["codex-worker",*sys.argv[1:]];runpy.run_module("robotagentbench.model_examples.codex.worker",run_name="__main__")'
        os.execve(sys.executable,[sys.executable,'-I','-S','-B','-c',code,str(a.config.resolve()),'--execute'],env)
    sys.path[:0]=[str(usd),str(physx),str(site)]
    verify_native_profile(site)
    base,root,config=runtime(value['runtime']);pins=dict(config['source_files'])
    selected=value['coding_files']
    if not isinstance(selected,list)or len(selected)!=len(set(selected))or not selected:
        raise ValueError('Explicit original coding dependency group required')
    required_sources={key for key in pins if key.startswith(CODING_PREFIX)}
    if not required_sources or not required_sources.issubset(selected)or any(key not in pins for key in selected):
        raise ValueError('Coding group must include the complete pinned original broker/starter closure')
    os.environ['PYTHONPATH']=os.pathsep.join(map(str,(root,root/'api-protocol',usd,physx,site)))
    os.environ['DOCKER_HOST']=value['docker_host']
    native=bind_collision_library(root,site,pins)
    from robotagentbench.simulation.source_tree import NativeSourceTree
    from robotagentbench.simulation.workspace import NativeWorkspace
    from robotagentbench.simulation.worker import NativeRunOptions
    from robotagentbench.simulation.original_backend import run_original_worker
    source=NativeSourceTree(root,pins,groups={'native':native,'coding':selected},
        fcl_namespace='short_tasks/agent_probe_v1/collision_guard/vendor/fcl')
    output=Path(value['output']).resolve();launch=Path(value['launch']).resolve(strict=True)
    if output.exists():raise ValueError('Native output already exists; never redispatch')
    sandbox_root=root/'short_tasks/agent_probe_v1/cap_v1'
    if not output.is_relative_to(sandbox_root)or output==sandbox_root:
        raise ValueError('Original coding sandbox/output must remain below its cap_v1 owner')
    output.parent.mkdir(parents=True,exist_ok=True)
    inputs=value.get('input_roots',[str(root)])
    if not isinstance(inputs,list)or not inputs or len(inputs)!=len(set(inputs)):
        raise ValueError('Explicit unique read-only prepared input roots required')
    input_roots=tuple(Path(item)for item in inputs)
    if any(not p.is_absolute()or p.resolve()!=p or not p.is_dir()for p in input_roots):
        raise ValueError('Canonical existing prepared input roots required')
    workspace=NativeWorkspace(input_roots,output.parent,Path(value['lock_root']).resolve(strict=True))
    extra={}if value.get('public_input_binding')is None else dict(public_input_binding=value['public_input_binding'])
    return run_original_worker(launch,output,options=NativeRunOptions(value['gpu'],coding_policy=True),
        sources=source,workspace=workspace,enable_coding=True,**extra)

if __name__=='__main__':raise SystemExit(main())
