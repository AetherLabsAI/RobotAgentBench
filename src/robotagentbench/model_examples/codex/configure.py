"""Write explicit operator/worker JSON from a configured Runtime; no execution."""
import argparse,hashlib,json,sys
from pathlib import Path
from .worker import CODING_PREFIX,selected_image

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','sdk-site','launch','lock-root','native-output','agent-output','destination','codex-executable'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--native-python',default=sys.executable);p.add_argument('--gpu',required=True,type=int)
    p.add_argument('--docker-host',required=True);p.add_argument('--cli-version',required=True)
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--effort',default='medium')
    p.add_argument('--max-model-responses',type=int,default=200)
    p.add_argument('--input-root',type=Path,action='append',default=[],help='Explicit read-only saved-launch input root; otherwise Runtime source')
    p.add_argument('--public-input-binding',type=Path,help='Hash-bound saved public room instruction/context')
    a=p.parse_args(argv)
    from robotagentbench.example import runtime
    base,root,value=runtime(a.runtime)
    image=selected_image(root,value['source_files'])
    if value.get('coding_policy_image')!=image:
        raise ValueError('Stage this Runtime with the user-built sandbox image first')
    from .sandbox_build import inspect_image
    if inspect_image(image,a.docker_host)!=image:raise ValueError('Staged local sandbox image differs')
    from .server_profile import deployment_payloads
    for relative,raw in deployment_payloads(image).items():
        if value['source_files'].get(relative)!=hashlib.sha256(raw).hexdigest():
            raise ValueError('Reviewed server starter/launcher source is not bound: '+relative)
    coding=[name for name in value['source_files']if name.startswith(CODING_PREFIX)]
    if not coding:raise ValueError('No original reviewed coding sources in Runtime')
    required=('broker.py','sandbox.py','sandbox_launcher.py','sandbox_boot.py','contract.py','starter/planned_motion.py','starter/robot/robot_contract.json')
    if any(CODING_PREFIX+name not in coding for name in required):raise ValueError('Original coding source/starter closure is incomplete')
    inputs=[path.resolve(strict=True)for path in (a.input_root or [root])]
    public_binding=None
    if a.public_input_binding is not None:
        from robotagentbench.simulation.public_input import load
        from robotagentbench.simulation.workspace import NativeWorkspace
        public_binding=a.public_input_binding.resolve(strict=True)
        load(public_binding,a.launch.resolve(strict=True),
             workspace=NativeWorkspace(tuple(inputs),root,a.lock_root.resolve(strict=True)))
    destination=a.destination.resolve();destination.mkdir(parents=True,exist_ok=False)
    worker=dict(schema='robotagentbench.codex_worker.v1',runtime=str(base),sdk_site=str(a.sdk_site.resolve(strict=True)),
        launch=str(a.launch.resolve(strict=True)),output=str(a.native_output.resolve()),lock_root=str(a.lock_root.resolve(strict=True)),
        gpu=a.gpu,coding_files=coding,policy_image=image,docker_host=a.docker_host,
        input_roots=list(map(str,inputs)))
    if public_binding is not None:worker['public_input_binding']=str(public_binding)
    path=destination/'worker.json'
    operator=dict(schema='robotagentbench.codex_example.v1',codex_executable=str(a.codex_executable.resolve(strict=True)),
        model=a.model,effort=a.effort,max_model_responses=a.max_model_responses,transport=None,
        expected_cli_version=a.cli_version,prepare_argv=[],
        worker_argv=[a.native_python,'-m','robotagentbench.model_examples.codex.worker',str(path),'--execute'],
        output=str(a.agent_output.resolve()))
    for name,data in (('worker.json',worker),('operator.json',operator)):
        with (destination/name).open('x')as stream:json.dump(data,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(operator=str(destination/'operator.json'),worker=str(path),model_started=False,native_started=False)))
    return 0

if __name__=='__main__':raise SystemExit(main())
