"""Read-only deployment checks and fresh operator configuration; no task sampling."""
import json
from pathlib import Path

from .config import sha
from .operator import template
from .server_policy import model_config
from .server_transport import validate_task_id


def worker_argv(native_python,runtime,sdk_site,launch,lock_root,*,input_roots=(),public_input_binding=None):
    argv = [str(native_python),'-B','-u','-m','robotagentbench.model_examples.native_worker',
        '--runtime',str(runtime),'--sdk-site',str(sdk_site),'--launch',str(launch),
        '--output','{output}/native','--lock-root',str(lock_root),'--gpu','{worker_gpu}','--execute']
    for root in input_roots:argv += ['--input-root',str(root)]
    if public_input_binding is not None:argv += ['--public-input-binding',str(public_input_binding)]
    return argv


def build(*,runtime,sdk_site,launch,native_python,model_python,checkpoint,stats,
          stats_key,control_schema,vlm,lock_root,worker_gpu,model_gpu,task,output,max_calls=100,
          input_roots=(),public_input_binding=None):
    from robotagentbench.example import sdk_paths
    from robotagentbench.model_examples.native_worker import configuration
    validate_task_id(task)  # Public routing label, not a training/answer index.
    if type(model_gpu)is not int or model_gpu<0 or model_gpu==worker_gpu:
        raise ValueError('Distinct exclusive native/model GPU indices required')
    if type(max_calls)is not int or max_calls<=0:raise ValueError('Positive model-call bound required')
    native_python=Path(native_python)
    if not native_python.is_absolute()or not native_python.is_file():raise ValueError('Existing native Python required')
    output=Path(output)
    if not output.is_absolute()or output.resolve()!=output or output.exists()or not output.parent.is_dir():
        raise ValueError('Choose a fresh canonical operator output with an existing parent')
    input_roots=tuple(map(Path,input_roots))
    base,root,_=configuration(runtime,Path(launch),output/'native',Path(lock_root),worker_gpu,
        input_roots=input_roots,public_input_binding=public_input_binding)
    if output==root or not output.is_relative_to(root):raise ValueError('Operator output must be inside Runtime source')
    site,_,_=sdk_paths(sdk_site)  # Directory inventory only; never imports SDK/App.
    model,pins=model_config(dict(python=str(model_python),checkpoint=str(checkpoint),stats=str(stats),
        stats_key=stats_key,control_schema=str(control_schema),vlm=str(vlm)))
    value=template();value.update(model=model,worker_gpu=worker_gpu,model_gpu=model_gpu,
        lock_root=str(Path(lock_root)),output_root=str(root),max_calls=max_calls,
        tasks={task:dict(worker_argv=worker_argv(native_python,base,site,launch,lock_root,
            input_roots=input_roots,public_input_binding=public_input_binding),
            native_report='{output}/native/native_report.json',public_prompt=None)})
    receipt=dict(schema='robotagentbench.internvla.configuration_receipt.v1',task=task,
        runtime=str(base),source_root=str(root),suggested_output=str(output),
        launch=str(launch),launch_sha256=sha(launch),runtime_sha256=sha(base/'runtime.json'),
        model_artifacts_sha256=pins,native_started=False,model_loaded=False,
        sdk_imported=False,task_sampled=False,real_model_wrapper_verified=False,
        public_input_binding=str(public_input_binding)if public_input_binding is not None else None)
    return value,receipt


def save(destination,**kwargs):
    value,receipt=build(**kwargs)
    destination=Path(destination).absolute()
    proof=destination.with_name(destination.name+'.receipt.json')
    if not destination.parent.is_dir()or destination.exists()or proof.exists():
        raise ValueError('Fresh configuration and receipt paths required')
    with destination.open('x')as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    receipt['configuration_sha256']=sha(destination)
    with proof.open('x')as stream:stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(config=str(destination),receipt=str(proof),**receipt)
