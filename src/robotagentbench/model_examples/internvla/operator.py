"""One operator command around existing worker argv and server-tested model loop.

No shell/module path is accepted from observations or model RPC. Worker GPU
admission, original initial acceptance, scoring and delayed close remain owned
by the existing task worker/supervisor named in the operator configuration.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .config import read
from .server_policy import camera_profile,model_config
from .server_transport import prompt_binding,rollout

SCHEMA='robotagentbench.internvla.tested_operator.v1'


def template():
    return dict(schema=SCHEMA,model=dict(python='/absolute/path/to/model-env/bin/python',
        checkpoint='/absolute/path/to/G1/pretrained_model',stats='/absolute/path/to/G1/pretrained_model/stats.json',
        stats_key='G1_omnipicker',control_schema='/absolute/path/to/control_schema.json',
        vlm='/absolute/path/to/local/Qwen3.5-2B'),model_endpoint=None,model_gpu=1,worker_gpu=0,
        lock_root='/absolute/path/to/existing/shared/gpu_locks',output_root=None,max_calls=100,
        tasks={'your_saved_task_id':dict(worker_argv=[],native_report='{output}/native/native_report.json',
            public_prompt=None)},
        lifecycle=dict(worker_owns_native_lease=True,worker_delays_closed_until_cleanup=True,
            external_independent_watchdog_required=True,no_automatic_retry=True))


def configuration(path,task):
    value=read(path)
    if set(value)!=set(template())or value['schema']!=SCHEMA:raise ValueError('Exact tested operator schema required')
    if value['lifecycle']!=template()['lifecycle']:raise ValueError('Preserve existing native owner/lifecycle contract')
    if (type(value['model_gpu'])is not int or type(value['worker_gpu'])is not int
            or min(value['model_gpu'],value['worker_gpu'])<0 or value['model_gpu']==value['worker_gpu']):
        raise ValueError('Explicit different exclusive native/model GPU indices required')
    lock=Path(value['lock_root'])
    if not lock.is_absolute()or not lock.is_dir()or lock.resolve()!=lock:raise ValueError('Existing common GPU lock directory required')
    if type(value['max_calls'])is not int or value['max_calls']<=0:raise ValueError('Explicit model-call bound required')
    if value['output_root']is not None:
        root=Path(value['output_root'])
        if not root.is_absolute()or not root.is_dir()or root.resolve()!=root:raise ValueError('Canonical existing output root required')
    row=value['tasks'][task]
    fields={'worker_argv','native_report','public_prompt'}
    if set(row) not in (fields, fields|{'native_receipt'}):raise ValueError('Trusted worker argv/report/public prompt required')
    if 'native_receipt' in row:
        from robotagentbench.model_examples.native_receipts import FILES
        receipt=row['native_receipt']
        if (not isinstance(receipt,dict) or set(receipt)!={'family','task_id'}
                or receipt['family'] not in FILES or receipt['task_id']!=task):
            raise ValueError('Explicit matching original-owner receipt family/task required')
    camera_profile(row.get('native_receipt'),task_id=task)
    argv=row['worker_argv']
    if not isinstance(argv,list)or not argv or any(not isinstance(x,str)or not x or '\0'in x for x in argv):
        raise ValueError('Configure the actual installed worker/supervisor argv first; no invented worker entry')
    if row['public_prompt']is not None and not isinstance(row['public_prompt'],str):raise ValueError('Only public text may replace the model instruction')
    if value['model_endpoint']is None:_,pins=model_config(value['model'])
    else:
        from .http import endpoint
        endpoint(value['model_endpoint']);pins={}
    return value,row,pins


def model_service(model,prompts,task,row):
    """Pass only the trusted native owner binding, never observed dimensions."""
    camera_profile(row.get('native_receipt'),task_id=task)
    service=dict(model=model,prompts=prompts)
    if 'native_receipt' in row:
        service['native_receipt']=dict(row['native_receipt'])
    return service


def native_cleanup(row,report_path,worker_close,episode_id):
    """Preserve ordinary-owner behavior; verify dedicated original receipts."""
    if 'native_receipt' not in row:
        from robotagentbench.simulation.shutdown import shutdown_succeeded
        native=read(report_path)
        return (shutdown_succeeded(native) and worker_close.get('closed_ack') is True
                and worker_close.get('exit_code')==0),native.get('task_result')
    from robotagentbench.model_examples.native_receipts import collect,verify
    binding=row['native_receipt'];bundle=collect(binding['family'],Path(report_path).parent)
    accepted=verify(binding['family'],binding['task_id'],episode_id,bundle,
        closed_ack=worker_close.get('closed_ack'),worker_exit_code=worker_close.get('exit_code'))
    files=bundle['files']
    native=files.get('native_report.json',files.get('formal_owner_report.json',{}))
    result=next((native[k] for k in ('task_result','closed_result','result','final_result') if k in native),None)
    return accepted,result


class RPC:
    """Original one-line request/reply semantics; owned child only, no retries."""
    def __init__(self,command,log,*,env=None,timeout=600):
        self.timeout=timeout;self.lines=queue.Queue();self.pending=False
        self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            text=True,bufsize=1,env=env,start_new_session=True)
        def drain():
            try:
                while True:
                    line=self.p.stdout.readline(64*1024*1024+1)
                    if not line:break
                    if len(line)>64*1024*1024:raise RuntimeError('Oversized original RPC line')
                    try:value=json.loads(line)
                    except ValueError:continue  # Original server startup messages.
                    self.lines.put(value)
            except BaseException as exc:self.lines.put(exc)
            finally:self.lines.put(EOFError('Owned process closed its original RPC stream'))
        threading.Thread(target=drain,daemon=True).start()

    def read(self):
        value=self.lines.get(timeout=self.timeout)
        if isinstance(value,BaseException):raise value
        self.pending=False
        return value

    def exchange(self,value):
        if self.pending:raise RuntimeError('Uncertain in-flight request; never resend or overlap')
        self.p.stdin.write(json.dumps(value,allow_nan=False)+'\n');self.p.stdin.flush();self.pending=True
        return self.read()

    def close(self):
        if self.p.poll()is not None:return dict(exit_code=self.p.returncode,closed_ack=False)
        if self.pending:
            # EOF is the original owner's cleanup route. Do not queue close
            # behind an uncertain action, and do not signal an unrelated job.
            self.p.stdin.close()
            return dict(exit_code=self.p.wait(timeout=120),closed_ack=False,uncertain_inflight=True)
        reply=self.exchange({'op':'close'});self.p.stdin.close()
        return dict(reply=reply,exit_code=self.p.wait(timeout=120),closed_ack=reply.get('op')=='closed')


def run(config_path,task,output):
    from types import SimpleNamespace
    from robotagentbench.simulation.gpu import acquire_gpu
    value,row,pins=configuration(config_path,task)
    expected_camera_profile=camera_profile(row.get('native_receipt'),task_id=task)
    output=Path(output)
    if not output.is_absolute()or output.resolve()!=output or output.exists()or not output.parent.is_dir():
        raise ValueError('Fresh canonical operator output required; run identity is never reused')
    if value['output_root']is not None:
        root=Path(value['output_root'])
        if output==root or not output.is_relative_to(root):raise ValueError('Operator output must be inside configured Runtime source')
    output.mkdir()
    def write(name,data):
        (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    def expand(text):
        import string
        fields={field for _,field,_,_ in string.Formatter().parse(text)if field is not None}
        if fields-{'output','task','worker_gpu'}:raise ValueError('Only explicit operator argv placeholders allowed')
        return text.format(output=str(output),task=task,worker_gpu=value['worker_gpu'])
    command=[expand(x)for x in row['worker_argv']]
    report_path=Path(expand(row['native_report']))
    if not report_path.is_absolute()or not report_path.is_relative_to(output):raise ValueError('Native report must belong to this fresh output')
    write('intent.json',dict(task=task,worker_argv=command,model_artifacts=pins,
        camera_profile=expected_camera_profile,
        projection_source_sha256='6939b1be0848261afb03829f4c29de57d7ff58d6b3f3d3b69ca7493a5202ad45',
        no_retry=True,model_or_native_acceptance_claimed=False))
    lease=worker=model=None;logs=[];result=dict(status='starting',native_cleanup_verified=False)
    try:
        # Reuse the project's common UUID lock/occupancy implementation. The
        # existing trusted worker separately owns its native GPU lease.
        if value['model_endpoint']is None:
            lease=acquire_gpu(value['model_gpu'],workspace=SimpleNamespace(lock_root=Path(value['lock_root'])))
            # Preserve the original serve_intern default admission as well as
            # the package's common UUID lease. No shared-GPU bypass is exposed.
            fields=subprocess.check_output(['nvidia-smi','--id',lease.uuid,
                '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip().split(',')
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                '--format=csv,noheader'],text=True)
            if (len(fields)!=4 or fields[0].strip()!=lease.uuid or int(fields[1])>32
                    or int(fields[2])<=20000 or int(fields[3])!=0 or lease.uuid in active):
                raise RuntimeError('Original model GPU admission failed; no displacement or sharing')
            write('model_gpu_admission.json',dict(uuid=lease.uuid,memory_used_mib=int(fields[1]),
                memory_free_mib=int(fields[2]),utilization=int(fields[3]),shared=False))
        log=(output/'worker.stderr.log').open('x');logs.append(log);worker=RPC(command,log)
        ready=worker.read()
        if ready.get('op')!='ready':raise ValueError('Original worker did not admit/produce ready')
        write('worker_ready.private.json',ready)
        key,prompt=prompt_binding(task,ready['observation'],row['public_prompt'])
        write('public_prompt_binding.json',dict(task_id=task,public_key=key,instruction=prompt,
            source='worker_public_instruction'if row['public_prompt']is None else'explicit_operator_public_prompt',
            training_task_table_used=False,ground_truth_used=False))
        service=output/'model_service.json';write(service.name,model_service(value['model'],{key:prompt},task,row))
        # No hand-written FD plumbing: this operator connects the two original
        # JSONL processes and preserves full model chunk execution.
        if value['model_endpoint']is None:
            model_command=[value['model']['python'],'-B','-u','-m',
                'robotagentbench.model_examples.internvla','tested-service','--service-config',str(service)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease.uuid,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(output/'model_cache'),
                CUDA_CACHE_PATH=str(output/'model_cache/cuda'),OMP_NUM_THREADS='4')
            write('model_command.json',model_command)
            log=(output/'model.stderr.log').open('x');logs.append(log);model=RPC(model_command,log,env=env)
            model_ready=model.read()
            if model_ready.get('op')!='ready'or model_ready.get('action_shape')!=[50,22]:raise ValueError('Original G1 model handshake differs')
            if model_ready.get('camera_profile')!=expected_camera_profile:
                raise ValueError('Model camera profile differs from trusted native owner binding')
            write('model_ready.json',model_ready)
        else:
            from .http import Client
            class RemoteRPC:
                def __init__(self,url):self.client=Client(url,600)
                def exchange(self,request):return self.client.infer(request)
                def close(self):return dict(external_service=True,service_stopped=False)
            model=RemoteRPC(value['model_endpoint'])
            write('model_endpoint.json',dict(endpoint=value['model_endpoint'],external_service=True,
                model_artifacts_not_verified_locally=True,model_gpu_owned_by_external_service=True))
        result.update(rollout(worker,model,ready,task_id=task,max_calls=value['max_calls'],
            explicit_prompt=row['public_prompt'],
            on_event=lambda name,index,data:write(f'{name}_{index:04d}.json',data)))
        result['worker_close']=worker.close();worker=None
        accepted,native_result=native_cleanup(row,report_path,result['worker_close'],ready['observation']['episode_id'])
        result['native_cleanup_verified']=accepted
        result.update(status='completed'if result['native_cleanup_verified']else'invalid_native_cleanup',
            native_result=native_result,full_model_behavior_parity_claim=False)
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__)
        raise
    finally:
        for name,rpc in [('worker',worker),('model',model)]:
            if rpc is not None:
                try:result[name+'_cleanup']=rpc.close()
                except BaseException as exc:result[name+'_cleanup_error']=type(exc).__name__
        # Do not release the model lock if that process is still alive: retain
        # this process until the external independent watchdog settles it.
        if isinstance(model,RPC)and model.p.poll()is None:
            result['model_cleanup_uncertain']=True;write('report.json',result)
            model.p.wait()
        cleanup=result.get('model_cleanup',{})
        if ('model_cleanup_error'in result or (cleanup and not cleanup.get('external_service')
                and (cleanup.get('exit_code')!=0 or cleanup.get('closed_ack')is not True))):
            result['status']='model_cleanup_not_accepted'
        if lease is not None:lease.close()
        for log in logs:log.close()
        write('report.json',result)
    return result
