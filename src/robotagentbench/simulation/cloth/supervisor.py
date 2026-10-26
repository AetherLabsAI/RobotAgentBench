"""Non-physical owner of the common GPU lease and delayed closed acknowledgement.

The original child may fast-exit during app.close(). This supervisor never
imports Isaac; it waits for actual child exit and the saved v18 cleanup record.
It forwards action bytes once, without retries or diagnostic action generation.
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

MAX_REPLY=64*1024*1024  # Above three source PNG caps plus base64/JSON overhead.


def closure_accepted(report,returncode,*,closed_reply,task_name):
    final=report.get('final_result',{})
    return bool(returncode==0 and closed_reply
        and report.get('status')=='source_native_session_closed'
        and report.get('task_name')==task_name
        and report.get('native_session_transport_accepted')is True
        and report.get('native_shutdown_complete')is True
        and report.get('render_process_shutdown_complete')is True
        and not any(k in report for k in('error','cleanup_error','native_stop_error','observation_setup_error'))
        and final.get('status')=='finished' and final.get('evaluation_valid')is True
        and final.get('scoring_available')is True and final.get('task_name')==task_name
        and final.get('episode_id')==report.get('episode_id'))


def child_environment(base,uuid,output):
    """Select the admitted device/cache only; retain original CPU/thread settings."""
    result=dict(base);out=Path(output)
    result.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HOME=str(out/'home'),XDG_CONFIG_HOME=str(out/'config'),XDG_DATA_HOME=str(out/'data'),
        XDG_CACHE_HOME=str(out/'cache'),CUDA_CACHE_PATH=str(out/'cache/cuda'),PYTHONDONTWRITEBYTECODE='1')
    return result


def forward_requests(source,destination,stop,errors):
    try:
        while not stop.is_set():
            # Same common serve request bound. Oversized input is forwarded
            # unchanged and rejected by the original common host, not clipped.
            line=source.readline(1024*1024+1)
            if not line:break
            raw=line.encode()if isinstance(line,str)else line
            remaining=memoryview(raw)
            while remaining and not stop.is_set():
                size=os.write(destination.fileno(),remaining);remaining=remaining[size:]
            try:closed=json.loads(line).get('op')=='close'
            except (ValueError,AttributeError):closed=False
            if closed:break
    except (OSError,ValueError)as exc:
        if not stop.is_set():errors.append(type(exc).__name__)
    finally:
        try:destination.close()
        except OSError:pass


def finish_child(child,replies):
    """EOF and wait, even if pipe draining fails; never signal/retry actions."""
    errors=[]
    try:child.stdin.close()
    except (OSError,ValueError):pass
    if child.poll()is None:
        try:
            drain=replies if replies is not None else child.stdout
            while drain.read(65536):pass
        except (OSError,ValueError)as exc:
            errors.append('drain:'+type(exc).__name__)
        finally:
            # Lease release is downstream of this actual wait. The existing
            # outer independent-job watchdog handles a native child hang.
            child.wait()
    return errors


def main():
    launch,output,gpu,boundary_text,expected=sys.argv[1:]
    from . import config
    path=Path(launch)
    if config.sha(path)!=expected:raise ValueError('Prepared cloth launch changed')
    value=config.read(path);owner=value['owner'];roots=owner['directories']
    sys.path.insert(0,roots['public_root'])
    from robotagentbench.simulation.workspace import NativeWorkspace
    from robotagentbench.simulation.gpu import acquire_gpu
    from robotagentbench.runtime.completion import CompletionStream
    boundary=json.loads(boundary_text)
    workspace=NativeWorkspace(tuple(map(Path,boundary['input_roots'])),Path(boundary['output_root']),Path(boundary['lock_root']))
    config.validate(value,workspace,mounted=True)
    out=workspace.output(output)
    lease=acquire_gpu(int(gpu),workspace=workspace)
    child=stderr=replies=None;stream=None;code=1;stop=threading.Event();input_errors=[]
    report=dict(schema='robotagentbench.formal_cloth_owner.v1',task_name=value['task_name'],
        status='starting',native_started=False,closed_ack_sent=False,lease_released=False,
        original_source_owner=True,source_scoring_modified=False,physical_parameters_modified=False)
    try:
        config.validate(value,workspace,mounted=True)
        workspace.create_output(out)
        for name in('home','logs','cache','data','config'):(out/name).mkdir()
        stream=CompletionStream(sys.stdout)
        stderr=(out/'physical_stderr.log').open('wb')
        with(out/'dispatch_intent.json').open('x')as intent:
            json.dump(dict(launch_sha256=expected,output=str(out),task_name=value['task_name'],
                automatic_retry=False,uncertain_actions_must_not_be_replayed=True),intent)
        child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',config.module_bootstrap('physical'),str(path),str(out),
            boundary_text,expected],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,bufsize=0,
            env=child_environment(os.environ,lease.uuid,out))
        report.update(native_started=True,physical_pid=child.pid)
        (out/'formal_owner_report.json').write_text(json.dumps(report,indent=2)+'\n')
        feeder=threading.Thread(target=forward_requests,args=(sys.stdin,child.stdin,stop,input_errors),daemon=True)
        feeder.start()
        replies=io.BufferedReader(child.stdout)
        while True:
            line=replies.readline(MAX_REPLY+1)
            if not line:break
            if len(line)>MAX_REPLY or not line.endswith(b'\n'):raise RuntimeError('Invalid bounded native reply')
            text=line.decode('utf-8');json.loads(text)
            if stream.pending is not None:raise RuntimeError('Reply after terminal close candidate')
            stream.write(text);stream.flush()
        returncode=child.wait()
        owner_path=out/'source_owner_report.json'
        native=config.read(owner_path)if owner_path.is_file()else{}
        accepted=closure_accepted(native,returncode,closed_reply=stream.pending is not None,task_name=value['task_name'])and not input_errors
        report.update(status='native_session_closed'if accepted else'native_or_transport_not_accepted',
            physical_returncode=returncode,v18_owner_status=native.get('status'),
            native_shutdown_complete=native.get('native_shutdown_complete'),
            render_process_shutdown_complete=native.get('render_process_shutdown_complete'),
            input_forward_errors=input_errors,final_result=native.get('final_result'),
            native_cleanup_accepted=accepted,full_task_success_claim=False)
        code=0 if accepted else 1
    except BaseException as exc:
        report.update(status='error',error=type(exc).__name__,closed_ack_sent=False)
        code=1
    finally:
        stop.set()
        # EOF requests cleanup without issuing a new robot action. There is no
        # task-time override or automatic action retry. The outer independent
        # job's existing watchdog remains responsible if a native child hangs.
        cleanup_errors=[]
        if child is not None:
            cleanup_errors.extend(finish_child(child,replies))
            report['physical_returncode']=child.returncode
        try:
            if stderr is not None:stderr.close()
        except (OSError,ValueError)as exc:
            cleanup_errors.append('stderr:'+type(exc).__name__)
        try:lease.close();report['lease_released']=True
        except (OSError,ValueError)as exc:
            cleanup_errors.append('lease:'+type(exc).__name__)
        report['cleanup_errors']=cleanup_errors
        if cleanup_errors:code=1;report['status']='supervisor_cleanup_failed'
        if code==0 and stream is not None:
            try:stream.complete();report['closed_ack_sent']=True
            except (OSError,ValueError)as exc:
                code=1;report.update(status='closed_ack_delivery_failed',error=type(exc).__name__,closed_ack_sent=False)
        if out.is_dir():(out/'formal_owner_report.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.stdout.flush();sys.stderr.flush()
    raise SystemExit(code)


if __name__=='__main__':main()
