"""Resource-gated diagnostic; bounded cleanup affects only our own container."""
import fcntl,json,subprocess,tempfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def command(args, timeout=5):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)

def idle_device(gpus, processes):
    occupied={row.split(',')[0].strip() for row in processes.splitlines() if row.strip()}
    for line in gpus.splitlines():
        fields=[v.strip() for v in line.split(',')]
        if len(fields)!=4:raise ValueError('Unexpected GPU query')
        index,uuid,memory,util=fields
        if uuid not in occupied and int(memory)<=32 and int(util)==0:return uuid
    return None

def main(config=None):
    started=time.monotonic()
    if config is not None:
        import sys
        sys.path.insert(0,str(ROOT))
        from openend.tasks.all_objects_integration.trial_input import load_config
        config=load_config(config,root=HERE)['config_path']
    with (HERE/'shared_launch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Shared test launcher already active')
        gpu=command(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
        proc=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
        if gpu.returncode or proc.returncode:raise RuntimeError('Cannot safely verify idle device')
        device=idle_device(gpu.stdout,proc.stdout)
        out=Path(tempfile.mkdtemp(prefix='shared_entry.',dir=HERE))
        name='openend-'+out.name.replace('.','-')
        report=dict(status='preflight',wall_budget_s=600,gpu_snapshot=gpu.stdout,
                    compute_snapshot=proc.stdout,device=device,container=name,
                    native_started=False,full_task_success=False)
        def save():(out/'launcher_result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(out,flush=True)
        if device is None:
            report['status']='not_started_no_idle_device';save();return 75
        args=['docker','run','-d','--user','0:0','--name',name,'--gpus','device='+device,
              '--network','none','--cpus','4','--memory','8g',
              '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y','-e','PYTHONDONTWRITEBYTECODE=1',
              '-v','/home/lingjun:/home/lingjun:ro',
              '-v','/mnt/agent-bench-data/lingjun:/mnt/agent-bench-data/lingjun:ro',
              '-v',str(out)+':/pilot:rw','--entrypoint','/isaac-sim/kit/kit','b272c84fbf8a',
              str(ROOT/'layout_task_activation_v1/task_scene_registration_v1/coffee_hotwake_cpu.kit'),
              '--ext-folder','/isaac-sim/exts','--ext-folder','/isaac-sim/extscache',
              '--allow-root','--exec',str(HERE/'probe_binding_native.py')]
        save();code=1
        if config is not None:
            # This is an input selector, never a physics/timing override.
            i=args.index('--entrypoint')
            args[i:i]=['-e','OPENEND_TRIAL_CONFIG='+config]
            report['trial_config']=config
        try:
            launched=command(args,timeout=15)
            report['launch_stderr']=launched.stderr
            if launched.returncode:raise RuntimeError('Docker launch failed')
            report['native_started']=True;report['status']='running';save()
            try:
                waited=command(['docker','wait',name],timeout=max(.1,started+570-time.monotonic()))
                if waited.returncode:raise RuntimeError('Docker wait failed')
                code=int(waited.stdout.strip())
                report.update(status='process_exited',process_exit_code=code)
            except subprocess.TimeoutExpired:
                report['status']='wall_budget_reached';code=124
        except Exception as exc:
            report.update(status='launcher_error',error=repr(exc))
        finally:
            # Reserve 30 seconds for cleanup; no docker-client SIGTERM reliance.
            try:
                inspect=command(['docker','inspect','--format','{{.State.Running}}',name],timeout=3)
                if inspect.returncode==0 and inspect.stdout.strip()=='true':
                    try:command(['docker','stop','--timeout','5',name],timeout=8)
                    except subprocess.TimeoutExpired:pass
                    inspect=command(['docker','inspect','--format','{{.State.Running}}',name],timeout=3)
                    if inspect.returncode==0 and inspect.stdout.strip()=='true':
                        command(['docker','kill',name],timeout=3)
                inspect=command(['docker','inspect','--format','{{.State.Running}}',name],timeout=3)
                report['container_stopped']=inspect.returncode==0 and inspect.stdout.strip()=='false'
                with (out/'launch.log').open('w') as log:
                    subprocess.run(['docker','logs',name],stdout=log,stderr=log,timeout=3)
                if report['container_stopped']:
                    removed=command(['docker','rm',name],timeout=3)
                    report['container_removed']=removed.returncode==0
            except Exception as exc:report['cleanup_error']=repr(exc)
            report['elapsed_wall_s']=time.monotonic()-started
            report['within_wall_budget']=report['elapsed_wall_s']<=600
            save()
        print(json.dumps(report),flush=True)
        return code

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    raise SystemExit(main(parser.parse_args().config))
