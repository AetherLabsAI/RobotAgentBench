"""One-command Codex robot operator; default describe does not execute anything."""
import argparse,json,math,os,subprocess,time
from pathlib import Path

CONTINUE='The native environment has not terminated. Continue with one safe Arena tool. A prose summary is not task completion. Use done with an honest reason to stop.'

def configuration(path):
    value=json.loads(Path(path).read_text())
    required={'schema','codex_executable','model','effort','max_model_responses','transport',
        'prepare_argv','worker_argv','output','expected_cli_version'}
    if set(value)!=required or value['schema']!='robotagentbench.codex_example.v1':raise ValueError('Explicit trusted operator configuration required')
    for name in ('worker_argv','prepare_argv'):
        if not isinstance(value[name],list)or any(not isinstance(x,str)or not x or '\0'in x for x in value[name]):raise ValueError('Use argv lists, not shell commands')
    if not value['worker_argv']:raise ValueError('An original coding worker command is required')
    if value['transport']is not None:raise ValueError('This exact server profile uses only its original default transport')
    if type(value['max_model_responses'])is not int or value['max_model_responses']<1:raise ValueError('Explicit model-response budget required')
    for name in ('codex_executable','model','effort','expected_cli_version'):
        if not isinstance(value[name],str)or not value[name].strip():raise ValueError('Explicit CLI/model recipe required')
    out=Path(value['output'])
    if not out.is_absolute()or out.resolve()!=out or out==Path(out.anchor)or out==Path.home():raise ValueError('Canonical fresh episode output required')
    return value

def save(path,value):
    temporary=path.with_suffix(path.suffix+'.next')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def run(value):
    from .app_server import AppServer
    from .bridge import RobotBridge,tool_specs,text_item
    from .protocol import Worker
    from .session import run_turn
    out=Path(value['output']);out.mkdir(parents=True,exist_ok=False);os.chmod(out,0o700)
    report=dict(status='initializing',model=value['model'],effort=value['effort'],
        model_responses=0,actions_executed=0,evaluation_valid=None,score=None,
        native_cleanup_verified=False,automatic_retries=False)
    worker=server=bridge=None
    def checkpoint():
        if bridge:report.update(model_responses=bridge.model_responses,actions_executed=bridge.observed['step'],
            in_flight_action=bridge.pending is not None)
        save(out/'operator_result.json',report)
    checkpoint()
    # Version inspection is explicit execute-only and cannot call a model.
    try:
        version=subprocess.run([value['codex_executable'],'--version'],capture_output=True,text=True,check=True).stdout.strip()
        if version!=value['expected_cli_version']:raise ValueError('CLI version differs from the explicitly selected deployment')
    except BaseException as error:
        report.update(status='invalid_before_start',error_type=type(error).__name__,evaluation_valid=False)
        checkpoint();raise
    report['cli_version']=version
    with (out/'worker_stderr.log').open('x')as stderr:
        try:
            if value['prepare_argv']:
                subprocess.run(value['prepare_argv'],stdout=stderr,stderr=stderr,check=True)
            worker=Worker(value['worker_argv'],stderr)
            bridge=RobotBridge(worker,out/'observations',max_model_responses=value['max_model_responses'])
            if bridge.observed['step']!=0:raise ValueError('A new episode must start at control step zero')
            server=AppServer(value['codex_executable'])
            started=server.start_thread(value['model'],bridge.instructions,tool_specs())
            thread=started['thread']['id'];items=bridge.initial_items();turn=0
            report['status']='model_running';checkpoint()
            while True:
                turn+=1
                result=run_turn(server,thread,bridge,input_items=items,effort=value['effort'])
                checkpoint()
                if bridge.pending:raise RuntimeError('Uncertain robot action: no replay or replacement')
                if bridge.completed or bridge.abandoned:break
                if result['status']!='turn_completed_environment_held':raise RuntimeError('Model turn stopped without a native result')
                items=[text_item(CONTINUE)]
            server.close();server=None
            ack=worker.exchange({'op':'close'})
            code=worker.close();worker=None
            clean=ack.get('op')=='closed'and code==0
            report.update(status='closed'if clean else'cleanup_unverified',native_cleanup_verified=clean,
                model_done_is_not_success=True,outcome='native_ended'if bridge.completed else'model_done',
                worker_exit_code=code)
            # Owner result is operator-only. Never pass it to App Server or
            # infer success merely from process completion / a model's done.
            native=ack.get('result',{})
            report['owner_status']=native.get('status')if isinstance(native,dict)else None
            if clean and isinstance(native,dict):
                for name in ('evaluation_valid','success'):
                    report[name]=native[name]if type(native.get(name))is bool else None
                for name in ('scoring_status','termination_reason'):
                    report[name]=native[name]if isinstance(native.get(name),str)else None
                if type(native.get('score'))in(int,float)and math.isfinite(native['score']):report['score']=native['score']
                scores=native.get('scores')
                if isinstance(scores,dict)and all(type(x)in(int,float)and math.isfinite(x)for x in scores.values()):
                    report['owner_scores']=scores
                if type(native.get('success'))is bool:report['owner_success']=native['success']
            checkpoint();return 0 if clean else 1
        except BaseException as error:
            report.update(status='invalid_or_held',error_type=type(error).__name__,score=None,evaluation_valid=False)
            checkpoint();raise
        finally:
            if server:server.close()
            if worker:
                if bridge is None or bridge.pending:
                    # Same conservative server behavior: no uncertain motion
                    # replay/force-kill/automatic recovery or scored-zero claim.
                    report['status']='uncertain_worker_held_for_operator';checkpoint()
                    worker.process.wait()
                else:
                    try:
                        ack=worker.exchange({'op':'close'});code=worker.close()
                        report['native_cleanup_verified']=ack.get('op')=='closed'and code==0
                    except Exception as error:report['cleanup_error_type']=type(error).__name__
            checkpoint()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    value=configuration(args.config)
    if not args.execute:
        print(json.dumps(dict(mode='describe',model=value['model'],effort=value['effort'],
            max_model_responses=value['max_model_responses'],worker_argv=value['worker_argv'],
            prepare_argv=value['prepare_argv'],model_started=False,native_started=False),indent=2));return 0
    return run(value)

if __name__=='__main__':raise SystemExit(main())
