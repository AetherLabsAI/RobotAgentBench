"""Template/verify are CPU-only; serve explicitly loads the operator's policy."""
import argparse
import json
import os
import sys

from . import config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('template')
    sub.add_parser('server-template',help='Portable configuration for the actual tested server chain')
    setup=sub.add_parser('configure',help='Generate a configuration from installed Runtime, saved launch and matching G1 model files; never run')
    for name in ('runtime','sdk-site','launch','native-python','model-python','checkpoint','stats',
                 'stats-key','control-schema','vlm','lock-root','task','output','config'):
        setup.add_argument('--'+name,required=True)
    setup.add_argument('--worker-gpu','--gpu',type=int,required=True)
    setup.add_argument('--model-gpu',type=int,required=True)
    setup.add_argument('--max-calls',type=int,default=100)
    setup.add_argument('--input-root',dest='input_roots',action='append',default=[],
                       help='Trusted read-only saved-input root; repeat as needed')
    setup.add_argument('--public-input-binding',help='Hash-bound saved public room instruction/context')
    tested=sub.add_parser('run',help='Run the server-derived loop with an operator-supplied existing worker argv')
    tested.add_argument('--config',required=True);tested.add_argument('--task',required=True)
    tested.add_argument('--output',required=True)
    service=sub.add_parser('tested-service',help=argparse.SUPPRESS)
    service.add_argument('--service-config',required=True)
    verify=sub.add_parser('verify');verify.add_argument('--config',required=True)
    server=sub.add_parser('serve');server.add_argument('--config',required=True)
    server.add_argument('--port',type=int,default=8765)
    bridge=sub.add_parser('bridge');bridge.add_argument('--endpoint',required=True)
    bridge.add_argument('--timeout-s',type=float,default=120)
    bridge.add_argument('--worker-read-fd',type=int,required=True)
    bridge.add_argument('--worker-write-fd',type=int,required=True)
    bridge.add_argument('--max-calls',type=int,required=True)
    args=parser.parse_args(argv)
    if args.command=='configure':
        from .configure import save
        values=vars(args).copy();values.pop('command');destination=values.pop('config')
        print(json.dumps(save(destination,**values),indent=2));return 0
    if args.command=='server-template':
        from .operator import template
        print(json.dumps(template(),indent=2));return 0
    if args.command=='run':
        from .operator import run
        print(json.dumps(run(args.config,args.task,args.output),indent=2));return 0
    if args.command=='tested-service':
        from .config import read
        from .server_policy import serve_stdio
        value=read(args.service_config)
        if set(value) not in ({'model','prompts'},{'model','prompts','native_receipt'}):
            raise ValueError('Exact model service configuration required')
        from .server_policy import camera_profile
        camera_profile(value.get('native_receipt'))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        serve_stdio(value['model'],value['prompts'],native_receipt=value.get('native_receipt'));return 0
    if args.command=='template':print(json.dumps(config.template(),indent=2));return 0
    if args.command=='verify':
        value=config.load(args.config)
        print(json.dumps(dict(configuration_verified=True,model_loaded=False,
            weights_downloaded=False,codec=value['codec'])));return 0
    if args.command=='serve':
        value=config.load(args.config)
        # Local resources only. Operators install upstream separately; this
        # example never patches Transformers or downloads model weights.
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        from .policy import Policy
        from .http import serve
        policy=Policy(value)
        print('InternVLA example ready on loopback only',file=sys.stderr,flush=True)
        serve(policy,port=args.port).serve_forever();return 0
    if args.worker_read_fd==args.worker_write_fd or min(args.worker_read_fd,args.worker_write_fd)<0:
        parser.error('Distinct already-open worker pipe descriptors required')
    from .http import Client
    from .bridge import run
    with os.fdopen(os.dup(args.worker_read_fd),'r')as src,os.fdopen(os.dup(args.worker_write_fd),'w')as dst:
        result=run(src,dst,Client(args.endpoint,args.timeout_s),max_calls=args.max_calls)
    print(json.dumps(result,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
