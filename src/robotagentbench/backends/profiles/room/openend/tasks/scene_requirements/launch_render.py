"""Render only on GPU1, with explicitly authorized scoped reclamation."""
import argparse,json,subprocess,sys
from pathlib import Path
from gpu_selection import inspect
from gpu1_reservation import reserve_gpu1
HERE=Path(__file__).resolve().parent
policy=json.loads((HERE/'gpu_policy.json').read_text())
if policy['selection'] != 'fixed_gpu1_reclaim':raise ValueError('Unsupported selection policy')
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(selection=policy['selection'],available=inspect(policy))))
    raise SystemExit(0)
parser=argparse.ArgumentParser();parser.add_argument('episode');parser.add_argument('--gpu',type=int);parser.add_argument('--output',default='renders_v2',choices=['renders_v2','renders_v3','renders_v4']);parser.add_argument('--reuse-output',choices=['renders_v2','renders_v3']);parser.add_argument('--views',nargs='+');args=parser.parse_args()
if bool(args.reuse_output)!=bool(args.views):parser.error('--reuse-output and --views must be supplied together')
folder=Path(args.episode).resolve()
if folder.parent != HERE/'generated_rooms':raise ValueError('Episode outside task')
if (folder/args.output).exists():raise ValueError('Render output already exists; retain it and use a fresh output')
if args.gpu is not None:
    chosen=args.gpu
    if chosen not in policy['allowed_indices']:raise ValueError('GPU outside allowed policy')
    policy=dict(policy,allowed_indices=[chosen])
# Held until process exit; advisory lock coordinates our launcher, not other users.
def progress(value):print('ROOM_PROGRESS '+json.dumps(value,ensure_ascii=False),flush=True)
gpu,lock=reserve_gpu1(policy,HERE,folder,progress)
uuid=gpu['uuid']
(folder/('render_gpu.json' if args.output=='renders_v2' else 'render_gpu_'+args.output+'.json')).write_text(json.dumps(dict(selection=policy['selection'],**gpu),indent=2))
for name in ('render_cache','render_logs'):(folder/name).mkdir(exist_ok=True)
cmd=['docker','run','--rm','--name','room-render-'+folder.name,'--gpus','device='+uuid,'--network','host','--user','0:0',
     '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y',
     '-v','/home/lingjun:/home/lingjun:ro','-v',str(HERE)+':'+str(HERE),
     '-v','/mnt/agent-bench-data/lingjun:/mnt/agent-bench-data/lingjun:ro',
     '-v',str(folder/'render_cache')+':/root/.cache',
     '-v',str(folder/'render_logs')+':/root/.nvidia-omniverse/logs',
     '--entrypoint','/isaac-sim/python.sh','b272c84fbf8a',str(HERE/'render_room.py'),str(folder),'--output',args.output]
if args.reuse_output:cmd+=['--reuse-output',args.reuse_output,'--views',*args.views]
print(f"START_GPU{gpu['index']}_STATIC_RENDER {uuid}",flush=True)
progress(dict(stage='render',message='GPU1：启动渲染器并加载房间'))
code=subprocess.call(cmd)
receipt=folder/args.output/'manifest.json'
if not receipt.is_file():
    error=folder/args.output/'error.json'
    detail=json.loads(error.read_text()).get('error') if error.exists() else '进程结束但缺少完整图片清单'
    raise RuntimeError('渲染未完成：'+detail)
raise SystemExit(code)
