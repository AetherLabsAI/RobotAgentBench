"""Continue fresh wide-range bag episodes until 50 complete physical three-camera results."""
import fcntl,json,os,random,subprocess,sys,time
from pathlib import Path
from success_coverage import choose
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/bag_until50_v1';OUT.mkdir(exist_ok=True)
TARGET=50

def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {} if default is None else default

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)

def eligible(batch):
 found=[]
 for row in read(batch/'batch.json').get('results',[]):
  motion=batch/f"episode_{row['episode']:04d}"/'motion'
  result=read(motion/'result.json');video=read(motion/'video_manifest.json');plan=read(motion/'plan.json')
  if not (result.get('passed') and result.get('motion_completed') and result.get('checks') and all(result['checks'].values())):continue
  if not (plan.get('skip_intermediate_holds') and plan.get('video_capture_fps')==30 and video.get('actual_render_fps')==30):continue
  if all((motion/video.get('views',{}).get(k,{}).get('video','missing')).is_file() and (motion/video['views'][k]['video']).stat().st_size>0 for k in ('head','left','right')):
   found.append(dict(batch=batch.name,episode=row['episode'],seed=row['seed']))
 return found

def main():
 with (OUT/'supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read(OUT/'progress.json',dict(target=TARGET,next_seed=2103,batches=['coffee_wide_machine_2100_v1'],infrastructure_streak=0))
  inherited_child=state.get('child_pid')
  state['pid']=os.getpid()
  def update(status):
   qualified=[x for name in state['batches'] for x in eligible(ROOT/'runs'/name)]
   state.update(state=status,qualified=qualified,completed=len(qualified),updated_at=time.time());save(OUT/'progress.json',state)
  update('waiting_for_current_batch')
  queue=ROOT/'runs/coffee_wide_machine_2100_v1.job.json'
  while read(queue).get('state') in {'queued_after_current_batch','running'}:
   job=read(queue)
   if not Path('/proc',str(job.get('pid',-1))).exists():
    state['error']='Existing wide-range queue exited without a final status';update('needs_attention');return
   update('waiting_for_current_batch');time.sleep(20)
  if inherited_child:
   cmd=Path('/proc',str(inherited_child),'cmdline')
   active=state.get('active_batch','')
   while cmd.exists():
    raw=cmd.read_bytes()
    if b'bag_cycle.py' not in raw or active.encode() not in raw:break
    update('waiting_for_existing_episode');time.sleep(20)
  while True:
   update('running')
   if state['completed']>=TARGET:update('complete');return
   seed,pose=choose(ROOT,state,0.1,20.0);name=f'bag_until50_seed_{seed}';batch=ROOT/'runs'/name
   # Reserve the seed persistently before any scene is authored; no failed-pose substitution.
   if name not in state['batches']:state['batches'].append(name)
   state['next_seed']=seed+1;state['active_batch']=name;update('running')
   if batch.exists():
    state['last_error']='Prior partial run retained: '+name
    state['infrastructure_streak']+=1
   else:
    source_path=ROOT/'batch_viewer/parallel_sources.json';sources=read(source_path,[])
    if not any(s['id']==name for s in sources):
     sources.append(dict(id=name,gpu=5,root=str(ROOT),batch=name));save(source_path,sources)
    env=dict(os.environ,COFFEE_ISAAC_CONTAINER='coffee_batch_202_gpu5')
    with (OUT/f'{seed}.log').open('w') as log:
     proc=subprocess.Popen([sys.executable,str(ROOT/'bag_task/bag_cycle.py'),'--episodes','1','--seed',str(seed),'--bag-width-m','0.065','--bag-depth-m','0.12','--continuous-pour','--machine-pose-json',json.dumps(pose),'--batch-name',name],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
     state['child_pid']=proc.pid;update('running')
     while proc.poll() is None:update('running');time.sleep(20)
    rows=read(batch/'batch.json').get('results',[])
    infrastructure=not rows or any(r['status'] in {'execution_error','running'} for r in rows)
    state['infrastructure_streak']=state.get('infrastructure_streak',0)+1 if infrastructure else 0
   if state['infrastructure_streak']>=3:update('needs_attention');return

if __name__=='__main__':main()
