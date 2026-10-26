"""Prioritize sparsely represented SUCCESS angle cells without moving authored scenes."""
import json,random
from pathlib import Path

def read(path):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return {}

def choose(root,state,extent,yaw_extent):
 cells={f'{m}:{b}':dict(successes=0,attempts=0) for m in range(3) for b in range(6)}
 qualified={(x['batch'],x['episode']) for x in state.get('qualified',[])}
 for name in state['batches']:
  batch=Path(root)/'runs'/name
  for row in read(batch/'batch.json').get('results',[]):
   sample=read(batch/f"episode_{row['episode']:04d}"/'initial/initialization.json').get('randomization')
   if not sample:continue
   m=min(2,max(0,int((sample['machine_yaw_deg']+yaw_extent)/(2*yaw_extent/3))))
   b=min(5,max(0,int(sample['bag_yaw_deg']/15)))
   c=cells[f'{m}:{b}'];c['attempts']+=1
   if (name,row['episode']) in qualified:c['successes']+=1
 rng=random.Random(9300000+state['next_seed'])
 bag_bins={b:dict(successes=sum(cells[f'{m}:{b}']['successes'] for m in range(3)),attempts=sum(cells[f'{m}:{b}']['attempts'] for m in range(3))) for b in range(6)}
 bag_order=list(bag_bins);rng.shuffle(bag_order)
 # Prioritize actual successful bag-angle coverage before machine-angle coverage.
 bag_bin=min(bag_order,key=lambda b:(bag_bins[b]['successes'],bag_bins[b]['attempts']))
 keys=[f'{m}:{bag_bin}' for m in range(3)];rng.shuffle(keys)
 key=min(keys,key=lambda k:(cells[k]['successes'],cells[k]['attempts']))
 state['bag_angle_coverage']={f'{b*15}-{(b+1)*15}':counts for b,counts in bag_bins.items()}
 state['coverage_policy']='bag_15deg_success_first_v2'
 m,b=map(int,key.split(':'));seed=state['next_seed'];skipped=[]
 while True:
  proposal=random.Random(seed)
  for _ in range(5):proposal.random()
  bag_yaw=proposal.uniform(0,90)
  if min(5,int(bag_yaw/15))==b:break
  skipped.append(seed);seed+=1
  if len(skipped)>=100:raise RuntimeError('Unable to find seeded bag-yaw stratum')
 pose=[rng.uniform(-extent,extent),rng.uniform(-extent,extent),rng.uniform(-yaw_extent+m*2*yaw_extent/3,-yaw_extent+(m+1)*2*yaw_extent/3)]
 state['coverage']=cells
 state['coverage_selection']=dict(cell=key,seed=seed,uninstantiated_seed_candidates=skipped,predicted_bag_yaw_deg=bag_yaw)
 return seed,pose
