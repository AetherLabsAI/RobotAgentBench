"""Compare the Python native adapter against original browser JS, not fixtures."""
import json,math,random,subprocess,sys,unittest
from pathlib import Path
from sector_activation_v1 import step
ROOT=Path(__file__).resolve().parent
NODE='/Users/dylan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
JS=ROOT.parent/'web/logic.js'

def compare(c,rows):
    # JS short-circuit returns undefined for absent flags (omitted by JSON).
    # Supply explicit false booleans to compare its ready value as well.
    inputs=[dict(robot=r['robot'],flags={**dict(task=False,stopped=False,loaded=False),**r.get('flags',{})}) for r in rows]
    js="const L=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d);process.stdin.on('end',()=>{let d=JSON.parse(s),p={active:false};console.log(JSON.stringify(d.inputs.map(x=>p=L.step(p,d.config,x.robot,x.flags))));});"
    expected=json.loads(subprocess.check_output([NODE,'-e',js,str(JS)],input=json.dumps(dict(config=c,inputs=inputs)).encode()))
    p={'active':False}
    for i,(inp,e) in enumerate(zip(inputs,expected)):
        p=step(p,c,inp['robot'],inp['flags'])
        for k,v in e.items():
            got=p[k]
            if isinstance(v,(int,float)) and not isinstance(v,bool):
                assert math.isclose(v,got,abs_tol=1e-9), (i,k,v,got)
            else: assert v==got,(i,k,v,got)
        if 'activation' in rows[i]:
            for k in ('active','state','reason','overlap','ready'): assert rows[i]['activation'][k]==e[k],(i,k)
    return len(inputs)

class Parity(unittest.TestCase):
    def test_random_and_boundary(self):
        c=next(e['config'] for e in json.loads((ROOT.parent/'calibration_drafts_layout01.json').read_text())['entries'] if e['id']=='coffee')
        c=dict(c,wake=1.32,exit=1.82,preload=2.12)
        rng=random.Random(0)
        rows=[dict(robot=dict(xy=[c['target'][0]+rng.uniform(-3,3),c['target'][1]+rng.uniform(-3,3)],yaw=rng.uniform(-720,720)),flags=dict(held=i%113==0,moving=i%127==0)) for i in range(4000)]
        for p in c['targetPolygon']+[c['target']]:
            rows += [dict(robot=dict(xy=p,yaw=yaw)) for yaw in (0,45,90,180,270)]
        self.assertEqual(compare(c,rows),4025)

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1].endswith('.json'):
        d=json.loads(Path(sys.argv[1]).read_text());print('Original JS native-row parity:',compare(d['activation_config'],d['rows']))
    else: unittest.main()
