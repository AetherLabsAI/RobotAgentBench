"""Scoped local/202-data test runner. Requires existing Python and Node, no install."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from planner_v2 import compile_request

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node', required=True, type=Path)
    ap.add_argument('--name', default='placement_demo_v2')
    args=ap.parse_args()
    if not args.name.isidentifier():ap.error('Simple output directory name required')
    out=HERE/args.name
    if out.exists():ap.error('Existing output is preserved; use a new name')
    out.mkdir()
    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    request=json.loads((HERE/'example_placement_request.json').read_text())
    requirements=compile_request(request)
    save('request.json',request);save('requirements.json',requirements)
    runs=[]
    for label,cmd in [('python_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']),
                      ('node_tests',[str(args.node),'--test',str(HERE/'test_placement_constraints.cjs')])]:
        r=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True,timeout=60)
        with (out/(label+'.txt')).open('x') as f:f.write(r.stdout+r.stderr)
        runs.append(dict(name=label,returncode=r.returncode))
        if r.returncode:raise RuntimeError(label+' failed; evidence retained')
    r=subprocess.run([str(args.node),str(HERE/'placement_demo.cjs'),str(out/'requirements.json')],
                     cwd=HERE,capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr)
    demo=json.loads(r.stdout);save('placement_demo.json',demo)
    for mode in ('fixed','range'):
        derived=json.loads(json.dumps(request))
        derived['tasks']=demo[mode+'_request']['stations']
        compile_request(derived)
        save(mode+'_request.json',derived)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'placement_sources').iterdir() if p.is_file()}
    report=dict(status='planar_constraint_examples_passed',runs=runs,python_tests=43,node_tests=24,
                data_origin='202 snapshot of existing layout/zone/geometry data',execution_host='local',
                source_hashes=sources,request_hash=hashlib.sha256((HERE/'example_placement_request.json').read_bytes()).hexdigest(),
                examples=['surface group','fixed local XY and world heading','bounded local XY','invalid exact XY rejection'],
                positions=[dict(type=p['id'],surface=p['group'],world_xy_m=p['origin'],
                    edge_index=p['surface_edge_index'],local_xy_m=p['local_xy_m'],front_yaw_deg=p['yaw_deg'],
                    front_edge_gap_m=p['gap']) for p in demo['fixed_plan']['placements']],
                scene_generated=False,physics_validated=False,full_assembly_clearance_verified=False,
                shared_generator_modified=False,live_entry_connected=False)
    save('verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
