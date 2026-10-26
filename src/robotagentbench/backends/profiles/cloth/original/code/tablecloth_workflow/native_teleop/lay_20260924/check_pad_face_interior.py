import sys,json,numpy as np
from pathlib import Path
p=Path(__file__).parent;source=(p/'plan_pad_face_interior.py').read_text().split('corner=corner+.010')[0];cs=json.loads(Path(sys.argv[1]).read_text());outpath=Path(sys.argv[2]);out=[]
for i,c in enumerate(cs):
 f=outpath.with_name(outpath.stem+f'_c{i}.json');f.write_text(json.dumps([c]));sys.argv=['plan_pad_face_interior.py',str(f),'unused'];ns={'__file__':str(p/'plan_pad_face_interior.py')};exec(source,ns)
 try:
  v=ns['solve'](ns['values'],ns['corner']+.010*ns['direction']);v=ns['robot'].calibrate_loops(v|{'idx71_gripper_r_inner_joint1':-.012,'idx81_gripper_r_outer_joint1':.012});mid,h,poses,o=ns['geom'](v);clear=float(ns['hand_distance'](poses).min());print(i,c['uv'],'closedclear',clear);out.append(dict(candidate=i,uv=c['uv'],closed_clearance_m=clear,input=str(f)))
 except Exception as e:print(i,repr(e))
outpath.write_text(json.dumps(out,indent=2))
