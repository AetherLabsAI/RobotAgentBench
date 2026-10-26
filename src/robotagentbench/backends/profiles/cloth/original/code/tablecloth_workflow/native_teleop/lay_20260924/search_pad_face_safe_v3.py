"""Proposal search and closed native-pad clearance for the same settled cloth."""
from pathlib import Path
import json,sys,runpy
p=Path(__file__).parent;snapshot=Path(sys.argv[1]);out=Path(sys.argv[2]);raw=out.with_name('raw_tip_candidates.json');check=out.with_name('closed_candidate_clearance.json')
sys.argv=['search_pad_face_interior_v3.py',str(snapshot),str(raw)];runpy.run_path(str(p/'search_pad_face_interior_v3.py'),run_name='__main__')
sys.argv=['check_pad_face_interior.py',str(raw),str(check)];runpy.run_path(str(p/'check_pad_face_interior.py'),run_name='__main__')
cs=json.loads(raw.read_text());good=[]
for row in json.loads(check.read_text()):
 if row['closed_clearance_m']>.008:
  c=cs[row['candidate']];c['predicted_closed_hand_basket_clearance_m']=row['closed_clearance_m'];good.append(c)
out.write_text(json.dumps(good,indent=2)+'\n')
