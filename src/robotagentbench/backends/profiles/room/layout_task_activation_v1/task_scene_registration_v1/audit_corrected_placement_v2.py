"""Independent measured-pose stability gate, not a manipulation qualification."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
r=json.loads((a.run/'navigation_result.json').read_text());d=np.load(a.run/'native_poses.npz')
c=json.loads((a.run/'composition.json').read_text());stage=Usd.Stage.Open(str(a.run/'native_initial.usdc'))
xf=UsdGeom.XformCache();rows=[]
for i,path in enumerate(r['body_paths']):
    if not any(k in path for k in ('/CoffeeBag/','/CoffeeCup/','/Shelf_tier')):continue
    authored=np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
    q=d['bodies'][:,i];rot=Rotation.from_quat(q[:,3:7]);reference=Rotation.from_matrix(authored[:3,:3].T)
    up=(rot*reference.inv()).apply([0,0,1]);tilt=np.degrees(np.arccos(np.clip(up[:,2],-1,1)))
    drift=np.linalg.norm(q[:,:2]-authored[3,:2],axis=1)
    limit=10 if '/CoffeeBag/' in path else 11.5 if '/CoffeeCup/' in path else 10
    # Coffee upright thresholds are the existing public task initial-state limits.
    # Shelf 10 deg and 10 mm XY are explicit preview-stability guards, not task thresholds.
    row={'path':path,'max_tilt_from_authored_upright_deg':float(tilt.max()),'final_tilt_deg':float(tilt[-1]),
         'max_xy_drift_m':float(drift.max()),'max_z_origin_drift_m':float(abs(q[:,2]-authored[3,2]).max()),
         'upright_limit_deg':limit,'upright_passed':bool(tilt.max()<=limit),'xy_stability_passed':bool(drift.max()<.01)}
    rows.append(row)
assert len(rows)==16,len(rows)
out={'frames_checked':len(r['rows']),'native_passed':r['status']=='native_action22_navigation_smoke_passed',
     'objects':rows,'passed':r['status']=='native_action22_navigation_smoke_passed' and all(x['upright_passed'] and x['xy_stability_passed'] for x in rows),
     'scope':'All recorded 20 Hz poses, not 60 Hz substeps or full task/continuous collision validation',
     'coffee_upright_source':'evaluation/tasks/coffee_machine/api_runtime.py INITIAL_TILT_DEG=11.5 and INITIAL_BAG_TILT_DEG=10',
     'preview_stability_guards':{'xy_drift_m':.01,'shelf_tilt_deg':10},'changes_to_physics_or_assets':False}
(a.run/'corrected_placement_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)
