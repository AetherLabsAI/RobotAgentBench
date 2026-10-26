"""Read-only actual material landmarks and real tabletop support, no pose writes."""
from pathlib import Path
import sys,json,numpy as np
p=Path(__file__).parent;ep=p/'runs'/sys.argv[1];stage=sys.argv[2];z=np.load(ep/('observed_'+stage+'.npz'));q=z['cloth_q'];uv=z['uv'];lo=uv.min(0);hi=uv.max(0);table_lo=np.array([-2.35,-.7210436]);table_hi=np.array([-1.574649,.4789564]);inside=np.all((q[:,:2]>=table_lo)&(q[:,:2]<=table_hi),axis=1);on=inside&(abs(q[:,2]-.451)<.012);result=dict(stage=stage,read_only=True,material_bounds_m=[lo.tolist(),hi.tolist()],table_bounds_xy_m=[table_lo.tolist(),table_hi.tolist()],corners=[],physical_state_writes=0)
for u in [lo[0],hi[0]]:
 for v in [lo[1],hi[1]]:
  target=np.array([u,v]);idx=int(np.linalg.norm(uv-target,axis=1).argmin());mask=np.linalg.norm(uv-target,axis=1)<.02;result['corners'].append(dict(uv=target.tolist(),vertex=idx,xyz=q[idx].tolist(),patch_median_xyz=np.median(q[mask],axis=0).tolist(),patch_table_fraction=float(on[mask].mean())))
audit=json.loads((ep/('recovery_grip_'+stage+'.json')).read_text());hands={}
for side in ['l','r']:
 ids=audit['hands'][side]['indices']
 if ids:hands[side]=dict(uv=uv[ids].mean(0).tolist(),xyz=q[ids].mean(0).tolist(),nodes=len(ids))
result['observed_between_pad_patches']=hands
if len(hands)==2:
 cu=np.array([hands[s]['uv']for s in ['l','r']]);cx=np.array([hands[s]['xyz']for s in ['l','r']]);boundary=[(float(np.mean(cu[:,axis]-lo[axis])),axis,0)for axis in [0,1]]+[(float(np.mean(hi[axis]-cu[:,axis])),axis,1)for axis in [0,1]];_,axis,end=min(boundary);other=1-axis;opposite=hi[axis]if end==0 else lo[axis];free=(abs(uv[:,axis]-opposite)<.02)&(uv[:,other]>lo[other]+.025)&(uv[:,other]<hi[other]-.025);result['held_edge']=dict(nearest_material_boundary_axis=axis,boundary_end='min'if end==0 else'max',captured_center_xyz=cx.mean(0).tolist(),captured_rest_span_m=float(np.linalg.norm(cu[1]-cu[0])),actual_patch_span_m=float(np.linalg.norm(cx[1]-cx[0])),rest_depth_to_opposite_edge_m=float(abs(opposite-cu[:,axis].mean())));result['free_edge']=dict(nodes=int(free.sum()),median_xyz=np.median(q[free],axis=0).tolist(),xyz_min=q[free].min(0).tolist(),xyz_max=q[free].max(0).tolist(),z_p05_p50_p95=np.quantile(q[free,2],[.05,.5,.95]).tolist(),inside_table_xy_fraction=float(inside[free].mean()),physical_table_contact_fraction=float(on[free].mean()))
out=ep/('actual_landmarks_'+stage+'.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
