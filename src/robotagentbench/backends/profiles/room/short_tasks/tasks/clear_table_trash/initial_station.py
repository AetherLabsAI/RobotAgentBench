"""Approved Short 7 station ranges; no object or physics mutation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'source_initial_samples.json'
AUDIT = HERE/'source_range_audit.json'
TABLE = '/World/Assets/CoffeeTable'


def sample_candidates(seed, delta, table_bounds, count=100):
    if type(seed) is not int or not 1 <= seed <= 4294967295:
        raise ValueError('Positive uint32 seed required')
    delta = np.asarray(delta,dtype=float)
    table_bounds = np.asarray(table_bounds,dtype=float)
    if delta.shape!=(4,4) or table_bounds.shape!=(2,3) or not np.isfinite(delta).all() or not np.isfinite(table_bounds).all():
        raise ValueError('Finite table frame required')
    rows = json.loads(SOURCE.read_text())['samples']
    audit = json.loads(AUDIT.read_text())['robot']
    # Source table is 0.8 m deep: convert measured pivot offsets to front edge.
    depth = np.array(audit['pivot_x_m'])-.4
    lateral = np.array(audit['pivot_y_m'])
    yaw = np.radians(audit['yaw_deviation_from_source_front_deg'])
    centre=(table_bounds[0]+table_bounds[1])/2
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'trash-station-v1:{seed}'.encode()).digest(),'big'))
    for i in range(count):
        distance=float(rng.uniform(*depth)); offset=float(rng.uniform(*lateral)); angle=float(rng.uniform(*yaw))
        base=np.array([table_bounds[1,0]+distance,centre[1]+offset,0.,1.])@delta
        inward=np.array([-1.,0.,0.])@delta[:3,:3]
        heading=math.atan2(inward[1],inward[0])+angle
        heading=math.atan2(math.sin(heading),math.cos(heading))
        row=rows[int(rng.integers(len(rows)))]
        names,values=row['joint_names'],row['first_measured_joints_rad_or_m']
        if len(names)!=34 or len(set(names))!=34 or len(values)!=34 or not np.isfinite(values).all():
            raise ValueError('Invalid source joint vector')
        yield dict(candidate_index=i,sampling='continuous_source_table_edge_station_v1',
            initial_base_xyyaw=[float(base[0]),float(base[1]),heading],
            edge_distance_m=distance,lateral_from_centre_m=offset,yaw_deviation_deg=math.degrees(angle),
            posture_source_id=row['id'],targets_rad=dict(zip(names,values)),
            joints_sampling='intact source initial vector, not independent joint mixing')


def table_context(stage, manifest):
    from pxr import Usd,UsdGeom
    baseline_path=Path(manifest['source_layout']).parent/'ID_01_LIVING_STRAIGHT.usdc'
    baseline=Usd.Stage.Open(str(baseline_path))
    def vertices(scene):
        cache=UsdGeom.XformCache(); out={}
        for p in Usd.PrimRange(scene.GetPrimAtPath(TABLE)):
            if '/black_box_1' in str(p.GetPath()) or not p.IsA(UsdGeom.Mesh):continue
            v=np.asarray(p.GetAttribute('points').Get())
            if len(v):
                out[str(p.GetPath())]=(np.c_[v,np.ones(len(v))]@np.asarray(cache.GetLocalToWorldTransform(p)))[:,:3]
        return out
    aa,bb=vertices(baseline),vertices(stage)
    if not aa or set(aa)!=set(bb) or any(aa[k].shape!=bb[k].shape for k in aa):
        raise ValueError('Coffee-table geometry differs from approved sampler')
    a,b=(np.concatenate([v[k] for k in sorted(v)]) for v in (aa,bb))
    ac,bc=a.mean(0),b.mean(0);u,_,vt=np.linalg.svd((a-ac).T@(b-bc));rotation=u@vt
    delta=np.eye(4);delta[:3,:3]=rotation;delta[3,:3]=bc-ac@rotation
    if np.linalg.det(rotation)<0 or not np.allclose(rotation[2],[0,0,1],atol=1e-7) or np.max(np.abs(a@rotation+delta[3,:3]-b))>1e-5:
        raise ValueError('Coffee-table transfer is not rigid and upright')
    bounds=np.array([a.min(0),a.max(0)])
    if not np.allclose(bounds[1,:2]-bounds[0,:2],[.8,1.2],atol=1e-5):
        raise ValueError('Unexpected canonical table dimensions')
    return delta,bounds,baseline_path

