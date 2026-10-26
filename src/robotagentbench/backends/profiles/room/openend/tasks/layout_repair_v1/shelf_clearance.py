"""Approved layout23-only pre-physics shelf translation, with a scene gate."""
import itertools
import numpy as np

def clear_wall(stage, path, bounds_of, floors):
    from pxr import Gf, UsdGeom
    from openend.tasks.scene_requirements.station_scene_checks import check, EPS
    group=dict(id='canonical_shelf',members=[path],support=floors[0],internal_supports=floors[1:])
    original=np.asarray(bounds_of(path),dtype=float)
    floor_boxes=[np.asarray(bounds_of(p)) for p in floors]
    def on_floor(box):
        return all(any(np.all(np.asarray(c)>=b[0,:2]-EPS) and np.all(np.asarray(c)<=b[1,:2]+EPS)
                       for b in floor_boxes) for c in itertools.product(*zip(box[0,:2],box[1,:2])))
    def probe(shift):
        box=original+shift
        return check(stage,[group],lambda _:box)
    first=probe(np.zeros(3))
    if first['passed']:return dict(translation_m=[0.,0.,0.],already_clear=True)
    candidates=[]
    for axis,direction in itertools.product((0,1),(-1,1)):
        shift=np.zeros(3);report=first
        for _ in range(32):
            obstacles=[np.asarray(r['obstacle_bounds']) for r in report['failures']]
            shift[axis]=(max(b[1,axis] for b in obstacles)-original[0,axis]+EPS if direction>0 else
                         min(b[0,axis] for b in obstacles)-original[1,axis]-EPS)
            if not on_floor(original+shift):break
            report=probe(shift)
            if report['passed']:
                candidates.append((float(np.linalg.norm(shift)),shift.copy(),report));break
    if not candidates:raise ValueError('No collision-free floor-contained axial shelf translation')
    _,shift,report=min(candidates,key=lambda r:r[0])
    prim=stage.GetPrimAtPath(path);cache=UsdGeom.XformCache()
    matrix=cache.GetLocalToWorldTransform(prim)
    matrix.SetTranslateOnly(matrix.ExtractTranslation()+Gf.Vec3d(*map(float,shift)))
    local=matrix*cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(local)
    actual=check(stage,[group],bounds_of)
    if not actual['passed']:raise ValueError('Shelf translation failed composed-geometry recheck')
    return dict(translation_m=shift.tolist(),distance_m=float(np.linalg.norm(shift)),
                method='shortest_collision_free_axis_aligned_translation_from_mesh_blocker_bounds',
                initial_obstacles=first['failures'],tested_clear_candidates=len(candidates),
                geometry_gate=actual,orientation_size_and_physics_unchanged=True,
                limitation='minimum among four axis-aligned directions, not a global continuous optimum')
