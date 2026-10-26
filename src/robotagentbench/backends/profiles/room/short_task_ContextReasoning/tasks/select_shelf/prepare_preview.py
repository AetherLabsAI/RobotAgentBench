"""Task-owned static initial scenes using unchanged stock packing and atomic FK.

No simulation, shared registration, scoring or physical parameter changes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from .inventory_plan import plan_inventory
from .bank import BANK_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REG = ROOT/'layout_task_activation_v1/task_scene_registration_v1'
LIB = ROOT/'all_assets'
BASE = ROOT/'short_tasks/agent_probe_v1/runs/all_001/stock_wooden_tray/native'
sys.path.extend([str(REG), str(ROOT/'openend/tasks/scene_requirements')])
sys.path.insert(0, '/home/lingjun/robot_agent_arena/openend/tasks/scene_requirements/cpu_dependencies')
from repair_room_placement_v2 import cs, _proxies, _separated, shape_mesh, trimesh, MultiPoint, Rotation
from shelf_sampler import sample_scene
from .expanded_placement import sample_stock
from .expanded_catalog import asset_path, NEW_PRODUCTS
from short_tasks.tasks.stock_wooden_tray import initial_station, station_geometry
from short_tasks.tasks.retrieve_from_drawer.prepare_initial import fk


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def build(question_id, seed, output, *, base_launch=None):
    output = output.resolve()
    if not output.is_relative_to(HERE/'checks'):
        raise ValueError('Output must remain inside this task checks directory')
    output.mkdir(parents=True, exist_ok=False)
    plan = plan_inventory(question_id, seed)
    save(output/'inventory_plan_private.json', plan)
    launch = json.loads(Path(base_launch).read_text()) if base_launch else json.loads((BASE/'launch.json').read_text())
    manifest = json.loads(Path(launch['manifest']).read_text())
    base_scene = manifest['scene']
    scene = output/'room.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    stage.GetRootLayer().subLayerPaths = [base_scene]
    original = Usd.Stage.Open(base_scene)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    old_stock = [r['path'] for r in manifest['objects'] if r.get('source') == 'snack_shelf']
    for path in old_stock:
        stage.OverridePrim(path).SetActive(False)
    counts = {l['support_id']:l['count'] for l in plan['inventory']['layers']}
    planned = {l['support_id']:[o['product_family'] for o in l['objects']] for l in plan['inventory']['layers']}
    pools = {sid:list(names) for sid,names in planned.items()}
    catalog, paths = {}, {}
    for sid,names in pools.items():
        for name in names:
            asset = Usd.Stage.Open(str(asset_path(name)))
            prim = asset.GetDefaultPrim()
            cache = UsdGeom.XformCache()
            visual = trimesh.util.concatenate([m for _,m in cs.mesh_parts(prim,cache,reference=np.eye(4))])
            proxies = [(b.name,b.mesh.copy()) for b in _proxies(asset,prim,cache)]
            vertices = np.concatenate([visual.vertices]+[m.vertices for _,m in proxies])
            catalog[name] = dict(visual=visual,proxies=proxies,vertices=vertices,
                footprint_xy=np.array(MultiPoint(vertices[:,:2]).convex_hull.exterior.coords[:-1]))
            paths[name] = '/World/RoomEpisode/ContextShelf_'+name
    shelf = '/World/RoomEpisode/Shelf'
    frame = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(shelf))
    inverse = frame.GetInverse()
    slabs = json.loads((LIB/'snack_shelf/manifest.json').read_text())['support_slab_checks']
    supports = {}
    for tier in counts:
        slab = next(s for s in slabs if tier.title()+'Overlay' in s['entry_prim'])
        low,high = np.array(slab['min_m']),np.array(slab['max_m'])
        supports[tier] = dict(polygon_xy_m=[[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]],
            top_z_m=high[2],prim_path=shelf+'/Asset'+slab['entry_prim'].removeprefix('/Asset'))
    low = np.min([np.min(s['polygon_xy_m'],axis=0) for s in supports.values()],axis=0)
    high = np.max([np.max(s['polygon_xy_m'],axis=0) for s in supports.values()],axis=0)
    fixtures = []
    cache = UsdGeom.XformCache()
    for p in stage.Traverse():
        shape = shape_mesh(p)
        if shape is None:
            continue
        visible = UsdGeom.Imageable(p).ComputeVisibility()!='invisible' and UsdGeom.Imageable(p).ComputePurpose() not in ('proxy','guide')
        collider = p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False
        if not (visible or collider):
            continue
        shape.apply_transform(cs.matrix(cache.GetLocalToWorldTransform(p)*inverse))
        body = cs.Body(str(p.GetPath()),shape)
        if body.hi[0]>low[0]-.3 and body.lo[0]<high[0]+.3 and body.hi[1]>low[1]-.3 and body.lo[1]<high[1]+.3:
            fixtures.append(body)
    def pose_item(name,xy,yaw,sid,role):
        c = catalog[name]
        rot = Rotation.from_euler('z',yaw,degrees=True)
        xyz = np.r_[xy,supports[sid]['top_z_m']+.0005-rot.apply(c['vertices'])[:,2].min()]
        quat = rot.as_quat()[[3,0,1,2]]
        tf = cs.pose(xyz,quat)
        path = paths[name]
        bodies = [cs.Body(path+'#visible',c['visual'],tf)]+[cs.Body(path+'#proxy'+str(i),m,tf) for i,(_,m) in enumerate(c['proxies'])]
        return dict(product_family=name,prim_path=path,support_id=sid,xyz_m=xyz.tolist(),wxyz=quat.tolist(),
            yaw_deg=float(yaw),collision_paths=[b.name for b in bodies]),bodies,MultiPoint(cs.points(c['vertices'],tf)[:,:2]).convex_hull
    print('PACKING',question_id,counts,flush=True)
    placed,metrics,receipt = sample_stock(plan['placement_policy'],plan['geometry_only_reservations'],seed,
        catalog,supports,fixtures,counts,pools,pose_item,_separated,planned,sample_scene)
    approved = Usd.Stage.Open(str(REG/'room_demo_v6/room.usda'))
    flat = approved.Flatten()
    prototypes = json.loads((REG/'room_demo_v6/composition.json').read_text())['objects']
    rows = []
    for pose,_,_ in placed:
        name,path = pose['product_family'],pose['prim_path']
        if name in NEW_PRODUCTS:
            UsdGeom.Xform.Define(stage,path).GetPrim().GetReferences().AddReference(str(asset_path(name)))
        else:
            prototype = next(r for r in prototypes if r['id']==name)
            Sdf.CreatePrimInLayer(stage.GetRootLayer(),path)
            assert Sdf.CopySpec(flat,prototype['path'],stage.GetRootLayer(),path)
        for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
            rel = p.GetRelationship('material:binding:physics')
            for target in rel.GetTargets() if rel else ():
                if not stage.GetPrimAtPath(target):
                    if not approved.GetPrimAtPath(target):
                        raise ValueError('Missing authored material '+str(target))
                    Sdf.CreatePrimInLayer(stage.GetRootLayer(),target)
                    assert Sdf.CopySpec(flat,target,stage.GetRootLayer(),target)
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(Gf.Matrix4d(*cs.pose(pose['xyz_m'],pose['wxyz']).T.reshape(-1).tolist())*frame)
        rows.append(dict(id=name,path=path,source='snack_shelf',instance_id=f"shelf/{pose['support_id']}/{name}/01",task_role='touch_candidate'))
    manifest['objects'] = [r for r in manifest['objects'] if r['path'] not in old_stock]+rows
    station_frame = initial_station.shelf_frame(stage,manifest)
    measured = station_geometry.geometry(stage,manifest)
    targets = plan['robot_initial']['joint_positions']
    rejected = []
    for candidate in initial_station.sample_candidates(station_frame,seed):
        reason = station_geometry.rejection(stage,measured,targets,candidate['position_xy'],candidate['yaw_rad'])
        if reason:
            rejected.append(reason)
        else:
            break
    else:
        raise ValueError('No admitted station in unchanged ranges: '+str(rejected))
    base = np.eye(4)
    base[:3,:3] = Rotation.from_euler('z',candidate['yaw_rad']).as_matrix()
    base[:3,3] = [*candidate['position_xy'],0.]
    links = fk(stage,targets,base)
    # Static visualization of source joint coordinates, not a physical rollout.
    for path,world in sorted(links.items(),key=lambda item:item[0].count('/')):
        prim = stage.GetPrimAtPath(path)
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
        xf = UsdGeom.Xformable(prim)
        reset = xf.GetResetXformStack()
        matrix = Gf.Matrix4d(*world.T.reshape(-1).tolist())
        xf.MakeMatrixXform().Set(matrix if reset else matrix*parent.GetInverse())
        xf.SetResetXformStack(reset)
    cache = UsdGeom.XformCache()
    assert all(np.allclose(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T,m) for p,m in links.items())
    bbox = UsdGeom.BBoxCache(0,['default','render'])
    for r in manifest['objects']:
        extent = bbox.ComputeWorldBound(stage.GetPrimAtPath(r['path'])).ComputeAlignedRange()
        r['visible_bounds_world_m'] = [list(extent.GetMin()),list(extent.GetMax())]
    manifest.update(scene=str(scene),seed=seed,tasks=[],static_scene_only=True,native_physics_validated=False)
    for row in manifest['objects']:
        row.pop('task_id',None)
    stage.GetRootLayer().Save()
    manifest['scene_sha256'] = hashlib.sha256(scene.read_bytes()).hexdigest()
    save(output/'composition.json',manifest)
    save(output/'initial_receipt.json',dict(question_id=question_id,seed=seed,variant=plan['robot_initial']['variant'],
        inventory_counts=counts,metrics=metrics,placement=receipt,station=candidate,rejections=rejected,
        joint_positions=targets,robot_room_geometry_checked=True,robot_self_collision_validated=False,
        static_fk_only=True,native_physics_validated=False,shared_runtime_registered=False,
        source_scene=base_scene,source_scene_sha256=hashlib.sha256(Path(base_scene).read_bytes()).hexdigest()))
    question = next(q for q in json.loads((BANK_DIR/'public_questions.json').read_text())['questions'] if q['id']==question_id)
    save(output/'public_question.json',question)
    print('PREPARED',str(output),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question_id')
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    build(args.question_id,args.seed,args.output)
