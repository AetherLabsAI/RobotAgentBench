"""Restore source coffee transforms and source shelf sampling in an isolated layer."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPoint
import trimesh

HERE=Path(__file__).resolve().parent
LIB=Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')
SOURCE=Path('/mnt/agent-bench-data/lingjun/ziche/work/shelf_training_visible360_20260916')
sys.path.extend([str(SOURCE),'/mnt/agent-bench-data/lingjun/ziche/tools/family3'])
import collision_scene as cs
from collision_initial_layout import _proxies, _separated
from omni_randomization import shape_mesh, full_environment_bodies
from random_shelf_scene import sample_scene
from build_room_demo_v1 import exact

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gf(m):return Gf.Matrix4d(*np.asarray(m).reshape(-1).tolist())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(exist_ok=False)
    previous=HERE/'room_demo_v5';scene=a.output/'room.usda'
    stage=Usd.Stage.CreateNew(str(scene));stage.GetRootLayer().subLayerPaths=[str(previous/'room.usda')]
    report=json.loads((previous/'composition.json').read_text())
    plan=json.loads((HERE/'room_preview_plan_seed42.json').read_text())
    receipt={'previous_scene':str(previous/'room.usda'),'previous_sha256':sha(previous/'room.usda'),
             'shared_assets_modified':False,'unrelated_placements_modified':False}
    before={r['path']:np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(r['path']))) for r in report['objects']}
    # Native asset axes are preserved. Use the full source relative transform,
    # not a hand-chosen offset or a bounding-box-center approximation.
    binding_path=LIB.parent/'evaluation/tasks/coffee_machine/resources/source_episodes/1-1/042_right_239.7deg/public_binding.json'
    binding=json.loads(binding_path.read_text());source_machine=np.array(binding['world']['machine']).reshape(4,4)
    dest=before['/World/RoomEpisode/CoffeeMachine'];coffee={}
    for role,name in [('cup','CoffeeCup'),('bag','CoffeeBag')]:
        relative=np.array(binding['world'][role]).reshape(4,4)@np.linalg.inv(source_machine)
        path='/World/RoomEpisode/'+name;world=relative@dest
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(gf(world))
        recovered=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)))@np.linalg.inv(dest)
        assert np.allclose(recovered,relative,atol=1e-10)
        coffee[role]={'source_relative_row_matrix':relative.tolist(),'world_matrix':world.tolist(),
                      'relative_transform_max_error':float(np.max(abs(recovered-relative)))}
    receipt['coffee']={'binding':str(binding_path),'sha256':sha(binding_path),'objects':coffee,
        'method':'Full original cup/bag-to-machine transforms; source 1.1 placement initial state, not continuous cycle validation'}
    # Reuse original random_shelf_scene.sample_scene including XY/yaw sampling,
    # footprint spacing, visible/proxy collision checks and structural clearance.
    catalog={};paths={}
    for layer in plan['shelf']['layers']:
        for i,item in enumerate(layer['objects']):
            name=item['product_family'];path=f'/World/RoomEpisode/Shelf_{layer["support_id"]}_{i}'
            assert name not in paths,'This initial all-category demo has one of each kind'
            paths[name]=path;s=Usd.Stage.Open(str(LIB/name/'asset.usda'));root=s.GetDefaultPrim();cache=UsdGeom.XformCache()
            parts=cs.mesh_parts(root,cache,reference=np.eye(4))
            visual=trimesh.util.concatenate([m for _,m in parts]);proxies=[(b.name,b.mesh.copy()) for b in _proxies(s,root,cache)]
            vertices=np.concatenate([visual.vertices]+[m.vertices for _,m in proxies])
            catalog[name]={'visual':visual,'proxies':proxies,'vertices':vertices,
                          'footprint_xy':np.array(MultiPoint(vertices[:,:2]).convex_hull.exterior.coords[:-1])}
    shelf_path='/World/RoomEpisode/Shelf';center=before[shelf_path][3,:3]
    manifest=json.loads((LIB/'snack_shelf/manifest.json').read_text());supports={}
    for layer in plan['shelf']['layers']:
        sid=layer['support_id'];slab=next(s for s in manifest['support_slab_checks'] if sid.title()+'Overlay' in s['entry_prim'])
        low=np.array(slab['min_m'])+center;high=np.array(slab['max_m'])+center
        supports[sid]={'polygon_xy_m':[[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]],
                       'top_z_m':high[2],'prim_path':shelf_path+'/Asset'+slab['entry_prim'].removeprefix('/Asset')}
    fixtures=full_environment_bodies(stage);cache=UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath(shelf_path)):
        shape=shape_mesh(p)
        if shape is None:continue
        visible=UsdGeom.Imageable(p).ComputeVisibility()!='invisible' and UsdGeom.Imageable(p).ComputePurpose() not in ('proxy','guide')
        collider=p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        if visible or collider:
            shape.apply_transform(cs.matrix(cache.GetLocalToWorldTransform(p)))
            fixtures.append(cs.Body(str(p.GetPath()),shape))
    fixtures=[b for b in fixtures if b.hi[0]>center[0]-.8 and b.lo[0]<center[0]+.8 and b.hi[1]>center[1]-.5 and b.lo[1]<center[1]+.5]
    def pose_item(name,xy,yaw,sid,role):
        c=catalog[name];rot=Rotation.from_euler('z',yaw,degrees=True)
        xyz=np.r_[xy,supports[sid]['top_z_m']+.0005-rot.apply(c['vertices'])[:,2].min()]
        quat=rot.as_quat()[[3,0,1,2]];world=cs.pose(xyz,quat);vertices=cs.points(c['vertices'],world);path=paths[name]
        bodies=[cs.Body(path+'#visible',c['visual'],world)]+[cs.Body(path+'#proxy'+str(i),m,world) for i,(_,m) in enumerate(c['proxies'])]
        fp=MultiPoint(vertices[:,:2]).convex_hull
        row={'product_family':name,'prim_path':path,'support_id':sid,'xyz_m':xyz.tolist(),'wxyz':quat.tolist(),
             'yaw_deg':float(yaw),'collision_paths':[b.name for b in bodies],'bounds_world_m':[vertices.min(0).tolist(),vertices.max(0).tolist()]}
        return row,bodies,fp
    counts={l['support_id']:l['count'] for l in plan['shelf']['layers']}
    pools={l['support_id']:[i['product_family'] for i in l['objects']] for l in plan['shelf']['layers']}
    assert all(len(set(pools[k]))==counts[k] for k in counts),'All required categories must appear'
    placed,metrics=sample_scene(np.random.default_rng(42),catalog,supports,fixtures,counts,pools,pose_item,_separated)
    assert {r['product_family'] for r,_,_ in placed}==set(paths)
    for row,_,_ in placed:
        UsdGeom.Xformable(stage.GetPrimAtPath(row['prim_path'])).MakeMatrixXform().Set(gf(cs.pose(row['xyz_m'],row['wxyz']).T))
    receipt['shelf']={'source':str(SOURCE/'random_shelf_scene.py'),'source_sha256':sha(SOURCE/'random_shelf_scene.py'),
        'source_inventory_sha256':sha(SOURCE/'snack_diversity.py'),'seed':42,'metrics':metrics,
        'placements':[r for r,_,_ in placed],'all_required_present':True,'support_geometry':'current all_assets/snack_shelf',
        'fixtures_checked':len(fixtures),'method':'Original generic sampler, no focus/targeted override; current shared-asset visible and collider geometry'}
    changed=set(paths.values())|{'/World/RoomEpisode/CoffeeCup','/World/RoomEpisode/CoffeeBag'}
    for row in report['objects']:
        if row['path'] not in changed:
            assert np.array_equal(before[row['path']],np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(row['path'])))),row['path']
        row['visible_bounds_world_m']=exact(stage,row['path']).tolist()
    report.update(scene=str(scene),status='corrected_initialization_requires_native_validation',placement_repair=receipt)
    report['asset_entry_hashes'][str(binding_path)]=sha(binding_path)
    stage.GetRootLayer().Save();save(a.output/'composition.json',report);save(a.output/'placement_repair.json',receipt)
    print(json.dumps({'status':report['status'],'coffee':coffee,'shelf':metrics},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
