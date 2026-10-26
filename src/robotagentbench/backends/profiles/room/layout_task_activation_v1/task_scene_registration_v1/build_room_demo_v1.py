"""Episode-local room composition. Never edits the source layout or all_assets."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from openend.tasks.trash_transfer.scene_semantics import mark_object, bin_roles
from openend.tasks.trash_transfer.paper_variants import annotate_object
from openend.tasks.trash_transfer.paper_variant_usd import apply_variant, plan_variant
LIB=Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')
BASE=Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def T(v):return Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float,v)))
def R(deg,axis=(0,0,1)):return Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis),float(deg)))
def matrix(v):return Gf.Matrix4d(*np.asarray(v).reshape(-1).tolist())
def exact(stage,path):
    points=[];xf=UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        im=UsdGeom.Imageable(p)
        if not im:continue
        if im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render'):continue
        if p.IsA(UsdGeom.Mesh):
            vs=UsdGeom.Mesh(p).GetPointsAttr().Get()
        elif p.IsA(UsdGeom.Cube):
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2
            vs=[(x,y,z) for x in (-h,h) for y in (-h,h) for z in (-h,h)]
        else:continue
        if vs is not None and len(vs):
            pts=np.c_[np.array(vs),np.ones(len(vs))]@np.array(xf.GetLocalToWorldTransform(p))
            points.extend(pts[:,:3])
    assert points,path
    return np.array([np.min(points,axis=0),np.max(points,axis=0)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((HERE/'room_preview_plan_seed42.json').read_text())
    inventory=json.loads((HERE/'room_demo_inventory_v1.json').read_text())
    source_hash=sha(BASE)
    out=args.output/'room.usda';stage=Usd.Stage.CreateNew(str(out));stage.GetRootLayer().subLayerPaths=[str(BASE)]
    rows=[];removed=[];hashes={};repairs=[]
    for path in ['/World/Assets/ShelvingUnit','/World/Assets/ShelfItems','/World/Assets/Tier2Items',
                 '/World/Assets/CoffeeGrinder','/World/Assets/AirFryer','/World/FoodTray','/World/Cutlery','/World/MealDiverse']:
        if stage.GetPrimAtPath(path):stage.OverridePrim(path).SetActive(False);removed.append(path)
    # The bare layout is missing the L1 task's seat support adapter. Reuse its exact
    # authored dimensions/material rather than inventing a replacement collider.
    sofa_support_source=Path('/home/lingjun/robot_agent_arena/api-protocol/assets/house_scene_l1/demo_sim/scene_room_baseline.usda')
    seat=stage.DefinePrim('/World/Assets/SeatOverlay')
    seat.GetReferences().AddReference(str(sofa_support_source),'/World/Assets/SeatOverlay')
    seat.GetRelationship('material:binding:physics').SetTargets(['/World/PhysicsMaterials/GripMat'])
    hashes[str(sofa_support_source)]=sha(sofa_support_source)
    # Existing microwave is fixed furniture; relocate its whole hierarchy without duplicating it.
    placements={p['part_id']:p['placement'] for p in plan['kitchen']['parts']}
    def zone_matrix(part):
        p=placements[part];fx,fy=p['front'];x,y=p['origin']
        return Gf.Matrix4d(fy,-fx,0,0,fx,fy,0,0,0,0,1,0,x,y,p['top'],1)
    def asset(id,name,m):
        path='/World/RoomEpisode/'+name
        node=UsdGeom.Xform.Define(stage,path);child=stage.DefinePrim(path+'/Asset')
        entry=LIB/id/'asset.usda';assert entry.is_file(),id
        child.GetReferences().AddReference(str(entry));hashes[str(entry)]=sha(entry)
        for q in Usd.PrimRange(child):
            if q.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(q).GetResetXformStack():
                UsdGeom.Xformable(q).SetResetXformStack(False)
        node.MakeMatrixXform().Set(m)
        child.SetCustomDataByKey('roomAssetId',id)
        rows.append(mark_object(dict(id=id,path=path,entry=str(entry))))
        return path
    def center_bottom(id,name,xyz,rot=None,paper_variant=None):
        rot=rot or Gf.Matrix4d(1)
        path=asset(id,name,rot)
        if paper_variant is not None:
            if id != 'paper_ball':raise ValueError('Only paper can use a paper variant')
            apply_variant(stage,path+'/Asset',paper_variant)
            rows[-1]=annotate_object(rows[-1],paper_variant)
        b=exact(stage,path);origin=np.array([(b[0,0]+b[1,0])/2,(b[0,1]+b[1,1])/2,b[0,2]])
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(rot*T(np.asarray(xyz)-origin))
        return path
    # The approved all_assets shelf includes its four physical support slabs.
    shelf_center=np.array([-2.3690211510018,2.252956604618633,0.])
    asset('snack_shelf','Shelf',T(shelf_center))
    manifest=json.loads((LIB/'snack_shelf/manifest.json').read_text())
    for layer in plan['shelf']['layers']:
        slab=next(s for s in manifest['support_slab_checks'] if layer['support_id'].title()+'Overlay' in s['entry_prim'])
        low=np.array(slab['min_m'])+shelf_center;high=np.array(slab['max_m'])+shelf_center
        # Same one-row footprint gap rule as the approved inventory; no asset scaling.
        objects=layer['objects'];gap=layer['minimum_footprint_gap_m']
        widths=[inventory['assets'][o['product_family']]['exact_bounds'][1][0]-inventory['assets'][o['product_family']]['exact_bounds'][0][0] for o in objects]
        total=sum(widths)+gap*(len(objects)-1)
        assert total<=high[0]-low[0]-.04,('shelf row does not fit',total,high[0]-low[0])
        cursor=(low[0]+high[0]-total)/2
        for i,(o,w) in enumerate(zip(objects,widths)):
            p=center_bottom(o['product_family'],f'Shelf_{layer["support_id"]}_{i}',[cursor+w/2,(low[1]+high[1])/2,high[2]+.001])
            b=exact(stage,p);assert np.all(b[0,:2]>=low[:2]) and np.all(b[1,:2]<=high[:2]);cursor+=w+gap
    # Use the curved seat's highest support under the complete footprint, not a flat
    # nominal 45.2 cm plane (which embeds thin props into parts of the cushions).
    triangles=[];xf=UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Assets/Sofa')):
        if not p.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(p);vs=np.array(mesh.GetPointsAttr().Get())
        vs=(np.c_[vs,np.ones(len(vs))]@np.array(xf.GetLocalToWorldTransform(p)))[:,:3]
        indices=np.array(mesh.GetFaceVertexIndicesAttr().Get());offset=0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            f=indices[offset:offset+count];offset+=count
            for j in range(1,count-1):triangles.append(vs[[f[0],f[j],f[j+1]]])
    tri=np.array(triangles);aa,bb,cc=tri[:,0],tri[:,1],tri[:,2];e0=bb-aa;e1=cc-aa
    den=e0[:,0]*e1[:,1]-e0[:,1]*e1[:,0];valid=abs(den)>1e-12
    def seat_height(bounds):
        heights=[]
        for x in np.linspace(bounds[0,0],bounds[1,0],9):
            for y in np.linspace(bounds[0,1],bounds[1,1],9):
                q=np.array([x,y])-aa[:,:2];u=np.zeros(len(tri));v=u.copy()
                u[valid]=(q[valid,0]*e1[valid,1]-q[valid,1]*e1[valid,0])/den[valid]
                v[valid]=(e0[valid,0]*q[valid,1]-e0[valid,1]*q[valid,0])/den[valid]
                z=aa[:,2]+u*e0[:,2]+v*e1[:,2];hit=valid&(u>=0)&(v>=0)&(u+v<=1)&(z>.35)
                if np.any(hit):heights.append(float(max(z[hit])))
        assert heights and max(heights)<.6,('footprint intersects sofa structural keepout',heights)
        return max(heights)+.005
    # Sofa uses the approved XY footprint positions and authored rest attitude.
    for p in plan['sofa']['placements']:
        path=center_bottom(p['id'],'Sofa_'+p['id'],[-.3503563938+p['y'],-.8710-p['x'],.453],R(p['yaw_deg']-90))
        bounds=exact(stage,path);z=seat_height(bounds)
        node=UsdGeom.Xformable(stage.GetPrimAtPath(path));node.MakeMatrixXform().Set(node.GetLocalTransformation()*T([0,0,z-bounds[0,2]]))
    # The small-table sampler supplies exact world delta matrices, including origin offsets.
    for key,p in plan['small_table']['placements'].items():
        id='remote_tv' if key=='remote' else key
        path=asset(id,'SmallTable_'+id,matrix(np.array(p['world_delta_column_matrix']).T))
        b=exact(stage,path)
        assert abs(b[0,2]-plan['small_table']['surface']['z_m'])<1e-5,(id,b)
    table=plan['coffee_table']
    for i,p in enumerate(table['objects']):
        rot=R(math.degrees(p['rollRad']),(1,0,0))*R(math.degrees(p['yaw'])-90)
        center_bottom(p['assetId'],f'Table_{i}_{p["assetId"]}',p['world_center']+[.451],rot,
                      paper_variant=plan_variant(p))
        if p['assetId']=='zbar_t':
            # Producer knows this instance is the approved empty wrapper template.
            rows[-1]['trash_category']='snack'
    cloth=np.array(table['cloth']);center_bottom('tablecloth_dirty','Tablecloth',[*cloth.mean(axis=0),.452])
    band=np.array(table['band']);center_bottom('japanese_tray','WoodenTray',[*band.mean(axis=0),.451])
    # Kitchen transforms: native fryer +X faces the operator, station +Y is front.
    asset('air_fryer','Fryer',R(90)*zone_matrix('fryer'))
    # Preserve task2.1 plate/cake relative transforms, normalizing the station placement only.
    episode=json.loads((LIB/'IMPORT_ROOM_ACCESSORIES_V1.json').read_text())['source_task2_episode']
    wf=matrix(episode['world']['fryer']);wp=matrix(episode['world']['plate'])
    plate_local=wp*wf.GetInverse()*R(90)
    # Match the selected standard station's plate center without changing cake/plate relationships.
    support=next(p for p in plan['kitchen']['parts'] if p['part_id']=='fryer')['physical_support_footprints'][1]
    uv=(np.array(support['min'])+np.array(support['max']))/2
    plate_xyz=np.array(plate_local.ExtractTranslation());offset=np.array([uv[0],uv[1],0])-plate_xyz
    rel=wf.GetInverse()*R(90)*T(offset)*zone_matrix('fryer')
    for id,key,name in [('paper_plate_task2','plate','FryerPlate'),('cake_plain_yellow','target','CakeYellow'),('cake_frosted_white','other','CakeWhite')]:
        if key=='plate':
            # Imported plate already retains the source world rotation, removed translation only.
            m=T(np.array(wp.ExtractTranslation()))*rel
        else:m=matrix(episode['world'][key])*rel
        asset(id,name,m)
    mic=stage.GetPrimAtPath('/World/Assets/DiningTableMicrowave')
    old=UsdGeom.XformCache().GetLocalToWorldTransform(mic);old_origin=np.array(old.ExtractTranslation())
    # Existing microwave is already oriented toward -X at origin (2.33,-1.34,.92).
    p=placements['microwave'];delta=np.array([p['origin'][0],p['origin'][1],p['top']])-old_origin
    UsdGeom.Xformable(mic).MakeMatrixXform().Set(old*T(delta))
    # Coffee source is Y-up, native +Z front. Its rigid body and button joint are retained.
    coffee_basis=Gf.Matrix4d(-1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1)
    asset('coffee_machine','CoffeeMachine',coffee_basis*zone_matrix('coffee'))
    for id,name,u in [('coffee_cup_blue_compact','CoffeeCup',.28),('coffee_bag','CoffeeBag',-.28)]:
        target=zone_matrix('coffee').Transform(Gf.Vec3d(u,0,.001))
        center_bottom(id,name,list(target),coffee_basis*R(math.degrees(math.atan2(placements['coffee']['front'][1],placements['coffee']['front'][0]))-90))
    # Undo the source tray heading, then apply the selected station frame (long side = u).
    source_tray=next(r for r in inventory['world'] if r['path']=='/World/FoodTray')
    orig=matrix(source_tray['matrix']);rot=Gf.Matrix4d(orig);rot.SetTranslateOnly(Gf.Vec3d(0))
    tray_rot=rot.GetInverse()*zone_matrix('tray')
    asset('meal_tray_large','MealTray',tray_rot)
    source_stage=Usd.Stage.Open(str(BASE));xf=UsdGeom.XformCache()
    holder=xf.GetLocalToWorldTransform(source_stage.GetPrimAtPath('/World/Cutlery/Holder'))
    relholder=holder*orig.GetInverse()
    holder_t=list(relholder.ExtractTranslation());holder_t[2]=.0003
    # Library holder retains its original rotation, so remove only the common tray heading.
    asset('cutlery_holder','CutleryHolder',rot.GetInverse()*T(holder_t)*zone_matrix('tray'))
    for name,id in [('knife_0','meal_knife'),('fork_1','meal_fork'),('fork_2','meal_fork')]:
        original=xf.GetLocalToWorldTransform(source_stage.GetPrimAtPath('/World/Cutlery/'+name))
        xyz=np.array((original*orig.GetInverse()).ExtractTranslation())
        asset(id,'Cutlery_'+name,rot.GetInverse()*T(xyz)*zone_matrix('tray'))
    # Rebind source world-scoped physics materials lost across USD reference boundaries.
    # This restores the authored material; it does not change friction/restitution values.
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/RoomEpisode')):
        rel=p.GetRelationship('material:binding:physics')
        if not rel or rel.GetTargets():continue
        for spec in rel.GetPropertyStack():
            targets=list(spec.targetPathList.GetAppliedItems())
            if targets and all(str(t).startswith('/World/PhysicsMaterials/') and stage.GetPrimAtPath(t) for t in targets):
                rel.SetTargets(targets);repairs.append(dict(path=str(p.GetPath()),targets=list(map(str,targets))));break
    # Record exact instantiated geometry; no physics or task success claim at this stage.
    from replace_room_refrigerator_v1 import apply as replace_refrigerator
    rows.append(replace_refrigerator(stage))
    from initialize_room_tote_v1 import apply as initialize_tote
    rows.append(initialize_tote(stage))
    from initialize_room_fridge_boxes_v1 import apply as initialize_fridge_boxes
    rows.extend(initialize_fridge_boxes(stage, seed=int(plan['seed'])))
    from fridge_to_microwave_v1.support import apply as initialize_task9_support
    rows.extend(initialize_task9_support(stage))
    for row in rows:row['visible_bounds_world_m']=exact(stage,row['path']).tolist()
    for id in plan['singleton_allocations']:
        assert sum(r['id']==id for r in rows)==1,id
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Joint):
            for relname in ('physics:body0','physics:body1'):
                for target in p.GetRelationship(relname).GetTargets():
                    assert stage.GetPrimAtPath(target) and stage.GetPrimAtPath(target).IsActive(),(p.GetPath(),target)
    stage.GetRootLayer().Save();assert sha(BASE)==source_hash
    report=dict(status='scene_composed_requires_native_validation',scene=str(out),source=str(BASE),source_sha256=source_hash,
                asset_entry_hashes=hashes,all_added_assets_from_all_assets=False,source_unchanged=True,
                fridge_box_asset_source='existing_api_resources/shared_assets_standard_packages',
                shelf_source='all_assets/snack_shelf',removed_episode_only=removed,objects=rows,restored_physics_material_bindings=repairs,
                restored_support={'source':str(sofa_support_source),'prim':'/World/Assets/SeatOverlay','dimensions_unchanged':True},
                control_hz=20,physics_validation=False,video_created=False,
                limitations=['Not manipulation success; coffee accessory counter positions require task-specific continuation validation',
                             'Cloth is the approved static source asset, not a deformable cloth simulation',
                             'zbar_t is the previously displayed packaging template, not an empty-wrapper simulation'])
    report.update(trash_semantics_version=1,task_destinations=bin_roles(stage))
    (args.output/'composition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'status':report['status'],'objects':len(rows),'scene':str(out)}))

if __name__=='__main__':main()
