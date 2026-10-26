"""Compose isolated alternate-layout previews from approved episode-local assets."""
import json, hashlib
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from build_room_demo_v1 import exact, matrix, T

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def world(s,path):return UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath(path))
def zone(p):
    fx,fy=p['front'];x,y=p['origin']
    return Gf.Matrix4d(fy,-fx,0,0,fx,fy,0,0,0,0,1,0,x,y,p['top'],1)

def main():
    plans=json.loads((HERE/'other_layouts_plans_v1.json').read_text())
    oldplan=json.loads((HERE/'room_preview_plan_seed42.json').read_text())
    oldparts={p['part_id']:p['placement'] for p in oldplan['kitchen']['parts']}
    approved=Usd.Stage.Open(str(HERE/'room_demo_v6/room.usda'))
    flat=approved.Flatten()
    original=Usd.Stage.Open(str(ROOT/'ID_01_LIVING_STRAIGHT.usdc'))
    oldreport=json.loads((HERE/'room_demo_v6/composition.json').read_text())
    output=HERE/'other_layouts_v3';output.mkdir(exist_ok=False)
    summaries=[]
    for key,entry in plans.items():
        n=int(key);folder=output/f'layout{n:02d}';folder.mkdir()
        base=ROOT/f'ID_{n:02d}_LIVING_STRAIGHT.usdc';before=sha(base)
        stage=Usd.Stage.CreateNew(str(folder/'room.usda'));stage.GetRootLayer().subLayerPaths=[str(base)]
        stage.SetEditTarget(stage.GetRootLayer())
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),'/World/RoomEpisode')
        Sdf.CreatePrimInLayer(stage.GetRootLayer(),'/World/Assets/SeatOverlay')
        # Copy ONLY the episode subtree. No layout-01 furniture or environment layer.
        Sdf.CopySpec(flat,'/World/RoomEpisode',stage.GetRootLayer(),'/World/RoomEpisode')
        Sdf.CopySpec(flat,'/World/Assets/SeatOverlay',stage.GetRootLayer(),'/World/Assets/SeatOverlay')
        for path in oldreport['removed_episode_only']:
            if stage.GetPrimAtPath(path):stage.OverridePrim(path).SetActive(False)
        parts={p['part_id']:dict(p['placement']) for p in entry['plan']['parts']}
        support_evidence={}
        for name,p in parts.items():
            bounds=exact(stage,p['support']);p['top']=float(bounds[1,2])
            support_evidence[name]={'path':p['support'],'actual_top_m':p['top'],'group':p['group'],'origin':p['origin'],'front':p['front']}
        deltas={name:zone(oldparts[name]).GetInverse()*zone(p) for name,p in parts.items()}
        furniture={}
        for name in ['Sofa','CoffeeTable','SmallCoffeeTable','ShelvingUnit']:
            path='/World/Assets/'+name
            # ShelvingUnit was deactivated above: source target stage gives its transform.
            target=Usd.Stage.Open(str(base)) if name=='ShelvingUnit' else stage
            delta=world(original,path).GetInverse()*world(target,path)
            linear=np.array(delta)[:3,:3]
            assert np.allclose(linear@linear.T,np.eye(3),atol=1e-6),('Furniture dimensions changed; needs resampling',name)
            furniture[name]=delta
        rows=[]
        for row in oldreport['objects']:
            if row['id'] in ('refrigerator','tote_bag','beef_box','salad_box','leftover_box','microwave_support_plate','microwave_plate_stand'):continue  # Rebind each layout's own anchor below.
            path=row['path'];name=path.rsplit('/',1)[-1]
            if name.startswith('Sofa_'):delta=furniture['Sofa']
            elif name.startswith('SmallTable_'):delta=furniture['SmallCoffeeTable']
            elif name.startswith('Shelf'):delta=furniture['ShelvingUnit']
            elif name.startswith('Table_') or name in ('Tablecloth','WoodenTray'):delta=furniture['CoffeeTable']
            elif name.startswith(('Fryer','Cake')):delta=deltas['fryer']
            elif name.startswith('Coffee'):delta=deltas['coffee']
            elif name.startswith(('MealTray','Cutlery')):delta=deltas['tray']
            else:raise AssertionError(name)
            UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(world(approved,path)*delta)
            rows.append({**row,'visible_bounds_world_m':exact(stage,path).tolist()})
        path='/World/Assets/SeatOverlay'
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(world(approved,path)*furniture['Sofa'])
        path='/World/Assets/DiningTableMicrowave'
        # Source asset front is -X; full heading must follow the selected operator edge.
        old=world(original,path);canonical=dict(front=[-1,0],origin=list(old.ExtractTranslation())[:2],top=float(old.ExtractTranslation()[2]))
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(old*zone(canonical).GetInverse()*zone(parts['microwave']))
        coffee={}
        for name in ('CoffeeCup','CoffeeBag'):
            path='/World/RoomEpisode/'+name
            a=world(stage,path)*world(stage,'/World/RoomEpisode/CoffeeMachine').GetInverse()
            b=world(approved,path)*world(approved,'/World/RoomEpisode/CoffeeMachine').GetInverse()
            error=float(np.abs(np.array(a)-np.array(b)).max());assert error<1e-9
            coffee[name]=error
        for singleton in oldplan['singleton_allocations']:
            assert sum(r['id']==singleton for r in rows)==1
        # Fail closed for broken articulation references after composition.
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Joint):
                for rel in ('physics:body0','physics:body1'):
                    for t in p.GetRelationship(rel).GetTargets():assert stage.GetPrimAtPath(t),(p.GetPath(),t)
        from replace_room_refrigerator_v1 import apply as replace_refrigerator
        fridge=replace_refrigerator(stage)
        fridge['visible_bounds_world_m']=exact(stage,fridge['path']).tolist()
        rows.append(fridge)
        from initialize_room_tote_v1 import apply as initialize_tote
        rows.append(initialize_tote(stage))
        from initialize_room_fridge_boxes_v1 import apply as initialize_fridge_boxes
        rows.extend(initialize_fridge_boxes(stage, seed=int(entry['plan']['request']['seed'])))
        from fridge_to_microwave_v1.support import apply as initialize_task9_support
        rows.extend(initialize_task9_support(stage))
        stage.GetRootLayer().Save();assert sha(base)==before
        report=dict(layout_id=n,seed=entry['plan']['request']['seed'],scene=str(folder/'room.usda'),
                    status='composed_static_preview_not_native_validated',source_layout=str(base),source_sha256=before,
                    source_unchanged=True,objects=rows,supports=support_evidence,coffee_relative_transform_errors=coffee,
                    shelf_method='Approved source-randomized all_assets shelf; same valid arrangement retained for layout comparison',
                    planning_attempts=entry['attempts'],physical_settling_test=False,manipulation_test=False,
                    limitations=['Planar workspace checks only; no new native settling or manipulation run',
                                 'Living props follow the unchanged-size furniture; not independently resampled',
                                 'Overhead clearance excluded as previously requested'])
        (folder/'composition.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        summaries.append({k:v for k,v in report.items() if k!='objects'})
        print('COMPOSED',n,len(rows),flush=True)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
