"""Re-sample living surfaces, preserving kitchen and fixed cloth/tray placements."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,Gf
from build_room_demo_v1 import exact,T,R,matrix
from openend.tasks.trash_transfer.scene_semantics import mark_object, bin_roles
from openend.tasks.trash_transfer.paper_variants import annotate_object
from openend.tasks.trash_transfer.paper_variant_usd import apply_variant, plan_variant
HERE=Path(__file__).resolve().parent
LIB=Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets')
def world(s,p):return UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath(p))
def setworld(s,p,m):UsdGeom.Xformable(s.GetPrimAtPath(p)).MakeMatrixXform().Set(m*world(s,str(s.GetPrimAtPath(p).GetParent().GetPath())).GetInverse())
def center_bottom(s,path,xyz,rot):
    setworld(s,path,rot)
    b=exact(s,path);origin=np.r_[(b[0,:2]+b[1,:2])/2,b[0,2]]
    setworld(s,path,rot*T(np.asarray(xyz)-origin))
def seat_height_function(stage):
    triangles=[];cache=UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Assets/Sofa')):
        if not p.IsA(UsdGeom.Mesh):continue
        m=UsdGeom.Mesh(p);v=np.array(m.GetPointsAttr().Get())
        v=(np.c_[v,np.ones(len(v))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:3]
        idx=np.array(m.GetFaceVertexIndicesAttr().Get());offset=0
        for count in m.GetFaceVertexCountsAttr().Get():
            f=idx[offset:offset+count];offset+=count
            for j in range(1,count-1):triangles.append(v[[f[0],f[j],f[j+1]]])
    tri=np.array(triangles);aa,bb,cc=tri[:,0],tri[:,1],tri[:,2];e0=bb-aa;e1=cc-aa
    den=e0[:,0]*e1[:,1]-e0[:,1]*e1[:,0];valid=abs(den)>1e-12
    def height(b):
        heights=[]
        for x in np.linspace(b[0,0],b[1,0],9):
            for y in np.linspace(b[0,1],b[1,1],9):
                q=np.array([x,y])-aa[:,:2];u=np.zeros(len(tri));v=u.copy()
                u[valid]=(q[valid,0]*e1[valid,1]-q[valid,1]*e1[valid,0])/den[valid]
                v[valid]=(e0[valid,0]*q[valid,1]-e0[valid,1]*q[valid,0])/den[valid]
                z=aa[:,2]+u*e0[:,2]+v*e1[:,2];hit=valid&(u>=0)&(v>=0)&(u+v<=1)&(z>.35)
                if np.any(hit):heights.append(float(max(z[hit])))
        assert heights and max(heights)<.6,('sofa structural keepout',heights)
        return max(heights)+.005
    return height

def main():
    run=HERE/'other_layouts_v4';run.mkdir(exist_ok=False)
    summaries=[];plans=json.loads((HERE/'other_living_plans_v1.json').read_text())
    baseline=Usd.Stage.Open('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
    for plan in plans:
        n=plan['layout_id'];previous=HERE/f'other_layouts_v3/layout{n:02d}'
        old=Usd.Stage.Open(str(previous/'room.usda'));report=json.loads((previous/'composition.json').read_text())
        folder=run/f'layout{n:02d}';folder.mkdir();s=Usd.Stage.CreateNew(str(folder/'room.usda'))
        s.GetRootLayer().subLayerPaths=[str(previous/'room.usda')];s.SetEditTarget(s.GetRootLayer())
        old_world={r['path']:world(old,r['path']) for r in report['objects']}
        def delta(name):
            p='/World/Assets/'+name;d=world(baseline,p).GetInverse()*world(s,p)
            assert np.allclose(np.array(d)[:3,:3],np.eye(3),atol=1e-8),'Rotated/changed support requires a new geometry adapter'
            return d
        dsofa,dsmall,dtable=delta('Sofa'),delta('SmallCoffeeTable'),delta('CoffeeTable')
        height=seat_height_function(s);changed=[]
        for p in plan['sofa']['placements']:
            path='/World/RoomEpisode/Sofa_'+p['id']
            xyz=[-.3503563938+p['y'],-.8710-p['x'],.453]
            center_bottom(s,path,xyz,R(p['yaw_deg']-90));setworld(s,path,world(s,path)*dsofa)
            b=exact(s,path);setworld(s,path,world(s,path)*T([0,0,height(b)-b[0,2]]));changed.append(path)
        for key,p in plan['small_table']['placements'].items():
            path='/World/RoomEpisode/SmallTable_'+('remote_tv' if key=='remote' else key)
            setworld(s,path,matrix(np.array(p['world_delta_column_matrix']).T)*dsmall);changed.append(path)
        rows=[]
        for r in report['objects']:
            if r['path'].rsplit('/',1)[-1].startswith('Table_'):s.OverridePrim(r['path']).SetActive(False)
            else:rows.append(mark_object(r))
        for i,p in enumerate(plan['coffee_table']['objects']):
            path=f'/World/RoomEpisode/TableRandom_{i}_{p["assetId"]}'
            UsdGeom.Xform.Define(s,path);child=s.DefinePrim(path+'/Asset');entry=LIB/p['assetId']/'asset.usda'
            child.GetReferences().AddReference(str(entry))
            for q in Usd.PrimRange(child):
                if q.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(q).GetResetXformStack():UsdGeom.Xformable(q).SetResetXformStack(False)
            rot=R(math.degrees(p['rollRad']),(1,0,0))*R(math.degrees(p['yaw'])-90)
            variant=plan_variant(p)
            if variant is not None:apply_variant(s,path+'/Asset',variant)
            center_bottom(s,path,p['world_center']+[.451],rot);setworld(s,path,world(s,path)*dtable)
            rows.append(mark_object(dict(id=p['assetId'],path=path,entry=str(entry)),
                                    empty_packaging=p['assetId']=='zbar_t'))
            if variant is not None:rows[-1]=annotate_object(rows[-1],variant)
        preserved=[]
        for r in rows:
            p=r['path']
            if p in old_world and p not in changed:
                assert np.allclose(np.array(world(s,p)),np.array(old_world[p]),atol=1e-12),p
                preserved.append(p)
            r['visible_bounds_world_m']=exact(s,p).tolist()
        evidence=[]
        for p in changed:
            a=np.array(world(s,p));b=np.array(old_world[p]);d=float(np.linalg.norm(a[3,:3]-b[3,:3]))
            assert d>.005,(p,'not re-randomized')
            evidence.append(dict(path=p,translation_change_m=d,old_world=b.tolist(),new_world=a.tolist()))
        for id in plan['singleton_allocations']:assert sum(r['id']==id for r in rows)==1,id
        assert len(plan['sofa']['placements'])==6 and len(changed)==9
        s.GetRootLayer().Save()
        report.update(scene=str(folder/'room.usda'),objects=rows,living_seed=plan['seed'],
            status='fresh_living_randomization_static_preview_not_native_validated',
            living_randomization=dict(seed=plan['seed'],sofa_count=6,small_table_count=3,trash_count=len(plan['coffee_table']['objects']),
                changed_objects=evidence,preserved_objects=preserved,cloth_and_wooden_tray_fixed=True,global_singletons_preserved=True),
            limitations=['New sofa/small-table/trash samples; unchanged fixed cloth/wooden tray and kitchen','No native settling or task execution in this static preview'])
        report.update(trash_semantics_version=1,task_destinations=bin_roles(s))
        (folder/'composition.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        (folder/'living_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        # The kitchen audit is inherited only after exact preservation assertions above.
        audit=json.loads((previous/'mesh_audit.json').read_text());audit['kitchen_geometry_unchanged_from']=str(previous/'room.usda')
        (folder/'mesh_audit.json').write_text(json.dumps(audit,indent=2))
        summaries.append({k:v for k,v in report.items() if k!='objects'})
        print('RANDOMIZED',n,'sofa 6 small 3 trash',len(plan['coffee_table']['objects']),flush=True)
    (run/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
