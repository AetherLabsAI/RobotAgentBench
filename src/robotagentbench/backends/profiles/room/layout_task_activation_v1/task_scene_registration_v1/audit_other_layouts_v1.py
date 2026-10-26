"""Independent actual-mesh planar audit; reports limitations, never native pass."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
from shapely.geometry import MultiPoint,Polygon
from shapely.ops import unary_union
HERE=Path(__file__).resolve().parent
data=json.loads((HERE.parent/'web/random-layouts-v1.json').read_text())
def footprint(stage,path):
    points=[];cache=UsdGeom.XformCache()
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not p.IsA(UsdGeom.Mesh):continue
        im=UsdGeom.Imageable(p)
        if im.ComputeVisibility()=='invisible' or im.ComputePurpose() not in ('default','render'):continue
        vs=UsdGeom.Mesh(p).GetPointsAttr().Get()
        if vs:points.extend((np.c_[np.array(vs),np.ones(len(vs))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:2])
    return MultiPoint(points).convex_hull
out=[]
for record in json.loads((HERE/'other_layouts_v3/summary.json').read_text()):
    s=Usd.Stage.Open(record['scene']);layout=next(l for l in data['layouts'] if l['id']==record['layout_id'])
    families={'fryer':['Fryer','FryerPlate','CakeYellow','CakeWhite'],'coffee':['CoffeeMachine','CoffeeCup','CoffeeBag'],'tray':['MealTray','CutleryHolder','Cutlery_knife_0','Cutlery_fork_1','Cutlery_fork_2'],'microwave':[]}
    shapes={};rows=[]
    for family,names in families.items():
        support=record['supports'][family]
        surface=unary_union([Polygon(x['polygon']) for x in layout['surfaces'] if x['group']==support['group']])
        paths=['/World/RoomEpisode/'+n for n in names] if names else ['/World/Assets/DiningTableMicrowave']
        shapes[family]=[]
        for p in paths:
            fp=footprint(s,p);shapes[family].append(fp)
            rows.append(dict(path=p,family=family,visible_projection_outside_support_m2=fp.difference(surface).area))
    overlaps=[]
    for i,(family,polys) in enumerate(shapes.items()):
        for other,op in list(shapes.items())[i+1:]:
            area=unary_union(polys).intersection(unary_union(op)).area
            if area>1e-7:overlaps.append(dict(families=[family,other],area_m2=area))
    r=dict(layout_id=record['layout_id'],method='Actual visible mesh XY convex hulls, not only nominal planning rectangles',
           support_rows=rows,cross_family_visible_overlap=overlaps,
           planar_mesh_check_passed=not overlaps and all(x['visible_projection_outside_support_m2']<1e-7 for x in rows),
           native_physics_validated=False,limitations=['Conservative full visible projection including overhangs; not contact geometry','No arm trajectory or native dynamics tested'])
    Path(record['scene']).with_name('mesh_audit.json').write_text(json.dumps(r,indent=2));out.append(r)
    print(json.dumps(r),flush=True)
(HERE/'other_layouts_v3/mesh_audits.json').write_text(json.dumps(out,indent=2))
