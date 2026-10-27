#!/usr/bin/env python3
"""Seeded, source-preserving stock support/clearance repair before physics.

Search is discrete translation-only (5 mm radial grid, 64 directions), choosing
nearest valid candidate for each conflicted distractor. It is a local repair,
not a claim of globally minimum displacement. Both rendered geometry and enabled
physical proxy envelopes are checked. Compound proxy pieces remain separate;
convexDecomposition is conservatively enclosed by its convex hull because the
exact PhysX cooked decomposition is unavailable offline.
"""
import json, random, sys, time
from pathlib import Path
from collision_scene import *
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from shapely.affinity import translate
sys.path.insert(0,str(ROOT/'code/family3_repair/real2sim/teleop'))
from scene_collision_profile import apply_scene_collision_profile


def _proxies(stage, root, cache):
    result=[]
    for p in traversal(root):
        if not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():continue
        m=shape_mesh(p)
        if m is None:raise RuntimeError('Unsupported enabled collider '+str(p.GetPath()))
        approximation=UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.IsA(UsdGeom.Mesh) else 'analytic_tessellation'
        if approximation in ('convexHull','convexDecomposition'):m=m.convex_hull
        if approximation=='analytic_tessellation' and not p.IsA(UsdGeom.Cube):
            # The default triangulation is INSCRIBED. Enclose the true analytic
            # surface, so polygon facets cannot hide physical proxy overlap.
            m=m.convex_hull; normals=m.face_normals
            distances=np.einsum('ij,ij->i',normals,m.triangles[:,0])
            if p.IsA(UsdGeom.Sphere):
                support=np.full(len(normals),float(UsdGeom.Sphere(p).GetRadiusAttr().Get()))
            else:
                g=UsdGeom.Cylinder(p) if p.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(p)
                axis='XYZ'.index(str(g.GetAxisAttr().Get())); axial=np.abs(normals[:,axis])
                radial=np.sqrt(np.maximum(0.,1.-axial**2));r=float(g.GetRadiusAttr().Get());h=float(g.GetHeightAttr().Get())
                support=r*(radial if p.IsA(UsdGeom.Cylinder) else np.ones(len(normals)))+h/2*axial
            factor=float(np.max(support/distances));m.vertices*=max(1.,factor)
            approximation='conservative_circumscribed_analytic_envelope'
        m.apply_transform(matrix(cache.GetLocalToWorldTransform(p)))
        result.append(Body(str(p.GetPath())+'#proxy',m,metadata=dict(approximation=approximation)))
    return result


def _surfaces(scene):
    boards=[b for b in scene.bodies if '/ShelvingUnit/' in b.name and 'shelf_boards_' in b.name]
    triangles=np.concatenate([points(b.v,b.transform)[b.f] for b in boards])
    flat=np.ptp(triangles[:,:,2],axis=1)<1e-5
    result=[]
    for lo,hi in [(0.,.4),(.4,.9),(.95,1.3),(1.35,1.65)]:
        cand=triangles[flat & (triangles[:,:,2].mean(1)>lo) & (triangles[:,:,2].mean(1)<hi)]
        if not len(cand):continue
        top=float(cand[:,:,2].max());topfaces=cand[np.abs(cand[:,:,2].mean(1)-top)<1e-5]
        poly=unary_union([Polygon(t[:,:2]) for t in topfaces if Polygon(t[:,:2]).area>1e-10])
        result.append(dict(top=top,polygon=poly,paths=[b.name for b in boards]))
    return result


def _separated(a,b,clearance):
    if not broad_overlap(a,b,clearance):return True
    # A triangle distance query establishes surface clearance; containment is
    # tested separately because nested closed surfaces can have positive gap.
    distance=fcl.distance(a.obj,b.obj,fcl.DistanceRequest(),fcl.DistanceResult())
    return distance>=clearance and contact(a,b,containment=True) is None


def repair_initial_layout(stage,target_path,seed,report_path,initial_state_npz=None):
    start=time.monotonic(); previous=stage.GetEditTarget()
    active=sorted(str(p.GetPath()) for parent in ('/World/Assets/ShelfItems','/World/Assets/Tier2Items')
                  for p in stage.GetPrimAtPath(parent).GetChildren())
    if target_path not in active:raise ValueError('Source target is not active: '+target_path)
    original={p:UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in active}
    robot_original=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot'))
    profile=apply_scene_collision_profile(stage)
    scene=Scene(stage.GetRootLayer().realPath or SCENE,stock=False,stage=stage)
    cache=UsdGeom.XformCache(); supports=_surfaces(scene)
    items={}
    for path in active:
        visual=scene.objects.get(path)
        if visual is None:raise ValueError('Active stock has no visual geometry: '+path)
        geometries=[visual]+_proxies(stage,stage.GetPrimAtPath(path),cache)
        v=np.concatenate([points(b.v,b.transform) for b in geometries])
        bottom=float(v[:,2].min()); shelf=min(supports,key=lambda s:abs(s['top']-bottom))
        items[path]=dict(visual=visual,geometry=geometries,support=shelf,
                         footprint=MultiPoint(v[:,:2]).convex_hull,bottom=bottom,
                         delta=np.zeros(3),original_bottom=bottom)
    fixtures=[b for b in scene.bodies if b.name not in active]
    if initial_state_npz:
        d=np.load(initial_state_npz,allow_pickle=False); robot=Robot()
        robot.fk(list(d['robot_dof_names']),d['robot_joint_position_rad_or_m'][0],pose(d['robot_root_xyz_m'],d['robot_root_wxyz']))
        fixtures+=robot.bodies
    clearance=.002;support_gap=.0005;support_margin=.001
    def move(item,delta):
        item['delta']=np.asarray(delta)
        for b in item['geometry']:
            w=b.authored_transform.copy();w[:3,3]+=delta;b.update(w)
    def supported(item):
        fp=translate(item['footprint'],xoff=item['delta'][0],yoff=item['delta'][1])
        return item['support']['polygon'].buffer(-support_margin).covers(fp)
    def pair_ok(a,b):
        return all(_separated(x,y,clearance) for x in a['geometry'] for y in b['geometry'])
    def fixture_ok(item):
        return all(_separated(x,b,.0002) for x in item['geometry'] for b in fixtures)
    # Align only distractors to their nearest existing horizontal shelf surface.
    # The target is fixed even if its source initialization itself is invalid.
    for path,item in items.items():
        if path!=target_path:move(item,np.array([0.,0.,item['support']['top']+support_gap-item['bottom']]))
    initial_pairs=[(a,b) for i,a in enumerate(active) for b in active[i+1:] if not pair_ok(items[a],items[b])]
    fixed={target_path}; pending=set(active)-fixed
    # First preserve every already valid, unconflicted stock placement. Objects
    # involved in conflicts are placed around these fixed initial obstacles.
    conflicted={p for pair in initial_pairs for p in pair}
    for p in sorted(pending):
        if p not in conflicted and supported(items[p]) and fixture_ok(items[p]):fixed.add(p)
    pending-=fixed
    rng=random.Random(str(seed)); tie={p:rng.random() for p in sorted(pending)}
    order=sorted(pending,key=lambda p:(-items[p]['footprint'].area,tie[p],p))
    # Try larger clutter first so a moved small can cannot consume the only
    # feasible location for a wide source box. Seed breaks ties/direction order.
    angles=np.arange(64)*2*np.pi/64; rng.shuffle(angles)
    unresolved=[]; candidates=0
    for path in order:
        item=items[path]; dz=item['delta'][2]; found=False
        offsets=[(0.,0.)]
        offsets.extend((r*np.cos(a),r*np.sin(a)) for r in np.arange(.005,.805,.005) for a in angles)
        for dx,dy in offsets:
            candidates+=1;move(item,np.array([dx,dy,dz]))
            if not supported(item):continue
            if not all(pair_ok(item,items[p]) for p in fixed):continue
            if not fixture_ok(item):continue
            found=True;fixed.add(path);break
        if not found:unresolved.append(path);move(item,np.array([0.,0.,dz]));fixed.add(path)
    final_pairs=[dict(a=a,b=b) for i,a in enumerate(active) for b in active[i+1:] if not pair_ok(items[a],items[b])]
    support_failures=[p for p in active if p!=target_path and not supported(items[p])]
    fixture_failures=[p for p in active if p!=target_path and not fixture_ok(items[p])]
    report=dict(schema='family3.initial_layout_repair.v1',seed=str(seed),target_prim=target_path,
                source_active_stock=active,source_active_count=len(active),source_layer=stage.GetRootLayer().realPath,
                robot_initial_state=str(initial_state_npz) if initial_state_npz else None,
                policy='Source-visible membership preserved; translation-only nearest discrete feasible search; no action/future-pose input.',
                clearance_m=clearance,support_gap_m=support_gap,support_inset_m=support_margin,
                initial_conflicts=[dict(a=a,b=b) for a,b in initial_pairs],unresolved_conflicts=final_pairs,
                unresolved_search=unresolved,unsupported_distractors=support_failures,fixture_conflicts=fixture_failures,
                fixed_target_initial_support='Unchanged; requires task-specific physical validation, including in-hand PLACE contact.',
                collision_profile=profile,search_candidates=candidates,
                objects=[dict(path=p,delta_world_m=items[p]['delta'].tolist(),support_top_world_m=items[p]['support']['top'],
                              original_bottom_world_m=items[p]['original_bottom'],support_geometry=items[p]['support']['paths'],
                              support_polygon_wkt=items[p]['support']['polygon'].wkt,
                              proxy_approximations=[b.metadata for b in items[p]['geometry'][1:]]) for p in active],
                passed=not(final_pairs or unresolved or support_failures or fixture_failures),elapsed_seconds=time.monotonic()-start,
                limitations=['Convex-decomposition colliders use conservative hull envelopes; analytic proxies use circumscribed envelopes.',
                             'Physical settling and full actual trajectory acceptance are mandatory after this initialization check.'])
    report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    stage.SetEditTarget(previous)
    if not report['passed']:raise RuntimeError('No valid source-preserving initial layout: '+str(report_path))
    # Author translations only after ALL candidate geometry checks pass.
    stage.SetEditTarget(stage.GetRootLayer())
    for p in active:
        if p==target_path:continue
        delta=items[p]['delta']
        if np.linalg.norm(delta)<1e-10:continue
        prim=stage.GetPrimAtPath(p); xf=UsdGeom.Xformable(prim)
        world=Gf.Matrix4d(original[p]);t=world.ExtractTranslation();world.SetTranslateOnly(t+Gf.Vec3d(*map(float,delta)))
        parent=cache.GetLocalToWorldTransform(prim.GetParent())
        xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix='initial_layout').Set(world*parent.GetInverse())
    check=UsdGeom.XformCache()
    assert check.GetLocalToWorldTransform(stage.GetPrimAtPath(target_path))==original[target_path]
    assert check.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot'))==robot_original
    assert active==sorted(str(p.GetPath()) for parent in ('/World/Assets/ShelfItems','/World/Assets/Tier2Items') for p in stage.GetPrimAtPath(parent).GetChildren())
    stage.SetEditTarget(previous)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--target',required=True);p.add_argument('--seed',default='0');p.add_argument('--report',required=True);p.add_argument('--output-scene',required=True);p.add_argument('--initial-state')
    a=p.parse_args();stage=Usd.Stage.CreateNew(a.output_scene);stage.GetRootLayer().subLayerPaths=[str(Path(a.scene).resolve())]
    result=repair_initial_layout(stage,a.target,a.seed,a.report,a.initial_state);stage.GetRootLayer().Save()
    stage.Export(a.output_scene+'.profile_snapshot.usda')
    print(json.dumps(dict(passed=result['passed'],elapsed_seconds=result['elapsed_seconds'],search_candidates=result['search_candidates'],deltas={x['path']:x['delta_world_m'] for x in result['objects'] if np.linalg.norm(x['delta_world_m'])>1e-9})))
