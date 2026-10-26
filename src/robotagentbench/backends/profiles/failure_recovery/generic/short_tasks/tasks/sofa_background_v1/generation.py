"""Episode-only sofa background; never registers or changes a scoring target."""
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np

TASKS = frozenset(('hang_cap', 'pack_object', 'sofa_to_coffee_table', 'sofa_to_drawer'))
POOL = ('book_red', 'remote_tv', 'wallet', 'glasses_case', 'baseball_cap',
        'power_bank', 'water_bottle', 'notebook_green', 'paper_cup', 'paper_ball')
VERSION = 'sofa_background.v1'


def enabled(plan):
    return plan.get('task_name') in TASKS and plan.get('source') == 'sofa'


def selection(seed, excluded):
    rng = random.Random(f'{VERSION}:{seed}')
    count = rng.randint(2, 4)
    pool = [x for x in POOL if x not in excluded]
    return rng.sample(pool, count), rng


def separated(a, b, gap):
    return bool(np.any(np.asarray(a)[1] + gap <= np.asarray(b)[0] + 1e-9)
                or np.any(np.asarray(b)[1] + gap <= np.asarray(a)[0] + 1e-9))


def proposals(sizes, fixed, policy, obstacles, rng):
    """Existing full sofa region/quarter-turn/gap rules, with targets fixed first."""
    width, depth = policy['reference_approximate_width_m'], policy['reference_depth_m']
    edge, gap = policy['edge_margin_m'], policy['ordinary_object_padding_xy_m']
    low, high = np.array([-width/2+edge, edge]), np.array([width/2-edge, depth-edge])
    for attempt in range(100):
        placed, result = list(fixed), []
        for size in sizes:
            for trial in range(300):
                quarter = rng.randrange(4)
                span = np.asarray(size)[[1,0] if quarter % 2 else [0,1]]
                if np.any(high-low < span):
                    continue
                center = np.array([rng.uniform(a,b) for a,b in zip(low+span/2, high-span/2)])
                box = np.array([center-span/2, center+span/2])
                if all(separated(box,b,gap) for b in (*placed,*obstacles)):
                    placed.append(box)
                    result.append(dict(center_uv=center.tolist(), yaw_deg=quarter*90, bounds_uv=box.tolist()))
                    break
            else:
                break
        if len(result) == len(sizes):
            return result
    raise ValueError('Sofa backgrounds do not fit: targets/ranges/10 cm clearance were not changed')


def add(plan, stage, manifest):
    """Called after the target reset, before saving a NEW prepared room."""
    if not enabled(plan):
        return None
    if 'sofa_background' in manifest:
        raise ValueError('Sofa background initialization already applied')
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    from short_tasks.scene import ROOT
    from short_tasks.tasks.sofa_to_drawer.native_geometry import shape_points
    reg = ROOT/'layout_task_activation_v1/task_scene_registration_v1'
    previous = list(sys.path)
    try:
        sys.path.insert(0,str(reg))
        from randomize_other_living_v1 import world, setworld, center_bottom, seat_height_function
        from build_room_demo_v1 import exact, T, R, matrix
    finally:
        sys.path[:] = previous
    policy_dir = ROOT/'openend/tasks/scene_requirements/placement_sources'
    policy_file = policy_dir/'sofa-placement-policy-v1.json'
    obstacles_file = policy_dir/'sofa-structural-obstacles-v2.json'
    policy = json.loads(policy_file.read_text())
    structure = json.loads(obstacles_file.read_text())
    obstacles = [np.array(o['bounds']).reshape(2,2) for o in structure['obstacles']]
    baseline = Path(manifest['source_layout']).parent/'ID_01_LIVING_STRAIGHT.usdc'
    source = Usd.Stage.OpenMasked(str(baseline), Usd.StagePopulationMask(['/World/Assets/Sofa']))
    def sofa_vertices(s):
        return {str(p.GetPath()): points for p,points,c,v in shape_points(s,'/World/Assets/Sofa') if v}
    aa, bb = sofa_vertices(source), sofa_vertices(stage)
    if set(aa) != set(bb) or any(aa[k].shape != bb[k].shape for k in aa):
        raise ValueError('Sofa geometry is not the approved rigid reference')
    a,b = (np.vstack([m[k] for k in sorted(m)]) for m in (aa,bb))
    ac,bc = a.mean(0), b.mean(0)
    u,_,vt = np.linalg.svd((a-ac).T@(b-bc)); rotation = u@vt
    delta = np.eye(4); delta[:3,:3] = rotation; delta[3,:3] = bc-ac@rotation
    if (np.linalg.det(rotation)<0 or not np.allclose(rotation[2],[0,0,1],atol=1e-7)
            or not np.allclose(a@rotation+delta[3,:3],b,atol=1e-5,rtol=0)):
        raise ValueError('Sofa background requires unchanged rigid sofa geometry')
    inverse = np.linalg.inv(delta)
    def uv_bounds(path):
        points=np.vstack([v for _,v,c,visible in shape_points(stage,path) if c or visible])
        points=(np.c_[points,np.ones(len(points))]@inverse)[:,:3]
        uv=np.c_[-points[:,1]-.8710,points[:,0]+.3503563938]
        return np.array([uv.min(0),uv.max(0)])
    originals = {o['path']: np.array(world(stage,o['path'])) for o in manifest['objects']}
    fixed = [uv_bounds(o['path']) for o in manifest['objects'] if o.get('source')=='sofa']
    assets, rng = selection(manifest['seed'], {o['id'] for o in manifest['objects']})
    approved = ROOT/'openend/tasks/all_objects_integration/room_all_dynamic_v2/composition.json'
    report = json.loads(approved.read_text())
    sources = [Path(__file__),policy_file,obstacles_file,baseline,approved,
               reg/'randomize_other_living_v1.py',reg/'build_room_demo_v1.py']
    destination = stage.GetEditTarget().GetLayer()
    before = destination.ExportToString()
    rows, sizes = [], []
    try:
        for asset in assets:
            if asset == 'paper_ball':
                folder=ROOT/'openend/tasks/trash_transfer/dynamic_room.IF8mai'
                evidence=json.loads((folder/'dynamic_report.json').read_text())
                if evidence.get('passed') is not True:raise ValueError('Missing approved dynamic paper evidence')
                original=next(r['target'] for r in evidence['results'] if 'paper_ball' in r['body'])
                scene=folder/'room.usda';sources.append(folder/'dynamic_report.json')
            else:
                matches=[r for r in report['objects'] if r['id']==asset]
                if asset=='paper_cup':matches=[r for r in matches if '/Sofa_' in r['path']]
                if len(matches)!=1:raise ValueError('Ambiguous sofa background prototype: '+asset)
                scene=Path(report['scene']);original=matches[0]['path']
            proto=Usd.Stage.OpenMasked(str(scene),Usd.StagePopulationMask([original]))
            sources.extend(Path(l.realPath) for l in proto.GetUsedLayers() if l.realPath)
            path='/World/RoomEpisode/SofaBackground_'+asset
            if stage.GetPrimAtPath(path):raise ValueError('Background path already exists')
            Sdf.CreatePrimInLayer(destination,path)
            if not Sdf.CopySpec(proto.Flatten(),original,destination,path):raise ValueError('Prototype copy failed')
            bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.HasAPI(UsdPhysics.RigidBodyAPI)
                    and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
            if len(bodies)!=1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:
                raise ValueError('Background prototype is not one approved dynamic body: '+asset)
            center_bottom(stage,path,[0,0,0],R(-90))
            box=exact(stage,path)
            sizes.append([box[1,1]-box[0,1],box[1,0]-box[0,0]])
            rows.append(dict(id=asset,path=path,instance_id='sofa_background_'+asset,
                source='sofa',role='background',task_role='background',task_id=None,
                required_by=[],scoring_target=False,prototype_scene=str(scene),prototype_path=original))
        sampled=proposals(sizes,fixed,policy,obstacles,rng)
        height=seat_height_function(stage)
        for row,placement in zip(rows,sampled):
            u,v=placement['center_uv'];path=row['path']
            center_bottom(stage,path,[v-.3503563938,-u-.8710,.453],R(placement['yaw_deg']-90))
            setworld(stage,path,world(stage,path)*matrix(delta))
            box=exact(stage,path)
            setworld(stage,path,world(stage,path)*T([0,0,height(box)-box[0,2]]))
            row.update(placement=placement,visible_bounds_world_m=exact(stage,path).tolist())
        actual=[uv_bounds(r['path']) for r in rows]
        gap=policy['ordinary_object_padding_xy_m']
        for i,box in enumerate(actual):
            if not all(separated(box,b,gap) for b in (*fixed,*obstacles,*actual[:i])):
                raise ValueError('Actual background footprint violated sofa clearance')
        for path,transform in originals.items():
            if not np.array_equal(transform,np.array(world(stage,path))):
                raise ValueError('Existing object changed while placing backgrounds: '+path)
    except Exception:
        destination.ImportFromString(before)
        raise
    manifest['objects'].extend(rows)
    manifest['source_hashes'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(sources)})
    manifest['sofa_background']=dict(version=VERSION,count=len(rows),assets=assets,
        target_first=True,existing_objects_unchanged=True,scoring_targets_added=0,
        physics_parameters_changed=False,native_physics_validated=False)
    return manifest['sofa_background']
