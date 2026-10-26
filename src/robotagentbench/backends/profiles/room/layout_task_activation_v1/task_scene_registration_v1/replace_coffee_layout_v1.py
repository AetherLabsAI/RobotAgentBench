"""Compose the approved layout-01 coffee replacement, preserving the release.

CPU composition and visual bounds checks only; no running simulator is modified.
"""
import hashlib
import json
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc')
ASSET = Path('/home/lingjun/robot_agent_arena/api-protocol/all_assets/coffee_machine/asset.usda')
OLD = '/World/Assets/CoffeeGrinder'
NEW = '/World/Assets/CoffeeMachine'
OUTPUT = ROOT / 'scene_overrides/layout01_coffee_all_assets_v2.usda'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bounds(stage, path):
    r = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy']).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    assert not r.IsEmpty(), path
    return {'min': list(r.GetMin()), 'max': list(r.GetMax())}

def main():
    before = sha(BASE)
    base = Usd.Stage.Open(str(BASE))
    source = Usd.Stage.Open(str(ASSET))
    assert source.GetDefaultPrim().GetCustomData()['frontAxis'] == '+Z'
    old = bounds(base, OLD)
    # Keep the old device's location, but NOT its 1.15 scale.
    origin = UsdGeom.XformCache().GetLocalToWorldTransform(base.GetPrimAtPath(OLD)).ExtractTranslation()
    data = json.loads((ROOT / 'web/kitchen-workspace-data.js').read_text().removeprefix('window.KITCHEN_WORKSPACE = ').rstrip(';\n'))
    # The original appliance straddles two adjacent countertop modules.
    # Use their measured top, not the old appliance's authored Z (which embeds it).
    local = bounds(source, '/Asset')
    footprint = {'min': [origin[0]-local['max'][2], origin[1]-local['max'][0]],
                 'max': [origin[0]-local['min'][2], origin[1]-local['min'][0]]}
    tiles = [s for s in data['surfaces'] if all(min(footprint['max'][i],s['max'][i])-max(footprint['min'][i],s['min'][i])>0 for i in (0,1))]
    assert tiles
    origin[2] = max(s['max'][2] for s in tiles)
    m = Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, origin[0], origin[1], origin[2], 1)
    OUTPUT.parent.mkdir(exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError('Versioned output exists; do not overwrite: ' + str(OUTPUT))
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    stage.GetRootLayer().subLayerPaths = [str(BASE)]
    stage.OverridePrim(OLD).SetActive(False)
    prim = UsdGeom.Xform.Define(stage, NEW)
    prim.GetPrim().GetReferences().AddReference(str(ASSET), '/Asset')
    prim.MakeMatrixXform().Set(m)
    prim.GetPrim().SetCustomDataByKey('layoutPlacementStatus', 'user_approved_asset_replacement_position_provisional')
    stage.GetRootLayer().Save()
    # Reopen from disk, verifying composition instead of only authored opinions.
    stage = Usd.Stage.Open(str(OUTPUT))
    assert not stage.GetPrimAtPath(OLD).IsActive()
    assert stage.GetPrimAtPath(NEW).IsActive()
    new = bounds(stage, NEW)
    supports = [s for s in tiles if abs(new['min'][2]-s['max'][2]) < 1e-6]
    cuts = [sorted({new['min'][i],new['max'][i]} | {max(new['min'][i],min(new['max'][i],s[k][i])) for s in supports for k in ('min','max')}) for i in (0,1)]
    for x0,x1 in zip(cuts[0],cuts[0][1:]):
        for y0,y1 in zip(cuts[1],cuts[1][1:]):
            x,y=(x0+x1)/2,(y0+y1)/2
            assert any(s['min'][0]-1e-6<=x<=s['max'][0]+1e-6 and s['min'][1]-1e-6<=y<=s['max'][1]+1e-6 for s in supports), 'Uncovered countertop footprint'
    collisions = [b['name'] for b in data['blockers'] if b['path'] != OLD and all(min(new['max'][i], b['max'][i])-max(new['min'][i], b['min'][i]) > .001 for i in range(3))]
    assert not collisions, collisions
    joints = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(NEW)) if p.IsA(UsdPhysics.Joint)]
    assert joints, 'Button joint must compose'
    for p in Usd.PrimRange(stage.GetPrimAtPath(NEW)):
        if p.IsA(UsdPhysics.Joint):
            for key in ('physics:body0', 'physics:body1'):
                for target in p.GetRelationship(key).GetTargets():
                    assert stage.GetPrimAtPath(target), (p.GetPath(), target)
    assert sha(BASE) == before
    report = dict(schema='arena.layout01.coffee_replacement.v1', base_scene=str(BASE), base_sha256=before,
                  scene=str(OUTPUT), scene_sha256=sha(OUTPUT), asset=str(ASSET), asset_sha256=sha(ASSET),
                  dependency_sha256={l.realPath: sha(l.realPath) for l in source.GetUsedLayers() if l.realPath},
                  removed_path=OLD, removed_bounds=old, path=NEW, bounds=new, world_transform=[list(r) for r in m],
                  size_m=[new['max'][i]-new['min'][i] for i in range(3)], front_world=[-1, 0, 0],
                  support=supports[0], support_tiles=supports, front_to_edge_m=new['min'][0]-supports[0]['min'][0],
                  joints=joints, static_visual_overlap=collisions, physical_validation='not_performed',
                  task_motion_envelope='pending', live_simulator_changed=False,
                  limitations=['Static visual bounds only; no contact, reachability, settling or task replay validation.',
                               'Coffee objects and device behaviors are not initialized by this placement layer.'])
    (ROOT / 'web/coffee-layout-replacement-v1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (ROOT / 'web/coffee-layout-replacement-data.js').write_text('window.COFFEE_LAYOUT_REPLACEMENT = '+json.dumps(report, ensure_ascii=False)+';\n')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
