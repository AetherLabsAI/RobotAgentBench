"""Inspect source-authored poses without resolving old external USD references."""
import json
from pathlib import Path
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from arena_api import l1_scene

TASK = Path(__file__).resolve().parent
SPECS = {'wallet': ('7.13', 'Wallet'), 'glasses_case': ('7.15', 'GlassesCase')}


def source_pose(layer, path):
    stage = Usd.Stage.CreateInMemory()
    for part in Sdf.Path(path).GetPrefixes():
        prim = stage.DefinePrim(part, 'Xform')
        original = layer.GetPrimAtPath(part)
        if original:
            for name, attr in original.attributes.items():
                if name == 'xformOpOrder' or name.startswith('xformOp:'):
                    prim.CreateAttribute(name, attr.typeName).Set(attr.default)
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)


def inspect():
    base = Usd.Stage.Open(str(l1_scene.BASE))
    result = {}
    for kind, (source_id, name) in SPECS.items():
        layer = Sdf.Layer.FindOrOpen(str(TASK/'references/l1_20260916'/source_id/'core/initial_scene.usda'))
        result[kind] = {}
        for path in ('/World/Assets/CoffeeTable', '/World/Assets/'+name, '/World/Robot'):
            current = base.GetPrimAtPath(path)
            before = layer.GetPrimAtPath(path)
            result[kind][path] = {
                'source_matrix': [list(row) for row in source_pose(layer, path)],
                'current_active': current.IsActive(),
                'current_matrix': [list(row) for row in UsdGeom.Xformable(current).ComputeLocalToWorldTransform(0)],
                'source_refs': str(before.referenceList) if before else None,
                'current_refs': str(current.GetMetadata('references')),
                'source_physics': {n: str(a.default) for n,a in before.attributes.items() if n.startswith(('physics:', 'physx'))} if before else {},
                'current_physics': {a.GetName(): str(a.Get()) for a in current.GetAttributes() if a.GetName().startswith(('physics:', 'physx'))},
            }
        # Inspect inactive geometry in a session layer only; no shared source write.
        base.SetEditTarget(base.GetSessionLayer())
        target = base.GetPrimAtPath('/World/Assets/'+name)
        target.SetActive(True)
        result[kind]['colliders'] = [str(p.GetPath()) for p in Usd.PrimRange(target) if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get() is not False]
    (TASK/'artifacts').mkdir(exist_ok=True)
    (TASK/'artifacts/initial_asset_inspection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    inspect()
