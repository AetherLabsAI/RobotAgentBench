"""Prove that a seeded appearance environment (tools/build_l1_scene.py) has the same physics as the plain baseline.

Composes the 1-2 fixed layout once over the public baseline and once over the given environment, then compares every
physics-relevant quantity prim by prim: authored physics:/physx attributes, collision/rigid-body/articulation/joint APIs,
world transforms of colliders and rigid bodies, physics material bindings. CPU USD only. Writes
references/data_side/appearance_physics_check.json.
"""
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO))


def fingerprint(stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    rows = {}
    for prim in stage.Traverse():
        physics = {a.GetName(): str(a.Get()) for a in prim.GetAuthoredAttributes() if a.GetName().startswith(('physics:', 'physx', 'drive:', 'limit:', 'state:'))}
        apis = [s for s in prim.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s]
        interesting = bool(physics or apis or prim.IsA(UsdPhysics.Joint))
        if not interesting:
            continue
        row = dict(type=str(prim.GetTypeName()), active=prim.IsActive(), apis=sorted(apis), attributes=physics)
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
            m = cache.GetLocalToWorldTransform(prim)
            row['world'] = [round(float(m[i][j]), 9) for i in range(4) for j in range(4)]
        rel = prim.GetRelationship('material:binding:physics')
        if rel and rel.GetTargets():
            row['physics_material'] = [str(t) for t in rel.GetTargets()]
        for name in ('physics:body0', 'physics:body1', 'physics:filteredGroups', 'physics:filteredPairs'):
            r = prim.GetRelationship(name)
            if r and r.GetTargets():
                row[name] = [str(t) for t in r.GetTargets()]
        rows[str(prim.GetPath())] = row
    return rows


def main():
    import argparse
    from pxr import Usd
    from arena_api import l1_scene
    from evaluation.tasks.coffee_machine import initial_state
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--api', default='1-2')
    a = p.parse_args()
    out = TASK / 'artifacts/cpu-checks/environment_physics'
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    scenes = {}
    for label, source in (('baseline', l1_scene.BASE), ('environment', a.environment.resolve())):
        scenes[label] = initial_state.prepare(source, out / label, api=a.api)
    prints = {label: fingerprint(Usd.Stage.Open(str(scene))) for label, scene in scenes.items()}
    base, env = prints['baseline'], prints['environment']
    differences = []
    for path in sorted(set(base) | set(env)):
        if path not in base or path not in env:
            differences.append(dict(prim=path, only_in='environment' if path in env else 'baseline'))
        elif base[path] != env[path]:
            keys = [k for k in set(base[path]) | set(env[path]) if base[path].get(k) != env[path].get(k)]
            differences.append(dict(prim=path, fields=keys, baseline={k: base[path].get(k) for k in keys}, environment={k: env[path].get(k) for k in keys}))
    report = dict(environment=str(a.environment.resolve()), api=a.api, physics_prims=len(base), differences=differences,
                  identical=not differences, scope='authored physics/physx/drive/limit attributes, applied physics APIs, collider and body world transforms, physics material bindings, joint bodies and filters')
    (TASK / 'references/data_side/appearance_physics_check.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(physics_prims=len(base), differences=len(differences), identical=report['identical'], first=differences[:3])))


if __name__ == '__main__':
    main()
