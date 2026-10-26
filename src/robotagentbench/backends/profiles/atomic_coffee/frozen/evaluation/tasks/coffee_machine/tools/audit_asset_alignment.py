"""Asset alignment check: public shared coffee assets (resources/shared_assets/coffee_v1, as composed in the scene the
evaluation opens) against the task objects of the data producer's source episodes (resources/source_episodes/...).

Compares COMPOSED quantities, not file hashes: per rigid body its mass, centre of mass, inertia, kinematic flag, solver
and damping attributes; per collider its type, approximation, enabled flag, local geometry (points relative to the
body), contact/rest offsets and the friction/restitution of the physics material it resolves to.
CPU USD only. Writes references/data_side/asset_alignment.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
BODY_ATTRS = ('physics:mass', 'physics:density', 'physics:centerOfMass', 'physics:diagonalInertia', 'physics:principalAxes',
              'physics:kinematicEnabled', 'physics:rigidBodyEnabled', 'physxRigidBody:solverPositionIterationCount',
              'physxRigidBody:solverVelocityIterationCount', 'physxRigidBody:sleepThreshold', 'physxRigidBody:linearDamping',
              'physxRigidBody:angularDamping', 'physxRigidBody:enableCCD', 'physxRigidBody:maxDepenetrationVelocity',
              'physxRigidBody:maxLinearVelocity', 'physxRigidBody:maxAngularVelocity')
COLLIDER_ATTRS = ('physics:approximation', 'physics:collisionEnabled', 'physxCollision:contactOffset', 'physxCollision:restOffset',
                  'physxCollision:torsionalPatchRadius', 'physxCollision:minTorsionalPatchRadius')
MATERIAL_ATTRS = ('physics:staticFriction', 'physics:dynamicFriction', 'physics:restitution', 'physics:density',
                  'physxMaterial:frictionCombineMode', 'physxMaterial:restitutionCombineMode',
                  'physxMaterial:compliantContactStiffness', 'physxMaterial:compliantContactDamping')


def plain(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return round(v, 7)
    try:
        return [round(float(x), 7) for x in v]
    except TypeError:
        try:
            return [round(float(x), 7) for x in (v.GetReal(), *v.GetImaginary())]
        except AttributeError:
            return str(v)


def geometry(prim, root, cache):
    from pxr import UsdGeom, Gf
    tf = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(root).GetInverse()
    if prim.IsA(UsdGeom.Mesh):
        pts = np.array([list(tf.Transform(Gf.Vec3d(*p))) for p in prim.GetAttribute('points').Get()])
    else:
        shape = {a.GetName(): plain(a.Get()) for a in prim.GetAttributes() if a.GetName() in ('radius', 'height', 'size', 'axis')}
        m = [[round(tf[i][j], 7) for j in range(4)] for i in range(4)]
        return dict(shape=shape, local_matrix=m)
    return dict(points=len(pts), min=np.round(pts.min(0), 6).tolist(), max=np.round(pts.max(0), 6).tolist(), _cloud=pts)


def physics_material(prim, stage):
    from pxr import UsdShade
    p = prim
    while p and not p.IsPseudoRoot():
        rel = p.GetRelationship('material:binding:physics')
        if rel and rel.GetTargets():
            m = stage.GetPrimAtPath(rel.GetTargets()[0])
            return {k: plain(m.GetAttribute(k).Get()) for k in MATERIAL_ATTRS if m and m.GetAttribute(k) and m.GetAttribute(k).HasAuthoredValue()} if m else 'unresolved'
        p = p.GetParent()
    return None


def body_facts(stage, path):
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(path)
    if not root:
        return None
    cache = UsdGeom.XformCache()
    out = dict(attributes={k: plain(root.GetAttribute(k).Get()) for k in BODY_ATTRS if root.GetAttribute(k) and root.GetAttribute(k).HasAuthoredValue()},
               rigid=root.HasAPI(UsdPhysics.RigidBodyAPI), colliders={})
    scale = cache.GetLocalToWorldTransform(root)
    out['world_scale'] = [round(scale.GetRow3(i).GetLength(), 7) for i in range(3)]
    for p in Usd.PrimRange(root):
        if p != root and p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI):
            # nested rigid bodies (the button under the machine) are reported under their own body
            owner = p
            while owner != root and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != root:
                continue
            name = str(p.GetPath())[len(path):]
            out['colliders'][name] = dict(type=str(p.GetTypeName()), geometry=geometry(p, root, cache),
                                          attributes={k: plain(p.GetAttribute(k).Get()) for k in COLLIDER_ATTRS if p.GetAttribute(k) and p.GetAttribute(k).HasAuthoredValue()},
                                          material=physics_material(p, stage))
    return out


def diff(a, b, prefix=''):
    """Flat list of differing leaves; geometry extents are compared with 1e-5 m tolerance."""
    out = []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # point clouds: every point of one set has a partner in the other within 10 micrometres
        from scipy.spatial import cKDTree
        if a.shape != b.shape:
            return [dict(item=prefix.rstrip('/'), public='%d points' % len(a), producer='%d points' % len(b))]
        worst = max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())
        return [dict(item=prefix.rstrip('/'), public='max point distance', producer=float(worst))] if worst > 1e-5 else []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append(dict(item=prefix + k, public=None, producer=b[k] if not isinstance(b[k], dict) else '…')); continue
            if k not in b: out.append(dict(item=prefix + k, public=a[k] if not isinstance(a[k], dict) else '…', producer=None)); continue
            out += diff(a[k], b[k], prefix + k + '/')
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(isinstance(x, (int, float)) for x in a + b):
        if max((abs(x - y) for x, y in zip(a, b)), default=0) > 1e-5:
            out.append(dict(item=prefix.rstrip('/'), public=a, producer=b))
        return out
    if isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-6 * max(1., abs(a)): out.append(dict(item=prefix.rstrip('/'), public=a, producer=b))
        return out
    if a != b:
        out.append(dict(item=prefix.rstrip('/'), public=a, producer=b))
    return out


def main():
    from pxr import Usd, Sdf
    sys.path.insert(0, str(REPO))
    from arena_api import l1_scene
    public = Usd.Stage.CreateInMemory()
    public.GetRootLayer().subLayerPaths = [str(p) for p in (*l1_scene.LAYERS, l1_scene.BASE)]
    with Usd.EditContext(public, public.GetSessionLayer()):
        public.GetPrimAtPath('/World/Task75').SetActive(True)
    T = '/World/Task75'
    report = dict(public_revision=json.loads((REPO / 'resources/shared_assets/coffee_v1/manifest.json').read_text())['revision'],
                  public_scene=[str(p.relative_to(REPO)) for p in (*l1_scene.LAYERS, l1_scene.BASE)], comparisons=[])
    P = '/World/CoffeeTaskPreview'
    cases = [('1-1', '044_right_241.3deg', 'green_tall'), ('1-1', '013_left_135.0deg', 'cream_classic'), ('1-1', '042_right_239.7deg', 'blue_compact'),
             ('1-2', 'bag_until50_seed_2130_episode_0000', 'cream_classic'), ('1-3', 'button_two_press_seed5058_v2', 'blue_compact'),
             ('1-4', 'cup_retrieve_prefix50_044_v1', 'cream_classic')]
    cups = {}
    for style in ('cream_classic', 'blue_compact', 'green_tall'):
        s = Usd.Stage.CreateInMemory()
        prim = s.DefinePrim('/Cup', 'Xform')
        prim.GetReferences().AddReference(str(REPO / f'resources/shared_assets/coffee_v1/cup_{style}.usda'))
        cups[style] = body_facts(s, '/Cup')
    pub = dict(machine=body_facts(public, T + '/Machine/Body') or body_facts(public, T + '/Machine'), button=body_facts(public, T + '/Machine/Button'),
               bag=body_facts(public, T + '/Bag'), bean=[body_facts(public, T + f'/CoffeeBeans/Bean_{i:03d}') for i in range(3)])
    report['public_prims'] = {k: bool(v) for k, v in pub.items() if not isinstance(v, list)}
    report['public_children'] = {p: [c.GetName() for c in public.GetPrimAtPath(p).GetChildren()][:12] for p in (T, T + '/Machine') if public.GetPrimAtPath(p)}
    for api, episode, style in cases:
        stage = Usd.Stage.Open(str(TASK / 'resources/source_episodes' / api / episode / 'coffee_objects.usdc'))
        manifest = json.loads((TASK / 'resources/source_episodes' / api / episode / 'episode.json').read_text())
        style = manifest['facts'].get('cup_style') or style
        pairs = [('machine body', pub['machine'], body_facts(stage, P + '/Machine/Body')),
                 ('button', pub['button'], body_facts(stage, P + '/Machine/Button')),
                 ('bag', pub['bag'], body_facts(stage, P + '/BeanBag')),
                 (f'cup {style}', cups[style], body_facts(stage, P + '/Cup'))]
        if stage.GetPrimAtPath('/World/PourBeans'):
            pairs += [(f'bean {i}', pub['bean'][i], body_facts(stage, f'/World/PourBeans/Bean_{i:03d}')) for i in range(3)]
        for name, a, b in pairs:
            if a is None or b is None:
                report['comparisons'].append(dict(api=api, episode=episode, object=name, status='missing', public=bool(a), producer=bool(b)))
                continue
            d = diff(a, b)
            report['comparisons'].append(dict(api=api, episode=episode, object=name, status='identical' if not d else 'different',
                                              colliders=dict(public=len(a['colliders']), producer=len(b['colliders'])), differences=d[:40], difference_count=len(d)))
            print(api, episode[:28], name, 'identical' if not d else f'DIFFERENT ({len(d)})', 'colliders', len(a['colliders']), len(b['colliders']))
            for x in d[:6]:
                print('      ', x['item'], '| public', str(x['public'])[:70], '| producer', str(x['producer'])[:70])
    (TASK / 'references/data_side/asset_alignment.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
