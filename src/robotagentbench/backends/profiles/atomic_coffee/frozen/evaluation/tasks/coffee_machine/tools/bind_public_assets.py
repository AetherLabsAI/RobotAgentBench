"""Bind every packaged source episode to the public shared coffee assets (resources/shared_assets/coffee_v1).

The evaluation scene uses the PUBLIC machine / bag / cups / beans, the same for every API and every user of the
house scene. From a source episode only the layout is taken: where the machine, the bag, the cup and each bean are,
which of the three public cup styles it is, and (1-4) the liquid that is already in the cup.

Per episode this writes next to episode.json:
  public_binding.json   cup style, world matrices (row-major Gf.Matrix4d), button local matrix, bean matrices,
                        and the geometry comparison public asset <-> producer object in WORLD space
  liquid.usdc           1-4 only: /World/CoffeeLiquid of the packaged episode, unchanged
CPU USD only. No physics parameter is read from the producer and none is written.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO))
TOLERANCE_M = 1e-5


def flat(m):
    return [float(m[i][j]) for i in range(4) for j in range(4)]


def cloud(stage, root, cache, skip_nested=True):
    """World-space points of every enabled collider owned by this rigid body."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import itertools
    prim = stage.GetPrimAtPath(root)
    out = []
    for p in Usd.PrimRange(prim):
        if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        owner = p
        while owner != prim and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        if skip_nested and owner != prim:
            continue
        tf = cache.GetLocalToWorldTransform(p)
        if p.IsA(UsdGeom.Mesh):
            pts = p.GetAttribute('points').Get()
        elif p.IsA(UsdGeom.Cube):
            s = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
            pts = list(itertools.product((-s, s), repeat=3))
        elif p.IsA(UsdGeom.Cylinder):
            c = UsdGeom.Cylinder(p); r, h, ax = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get() / 2, 'XYZ'.index(c.GetAxisAttr().Get())
            pts = []
            for s in (-h, h):
                for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                    v = [r * np.cos(a), r * np.sin(a)]; v.insert(ax, s); pts.append(v)
        elif p.IsA(UsdGeom.Sphere):
            r = UsdGeom.Sphere(p).GetRadiusAttr().Get()
            pts = [(0, 0, 0), (r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
        elif p.IsA(UsdGeom.Capsule):
            c = UsdGeom.Capsule(p); r, h, ax = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get() / 2, 'XYZ'.index(c.GetAxisAttr().Get())
            pts = []
            for s in (-h - r, -h, h, h + r):
                v = [0., 0.]; v.insert(ax, s); pts.append(v)
            for s in (-h, h):
                for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    v = [r * np.cos(a), r * np.sin(a)]; v.insert(ax, s); pts.append(v)
        else:
            raise ValueError('Unsupported collider type: ' + str(p.GetPath()) + ' ' + str(p.GetTypeName()))
        out.extend(list(tf.Transform(Gf.Vec3d(*[float(x) for x in q]))) for q in pts)
    return np.asarray(out, dtype=float)


def distance(a, b):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def main():
    from pxr import Usd, UsdGeom, Sdf, Gf
    from evaluation.tasks.coffee_machine.specs import EPISODES, PACKAGED, CUP_STYLES, SHARED_ASSETS, APIS, spec
    from evaluation.tasks.coffee_machine import initial_state
    from arena_api import coffee_assets
    revision = coffee_assets.verify_files()['revision']
    # local (root frame) clouds of the three public cups, to identify the style of a packaged cup
    styles = {}
    for style in CUP_STYLES:
        s = Usd.Stage.CreateInMemory()
        s.DefinePrim('/Cup', 'Xform').GetReferences().AddReference(str(SHARED_ASSETS / f'cup_{style}.usda'))
        styles[style] = cloud(s, '/Cup', UsdGeom.XformCache())
    summary = []
    for api in APIS:
        for folder in sorted((EPISODES / api).iterdir()):
            if not (folder / 'episode.json').is_file():
                continue
            manifest = json.loads((folder / 'episode.json').read_text())
            objects = folder / 'coffee_objects.usdc'
            if hashlib.sha256(objects.read_bytes()).hexdigest() != manifest['coffee_objects_sha256']:
                raise ValueError('Packaged task objects changed: ' + str(objects))
            stage = Usd.Stage.Open(str(objects))
            cache = UsdGeom.XformCache()
            world = lambda path: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            cup_local = cloud(stage, PACKAGED['cup'], cache)
            inverse = world(PACKAGED['cup']).GetInverse()
            cup_local = np.asarray([list(inverse.Transform(Gf.Vec3d(*p))) for p in cup_local])
            match = {style: distance(points, cup_local) for style, points in styles.items()}
            style = min(match, key=match.get)
            if match[style] > TOLERANCE_M:
                raise ValueError(f'{api} {folder.name}: the packaged cup is none of the public cup styles: {match}')
            recorded = manifest['facts'].get('cup_style')
            if recorded is not None and recorded != style:
                raise ValueError(f'{api} {folder.name}: cup geometry says {style}, the producer record says {recorded}')
            button = stage.GetPrimAtPath(PACKAGED['machine'] + '/Button')
            binding = dict(
                schema='arena.coffee_machine.public_binding.v1', api=api, episode=folder.name, public_revision=revision,
                cup_style=style, cup_style_source='producer record and geometry' if recorded else 'geometry (the producer did not record it)',
                cup_style_distance_m=match,
                world=dict(machine=flat(world(PACKAGED['machine'])), cup=flat(world(PACKAGED['cup'])), bag=flat(world(PACKAGED['bag']))),
                button_local=flat(UsdGeom.Xformable(button).GetLocalTransformation()),
                beans=None, liquid=None, coffee_objects_sha256=manifest['coffee_objects_sha256'])
            beans = stage.GetPrimAtPath(PACKAGED['beans'])
            if spec(api).has_beans:
                children = beans.GetChildren()
                if [c.GetName() for c in children] != [f'Bean_{i:03d}' for i in range(100)]:
                    raise ValueError('Expected the producer\'s 100 beans')
                binding['beans'] = [flat(world(str(c.GetPath()))) for c in children]
            elif beans:
                raise ValueError('Unexpected beans in a source episode without beans')
            liquid = stage.GetPrimAtPath(PACKAGED['liquid'])
            if api == '1-4':
                if not liquid:
                    raise ValueError('1-4 needs the liquid that is already in the cup')
                target = folder / 'liquid.usdc'
                if target.exists():
                    target.unlink()
                layer = Sdf.Layer.CreateNew(str(target))
                Sdf.CreatePrimInLayer(layer, '/World').specifier = Sdf.SpecifierOver
                source = stage.Flatten()
                if not Sdf.CopySpec(source, PACKAGED['liquid'], layer, '/World/CoffeeLiquid'):
                    raise ValueError('Could not copy the liquid')
                layer.defaultPrim = 'World'
                layer.Save()
                points = stage.GetPrimAtPath(PACKAGED['liquid'] + '/Particles').GetAttribute('points').Get()
                binding['liquid'] = dict(file='liquid.usdc', sha256=hashlib.sha256(target.read_bytes()).hexdigest(), particles=len(points),
                                         origin='particles of the producer\'s final frame of this episode, moved rigidly with the cup to its initial pose '
                                                '(the producer\'s own pre-fill code is not on 202)')
            (folder / 'public_binding.json').write_text(json.dumps(binding, indent=1, allow_nan=False) + '\n')
            # compose the evaluation scene from public assets and compare it with the producer's objects in WORLD space
            out = TASK / 'artifacts/binding_check' / api / folder.name
            if out.exists():
                import shutil
                shutil.rmtree(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            from arena_api import l1_scene
            scene = initial_state.prepare(l1_scene.BASE, out, api=api, episode=folder.name)
            public = Usd.Stage.Open(str(scene))
            pc = UsdGeom.XformCache()
            from evaluation.tasks.coffee_machine import specs as S
            pairs = [('machine body', S.MACHINE_BODY, PACKAGED['machine'] + '/Body'), ('button', S.BUTTON, PACKAGED['machine'] + '/Button'),
                     ('cup', S.CUP, PACKAGED['cup']), ('bag', S.BAG, PACKAGED['bag'])]
            if spec(api).has_beans:
                pairs += [(f'bean {i:03d}', f'{S.BEANS}/Bean_{i:03d}', f'{PACKAGED["beans"]}/Bean_{i:03d}') for i in range(100)]
            comparison = {}
            for name, a, b in pairs:
                pa, pb = cloud(public, a, pc), cloud(stage, b, cache)
                origin = (pc.GetLocalToWorldTransform(public.GetPrimAtPath(a)).ExtractTranslation()
                          - cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b)).ExtractTranslation()).GetLength()
                d = distance(pa, pb)
                comparison[name] = dict(origin_difference_m=float(origin), collider_points=dict(public=len(pa), producer=len(pb)),
                                        max_point_distance_m=d, same_geometry=bool(d <= TOLERANCE_M))
            bean_rows = [v for k, v in comparison.items() if k.startswith('bean ')]
            report = {k: v for k, v in comparison.items() if not k.startswith('bean ')}
            if bean_rows:
                report['beans (100)'] = dict(origin_difference_m=max(r['origin_difference_m'] for r in bean_rows),
                                             max_point_distance_m=max(r['max_point_distance_m'] for r in bean_rows),
                                             same_geometry=all(r['same_geometry'] for r in bean_rows))
            if api == '1-4' and public.GetPrimAtPath(S.LIQUID + '/Particles'):
                a = np.asarray(public.GetPrimAtPath(S.LIQUID + '/Particles').GetAttribute('points').Get(), dtype=float)
                b = np.asarray(stage.GetPrimAtPath(PACKAGED['liquid'] + '/Particles').GetAttribute('points').Get(), dtype=float)
                report['liquid'] = dict(particles=dict(public=len(a), producer=len(b)), max_point_distance_m=float(np.abs(a - b).max()) if a.shape == b.shape else None,
                                        same_geometry=bool(a.shape == b.shape and np.abs(a - b).max() <= TOLERANCE_M))
            elif api == '1-4':
                # since 2026-09-30 the run authors the data producer's own prefill (vendor_producer/filled_coffee.py) before physics;
                # the packaged final-frame particles are reference material only and are not in the prepared scene
                binding['liquid']['note'] = 'reference only: the run uses the producer prefill code, not these particles'
            binding['world_comparison'] = report
            (folder / 'public_binding.json').write_text(json.dumps(binding, indent=1, allow_nan=False) + '\n')
            row = dict(api=api, episode=folder.name, cup_style=style, **{k: ('same' if v['same_geometry'] else 'DIFFERENT %.4f m' % v['max_point_distance_m']) for k, v in report.items()},
                       origins_max_m=max(v.get('origin_difference_m', 0.) for v in report.values()))
            summary.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (TASK / 'references/data_side/public_binding_summary.json').write_text(json.dumps(dict(
        public_revision=revision, tolerance_m=TOLERANCE_M,
        note='Evaluation scenes use the public shared assets; only the layout comes from the source episode. '
             'DIFFERENT marks an object whose public geometry is not the one the source episode was recorded with.',
        rows=summary), indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
