"""Atomic 1-2 composition with ONE explicitly approved physical opinion.

Only /World/Task75/Machine/Body.physics:kinematicEnabled becomes false in
this fresh variant layer. Original source layers, joint and other physics
properties remain unchanged. No bag-relocation initialization is imported.
"""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACHINE_BODY = '/World/Task75/Machine/Body'
ALLOWED = MACHINE_BODY + '.physics:kinematicEnabled'


def physical_properties(stage):
    from pxr import Usd
    values = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Task75')):
        for prop in prim.GetProperties():
            name = prop.GetName()
            if name.startswith(('physics:', 'physx', 'drive:', 'material:binding')):
                value = prop.Get() if hasattr(prop, 'Get') else prop.GetTargets()
                values[str(prop.GetPath())] = repr(value)
    return values


def authorize_dynamic_body(stage):
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(MACHINE_BODY)
    if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Original coffee-machine rigid body required')
    before = physical_properties(stage)
    if prim.GetAttribute('physics:kinematicEnabled').Get() is not True:
        raise ValueError('Expected unchanged source kinematic body')
    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
    after = physical_properties(stage)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    if changed != [ALLOWED]:
        raise RuntimeError('Unexpected physical changes: ' + repr(changed))
    return dict(changed_properties=changed, before=before[ALLOWED], after=after[ALLOWED],
                all_other_physical_properties_equal=True, source_assets_modified=False)


def build(output, *, seed, role):
    from pxr import Gf, Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    from evaluation.tasks.coffee_machine import initial_state as src, randomization
    from evaluation.tasks.coffee_machine.specs import (
        STATION, MACHINE, BUTTON, CUP, BAG, BEANS, SHARED_ASSETS, ROBOT, episodes)
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit nonnegative integer seed required')
    if role not in ('source', 'random'):
        raise ValueError('Explicit source or random initialization required')
    output = Path(output).resolve()
    if output == HERE or not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh output inside coffee_machine_push_interference required')
    manifest, binding, _ = src.package('1-2', episodes('1-2')[0])
    output.mkdir(parents=True)
    scene = output / 'initial.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(p, output) for p in
        [src.REFERENCE, *l1_scene.LAYERS, l1_scene.BASE]]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetPrimAtPath('/World').CreateAttribute(l1_scene.MARKER, Sdf.ValueTypeNames.String).Set(l1_scene.POLICY)
    stage.OverridePrim(STATION).SetActive(True)
    if UsdGeom.Xformable(stage.GetPrimAtPath(STATION)).ComputeLocalToWorldTransform(0) != Gf.Matrix4d(1):
        raise ValueError('Source station must retain identity transform')
    cup = stage.GetPrimAtPath(CUP)

    def cup_style(style):
        cup.GetReferences().SetReferences([Sdf.Reference(os.path.relpath(
            SHARED_ASSETS / f'cup_{style}.usda', output))])
        cup.SetCustomDataByKey(src.STYLE_KEY, style)

    cup_style(binding['cup_style'])
    for path, key in [(MACHINE, 'machine'), (CUP, 'cup'), (BAG, 'bag')]:
        src.place(stage, path, binding['world'][key])
    src.place(stage, BUTTON, binding['button_local'])
    for i, matrix in enumerate(binding['beans']):
        src.place(stage, f'{BEANS}/Bean_{i:03d}', matrix)
    record = (randomization.apply(stage, '1-2', manifest, seed, set_cup_style=cup_style)
              if role == 'random' else manifest['facts']['randomization'])
    l1_scene.verify(stage)
    inspection = src.inspect_stage(stage, '1-2')
    base = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT)).ExtractTranslation()
    if max(abs(a - b) for a, b in zip(base, manifest['robot_base']['root_xyz'])) > 1e-9:
        raise ValueError('Source robot station changed')
    source_hashes = {p.realPath: src.digest(p.realPath) for p in stage.GetUsedLayers()
                     if p.realPath and p.realPath != str(scene)}
    dynamic = authorize_dynamic_body(stage)
    # Runtime cannot call the original kinematic-only stage inspector unchanged.
    inspection.update(machine_dynamic=True, original_structure_checked_before_approved_override=True,
                      physical_settle_acceptance=None)
    from short_task_robustness.tasks.coffee_bag_interference.build_preview_scene import visual_preload
    visual_receipt = visual_preload(stage, owner_root=HERE)
    stage.GetRootLayer().Save()
    from .geometry import inspect_routes
    try:
        routes = inspect_routes(stage, seed)
    except Exception as error:
        (output / 'route_failure.json').write_text(json.dumps(dict(
            accepted=False, reason=str(error), seed=seed), indent=2) + '\n')
        raise
    if any(src.digest(path) != digest for path, digest in source_hashes.items()):
        raise RuntimeError('Read-only source layer changed during composition')
    plan = dict(scene=str(scene), source_api='1-2', source_episode=manifest['episode'], seed=seed,
                role=role, robot_initialization=manifest['robot_initialization'], robot_base=manifest['robot_base'],
                randomization=record, settle_seconds=src.SETTLE_SECONDS['1-2'], inspection=inspection,
                approved_dynamic_body=dynamic, push_routes=routes, source_layers=source_hashes,
                visual_preload=visual_receipt,
                initial_layer_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                scored_episode=False, physics_settle_verified=False,
                runtime_registered=False, bag_relocation_variant=False)
    (output / 'initial.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    return plan


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--role', choices=['source', 'random'], required=True)
    args = parser.parse_args()
    value = build(args.output, seed=args.seed, role=args.role)
    print(json.dumps({k: value[k] for k in ('scene', 'seed', 'approved_dynamic_body', 'push_routes')}))
