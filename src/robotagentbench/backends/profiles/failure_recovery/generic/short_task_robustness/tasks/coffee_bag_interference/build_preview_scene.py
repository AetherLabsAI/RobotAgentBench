"""Task-owned composition of unchanged atomic 1-2 initialization helpers.

The original initializer's output guard is retained. No monkey patch, source
write, physical-parameter opinion, or robot preparation override is used.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def visual_preload(stage, *, owner_root=None):
    """Author visual instance metadata before the renderer first opens this layer.

    The approved failure-recovery comparison established that a late session
    update is insufficient. Only this coffee variant's new scene layer changes;
    all composed attributes, relationships, assets and physical APIs stay intact.
    """
    from pxr import Usd, UsdPhysics
    owner = HERE.resolve() if owner_root is None else Path(owner_root).resolve()
    allowed = {HERE.resolve(), (HERE.parent / 'coffee_machine_push_interference').resolve(),
               Path('/mnt/agent-bench-data/lingjun/short_task_robustness/tasks/coffee_button_lighting_interference').resolve()}
    if owner not in allowed:
        raise ValueError('Visual preload owner is not an approved coffee robustness task')
    layer = stage.GetRootLayer()
    if (not layer.realPath or not Path(layer.realPath).resolve().is_relative_to(owner)
            or Path(layer.realPath).resolve().parent == owner
            or stage.GetEditTarget().GetLayer() != layer
            or not stage.GetSessionLayer().empty):
        raise ValueError('Visual preload requires a fresh task-owned root edit target')
    robot = stage.GetPrimAtPath('/World/Robot')
    if not robot:
        raise ValueError('Robot visual root is missing')

    def properties():
        # Include instance proxies on both sides, and retain time samples.
        # This covers transforms, cameras, topology and materials as well as
        # physics/drive/collision properties; none may change with this repair.
        return {str(prop.GetPath()): (
            repr(prop.Get()), tuple((t, repr(prop.Get(t))) for t in prop.GetTimeSamples()))
            if isinstance(prop, Usd.Attribute) else repr(prop.GetTargets())
            for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
            for prop in prim.GetProperties()}

    targets = []
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not (prim.IsInstance() and (path.endswith('/visuals') or '/visuals/' in path)):
            continue
        if prim.IsInstanceProxy():
            raise ValueError('Cannot author nested visual instance proxy: ' + path)
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child.IsA(UsdPhysics.Joint) or any(child.HasAPI(api) for api in (
                    UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI,
                    UsdPhysics.ArticulationRootAPI)):
                raise ValueError('Visual subtree contains physical prim: ' + str(child.GetPath()))
        targets.append(path)
    if not targets:
        raise ValueError('No independent robot visual instances to repair')
    before = properties()
    original = layer.ExportToString()
    try:
        for path in targets:
            stage.OverridePrim(path).SetInstanceable(False)
        if before != properties():
            raise ValueError('Composed scene properties changed during visual preload')
        if any(stage.GetPrimAtPath(path).IsInstance() for path in targets):
            raise ValueError('Visual preload did not compose')
    except Exception:
        layer.ImportFromString(original)
        raise
    return dict(schema='coffee_bag_interference.visual_preload.v1',
        before_renderer_first_load=True, task_root_layer_only=True,
        instanceable_disabled_paths=targets, source_assets_written=False,
        composed_properties_unchanged=True, physics_and_material_properties_unchanged=True,
        camera_renderer_parameters_changed=False, visual_acceptance=False)


def build(output, seed=4126, role='random'):
    from pxr import Gf, Sdf, Usd, UsdGeom
    from arena_api import l1_scene
    from evaluation.tasks.coffee_machine import initial_state as src, randomization
    from evaluation.tasks.coffee_machine.specs import (
        STATION, MACHINE, BUTTON, CUP, BAG, BEANS, SHARED_ASSETS, ROBOT, episodes)
    output = Path(output).resolve()
    if output == HERE or not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh output inside coffee_bag_interference required')
    manifest, binding, _ = src.package('1-2', episodes('1-2')[0])
    output.mkdir(parents=True)
    scene = output/'initial.usda'
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
            SHARED_ASSETS/f'cup_{style}.usda', output))])
        cup.SetCustomDataByKey(src.STYLE_KEY, style)
    cup_style(binding['cup_style'])
    for path,key in [(MACHINE,'machine'), (CUP,'cup'), (BAG,'bag')]:
        src.place(stage,path,binding['world'][key])
    src.place(stage,BUTTON,binding['button_local'])
    for i,matrix in enumerate(binding['beans']):
        src.place(stage,f'{BEANS}/Bean_{i:03d}',matrix)
    if role not in ('source','random'): raise ValueError('Source or random initial state required')
    record = (randomization.apply(stage,'1-2',manifest,seed,set_cup_style=cup_style)
              if role == 'random' else manifest['facts']['randomization'])
    l1_scene.verify(stage)
    inspection = src.inspect_stage(stage,'1-2')
    base = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT)).ExtractTranslation()
    if max(abs(a-b) for a,b in zip(base,manifest['robot_base']['root_xyz'])) > 1e-9:
        raise ValueError('Source robot root changed')
    plan = dict(scene=str(scene),source_api='1-2',source_episode=manifest['episode'],seed=seed,role=role,
        robot_initialization=manifest['robot_initialization'],robot_base=manifest['robot_base'],
        randomization=record,settle_seconds=src.SETTLE_SECONDS['1-2'],inspection=inspection,
        source_layers={p.realPath:src.digest(p.realPath) for p in stage.GetUsedLayers() if p.realPath},
        physical_parameters_overridden=False,scored_episode=False)
    if role=='random':
        from .joint_initialization import author_pair
        try:joint=author_pair(stage,plan)
        except ValueError as error:
            stage.GetRootLayer().Save()
            (output/'joint_initialization_failure.json').write_text(json.dumps(dict(
                accepted=False,reason=str(error),seed=seed,source_proposal=record),indent=2)+'\n')
            raise
        plan['joint_initialization']=joint
        plan['relocation_pair']=joint['pair']
        plan['randomization_scope']='source machine/cup/yaw draws retained; variant paired bag positions'
        plan['inspection']=src.inspect_stage(stage,'1-2')
        from .route_options import reserve_routes
        plan['relocation_options'] = reserve_routes(stage, plan, joint['reservation'])
    plan['visual_preload'] = visual_preload(stage)
    stage.GetRootLayer().Save()
    plan['source_layers']={p.realPath:src.digest(p.realPath) for p in stage.GetUsedLayers() if p.realPath}
    (output/'initial.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=4126)
    p.add_argument('--role',choices=['source','random'],default='random')
    a=p.parse_args();plan=build(a.output,a.seed,a.role);print(json.dumps({k:plan[k] for k in ['scene','role','source_episode','seed','inspection']}))
