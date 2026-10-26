"""Explicit opt-in experiment: bind only the judged beef colliders to GripMat."""
import json
from pathlib import Path

def apply(stage, output):
    from pxr import Usd, UsdPhysics, UsdShade
    target = '/World/Assets/MicrowaveBeefbox'
    material_path = '/World/PhysicsMaterials/GripMat'
    material = UsdShade.Material.Get(stage, material_path)
    if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('Existing GripMat missing')
    params = {a.GetName(): str(a.Get()) for a in material.GetPrim().GetAttributes()}
    colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(target))
                 if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute('physics:collisionEnabled').Get()]
    if len(colliders) != 876:
        raise ValueError('Unexpected beef collider inventory')
    before = {}
    for p in colliders:
        m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
        before[str(p.GetPath())] = str(m.GetPath()) if m else None
    if any(v is not None for v in before.values()):
        raise ValueError('Baseline beef binding differs; do not overwrite it')
    physics_before = {str(a.GetPath()): str(a.Get()) for p in stage.Traverse() for a in p.GetAttributes()
                      if a.GetName().startswith(('physics:', 'physx', 'drive:'))}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for p in colliders:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    for p in colliders:
        m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
        if str(m.GetPath()) != material_path:
            raise ValueError('Beef material failed to resolve')
    physics_after = {str(a.GetPath()): str(a.Get()) for p in stage.Traverse() for a in p.GetAttributes()
                     if a.GetName().startswith(('physics:', 'physx', 'drive:'))}
    if physics_before != physics_after:
        raise ValueError('Non-binding physical parameter changed')
    receipt = dict(experiment='beef_gripmat_v1', target=target, material=material_path,
                   collider_count=len(colliders), before_unbound=True, all_resolved=True,
                   material_attributes=params, physics_attributes_unchanged=True,
                   source_assets_modified=False, layer='unsaved_session', promoted_default=True,
                   default_scope='refrigerator_take_only', selection='876_convex_gripmat_20260914')
    with (Path(output)/'beef_gripmat_binding.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    return receipt
