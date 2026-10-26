"""Compare gripper physical properties to the referenced robot.usda, not task defaults.

This guard deliberately rejects divergent scenes; it never repairs a running
articulation by rewriting its physical properties. Targets and measured state
are excluded. Call assert_unchanged before reset and each physics step.
"""
from pathlib import Path
from pxr import Usd, UsdShade, Sdf

_PREFIXES = ('physics:', 'physx', 'drive:')
_DYNAMIC = {'physics:velocity', 'physics:angularVelocity'}


def _fixed(name):
    return (name.startswith(_PREFIXES) and name not in _DYNAMIC
            and not name.startswith('physxContactReport:')
            and not name.endswith((':targetPosition', ':targetVelocity')))


def _value(value):
    # USD values have stable, type-preserving reprs within the same runtime.
    return repr(value)


def _properties(prim):
    result = {'type': prim.GetTypeName(), 'active': prim.IsActive(),
              'apis': tuple(sorted(x for x in prim.GetAppliedSchemas()
                                   if x.startswith(('Physics', 'Physx')) and x != 'PhysxContactReportAPI'))}
    for attr in prim.GetAttributes():
        if _fixed(attr.GetName()):
            if attr.GetNumTimeSamples():
                raise RuntimeError('Time-varying gripper physical property: ' + str(attr.GetPath()))
            result[attr.GetName()] = _value(attr.Get())
    for rel in prim.GetRelationships():
        if rel.GetName().startswith(('physics:', 'physx')) and not rel.GetName().startswith('physxContactReport:'):
            result[rel.GetName()] = tuple(str(x) for x in rel.GetForwardedTargets())
    return result


def snapshot(stage, root='/World/Robot'):
    result = {}
    robot = stage.GetPrimAtPath(root)
    if not robot:
        raise RuntimeError('Missing robot: ' + root)
    for prim in Usd.PrimRange(robot, Usd.PrimAllPrimsPredicate):
        path = str(prim.GetPath())
        if not any('gripper_' in part for part in path.split('/')):
            continue
        values = _properties(prim)
        # Effective binding also catches ancestor/environment material overrides.
        if prim.HasAPI(UsdShade.MaterialBindingAPI) or prim.GetAttribute('physics:collisionEnabled'):
            material, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            values['bound_material'] = str(material.GetPath()) if material else None
            if material:
                values['material_properties'] = _properties(material.GetPrim())
        result[path] = values
    if not result:
        raise RuntimeError('No gripper found in robot asset')
    return result


class GripperAuthority:
    def __init__(self, stage, root='/World/Robot'):
        self.stage = stage
        self.root = root
        robot = stage.GetPrimAtPath(root)
        candidates = [(spec.layer.realPath, spec.path.StripAllVariantSelections()) for spec in robot.GetPrimStack()
                      if Path(spec.layer.realPath).name == 'robot.usda']
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise RuntimeError('Expected one referenced robot.usda, found ' + repr(candidates))
        self.asset_path, asset_root = candidates[0]
        self.reference = Usd.Stage.CreateInMemory()
        self.reference.DefinePrim(root).GetReferences().AddReference(self.asset_path, asset_root)
        self.expected = snapshot(self.reference, root)

    def differences(self):
        actual = snapshot(self.stage, self.root)
        return {path: {'asset': self.expected.get(path), 'scene': actual.get(path)}
                for path in sorted(set(self.expected) | set(actual))
                if self.expected.get(path) != actual.get(path)}

    def assert_unchanged(self):
        differences = self.differences()
        if differences:
            raise RuntimeError('Environment/task overrides robot.usda gripper: ' + repr(differences))

    def protect_simulation(self, simulation):
        self.assert_unchanged()
        # Bind before reset: no physics step may consume a divergent USD stage.
        for name in ('reset', 'step'):
            original = getattr(simulation, name)
            def checked(*args, _original=original, **kwargs):
                self.assert_unchanged()
                return _original(*args, **kwargs)
            setattr(simulation, name, checked)


def remove_scene_gripper_overrides(stage, root='/World/Robot'):
    """Before physics initialization, discard scene-owned fixed gripper opinions.

    Reference layers are never edited. The asset owns physical calibration;
    commanded targets and measured state remain independent of calibration.
    Later task writes are rejected by GripperAuthority.
    """
    removed = []
    for layer in stage.GetLayerStack():
        paths = []
        layer.Traverse(Sdf.Path(root), lambda p: paths.append(p))
        for path in paths:
            spec = layer.GetPrimAtPath(path)
            if not spec or not any('gripper_' in part for part in str(path).split('/')):
                continue
            for prop in list(spec.properties.values()):
                if _fixed(prop.name) or prop.name.startswith('material:binding'):
                    removed.append(str(prop.path))
                    spec.RemoveProperty(prop)
            if spec.HasInfo('apiSchemas'):
                spec.ClearInfo('apiSchemas')
    return removed


def disable_legacy_grip_compliance(stage):
    """Migrate old House rigid-object contact materials before physics starts.

    Either member of a contact pair can enable compliance. These three legacy
    materials must not silently re-enable the old gripper-contact spring.
    Deformable materials and cushion/support models are outside this migration.
    """
    changed = []
    for name in ('GripMat', 'Tier2BoxMat', 'PringMat'):
        prim = stage.GetPrimAtPath('/World/PhysicsMaterials/' + name)
        if not prim:
            continue
        for suffix in ('Stiffness', 'Damping'):
            attr = prim.GetAttribute('physxMaterial:compliantContact' + suffix)
            if attr and attr.Get() not in (None, 0.0):
                changed.append((str(attr.GetPath()), float(attr.Get())))
                attr.Set(0.0)
    return changed
