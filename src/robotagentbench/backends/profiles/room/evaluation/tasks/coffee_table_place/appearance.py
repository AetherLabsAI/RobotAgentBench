"""把一次外观抽样写到场景上。渲染专用，物理一个量都不碰。

抽样规则在 `appearance_sampling`（取自数据方 `place_l1_four_table_neutral_tray_v5`）。
这里只负责"落到 stage 上"，并在落完之后**自证没有动过物理**：

- 写之前记下所有刚体的位姿、质量、密度、碰撞开关与物理材质摩擦；
- 写之后逐项比对，任何一项变化即报错，而不是记一条日志放行；
- 写入的属性名有白名单（`ALLOWED_SUFFIXES`），越界即报错。

这样"外观不影响物理"就不是一句声明，而是每次运行都验一遍的断言。
"""
from pathlib import Path

from . import appearance_sampling as S

TASK = Path(__file__).resolve().parent
SCHEMA = 'arena.coffee_table_place.appearance_applied.v1'
# 允许写入的属性后缀。灯光强度/色温、材质漫反射/粗糙度/金属度与纹理路径。
ALLOWED_SUFFIXES = ('inputs:intensity', 'inputs:colorTemperature', 'inputs:enableColorTemperature',
                    'inputs:diffuseColor', 'inputs:diffuse_color_constant', 'inputs:diffuse_tint',
                    'inputs:roughness', 'inputs:reflection_roughness_constant',
                    'inputs:metallic', 'inputs:metallic_constant', 'inputs:scale',
                    'inputs:file', 'inputs:diffuse_texture')
PHYSICS_MARKERS = ('physics:', 'physx')


def _physics_fingerprint(stage):
    """所有与物理有关的权威量，用来证明外观写入没有碰它们。"""
    from pxr import Usd, UsdGeom, UsdPhysics
    rows = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        entry = {}
        for attr in prim.GetAuthoredAttributes():
            name = attr.GetName()
            if name.startswith(PHYSICS_MARKERS):
                entry[name] = str(attr.Get())
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            xform = UsdGeom.Xformable(prim)
            if xform:
                entry['#xform'] = str(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if entry:
            rows[path] = entry
    return rows


def _coerce(attr, value):
    """按属性自己声明的类型转换。

    同一个 `inputs:scale` 在不同着色器上可能是 Vec3f 也可能是 Vec4f（带 alpha），
    直接 Set 会抛类型不匹配。颜色补 alpha=1，标量取第一个分量。
    """
    from pxr import Gf, Sdf
    name = attr.GetTypeName()
    if name == Sdf.ValueTypeNames.Float4 or name == Sdf.ValueTypeNames.Color4f:
        if isinstance(value, (Gf.Vec3f, Gf.Vec3d, tuple, list)):
            v = list(value)
            return Gf.Vec4f(v[0], v[1], v[2], 1.) if len(v) == 3 else Gf.Vec4f(*v[:4])
        return Gf.Vec4f(value, value, value, 1.)
    if name in (Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Color3f,
                Sdf.ValueTypeNames.Vector3f):
        if isinstance(value, (Gf.Vec3f, Gf.Vec3d)):
            return Gf.Vec3f(value)
        if isinstance(value, (tuple, list)):
            return Gf.Vec3f(*list(value)[:3])
        return Gf.Vec3f(value, value, value)
    if name in (Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Double):
        if isinstance(value, (Gf.Vec3f, Gf.Vec3d, tuple, list)):
            return float(list(value)[0])
        return float(value)
    return value


def _set(stage, prim_path, name, value, written):
    if not name.endswith(ALLOWED_SUFFIXES):
        raise ValueError('Appearance may only write render attributes, not ' + name)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    attr = prim.GetAttribute(name)
    if not attr:
        return False
    attr.Set(_coerce(attr, value))
    written.append(f'{prim_path}.{name}')
    return True


def apply(stage, api_id, seed, *, plan=None):
    """在会话层上施加外观。返回回执，含实际写入的属性与物理不变证明。"""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    plan = plan or S.sample(api_id, seed)
    before = _physics_fingerprint(stage)
    written, skipped = [], []

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for light in plan.lights:
            prim = stage.GetPrimAtPath(light['prim_path'])
            if not prim or not prim.IsValid():
                skipped.append(light['prim_path'])
                continue
            base = prim.GetAttribute('inputs:intensity')
            if base and base.Get() is not None:
                base.Set(float(base.Get())*light['intensity_scale'])
                written.append(light['prim_path']+'.inputs:intensity')
            _set(stage, light['prim_path'], 'inputs:enableColorTemperature', True, written)
            _set(stage, light['prim_path'], 'inputs:colorTemperature',
                 float(light['color_temperature_k']), written)

        for mesh in plan.tray['bound_meshes']:
            material = UsdShade.MaterialBindingAPI(
                stage.GetPrimAtPath(mesh)).GetDirectBinding().GetMaterial() \
                if stage.GetPrimAtPath(mesh) else None
            if not material:
                skipped.append(mesh)
                continue
            rgb = Gf.Vec3f(*plan.tray['linear_rgb'])
            for shader in Usd.PrimRange(material.GetPrim()):
                for name, value in (('inputs:diffuseColor', rgb),
                                    ('inputs:diffuse_color_constant', rgb),
                                    ('inputs:roughness', float(plan.tray['roughness'])),
                                    ('inputs:reflection_roughness_constant',
                                     float(plan.tray['roughness']))):
                    if shader.GetAttribute(name):
                        _set(stage, str(shader.GetPath()), name, value, written)

        # 茶几用的是 MDL 着色器：色调走 `diffuse_tint`（不是 `scale`），
        # 金属度走 `metallic_constant`。它没有独立的 roughness 输入，
        # 所以 roughness 这一项落不下去——记进 unapplied，不静默丢弃。
        table_rgb = Gf.Vec3f(*plan.table['rgb_scale'])
        table_root = stage.GetPrimAtPath(plan.table['material_path'])
        table_fields = {'rgb_scale': False, 'roughness': False, 'metallic': False}
        for shader in (Usd.PrimRange(table_root) if table_root and table_root.IsValid() else ()):
            for name, value, field in (
                    ('inputs:diffuse_tint', table_rgb, 'rgb_scale'),
                    ('inputs:scale', table_rgb, 'rgb_scale'),
                    ('inputs:roughness', float(plan.table['roughness']), 'roughness'),
                    ('inputs:reflection_roughness_constant', float(plan.table['roughness']),
                     'roughness'),
                    ('inputs:metallic', float(plan.table['metallic']), 'metallic'),
                    ('inputs:metallic_constant', float(plan.table['metallic']), 'metallic')):
                if shader.GetAttribute(name) and _set(stage, str(shader.GetPath()), name,
                                                      value, written):
                    table_fields[field] = True

        for tint in plan.tints:
            root = stage.GetPrimAtPath(tint['prim_path'])
            if not root or not root.IsValid():
                skipped.append(tint['prim_path'])
                continue
            value = Gf.Vec3f(*tint['rgb_scale'])
            for shader in Usd.PrimRange(root):
                if shader.GetAttribute('inputs:scale'):
                    _set(stage, str(shader.GetPath()), 'inputs:scale', value, written)

    after = _physics_fingerprint(stage)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise RuntimeError('Appearance changed physics on: ' + ', '.join(changed[:5]))

    unapplied = sorted(k for k, done in table_fields.items() if not done)
    return dict(schema=SCHEMA, api_id=api_id, seed=seed,
                plan=S.record(plan), written=sorted(written), skipped=sorted(set(skipped)),
                table_fields_unapplied=unapplied,
                physics_unchanged=True, physics_prims_checked=len(before))
