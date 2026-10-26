"""Author a new deformable only. Never alter existing assets or scene settings.

Call author() in attach, before physics starts. The caller supplies the current
table-relative placement converted to world coordinates, not old L2 coordinates.
preflight_runtime() is a read-only guard for before_physics_reset; it does NOT
enable GPU dynamics or beta support. There is no runtime animation/attachment.
"""
import json
import math
from pathlib import Path

ROOT = '/World/Assets/TableclothRuleState'
SCHEMAS = (
    'OmniPhysicsSurfaceDeformableSimAPI', 'OmniPhysicsDeformablePoseAPI',
    'OmniPhysicsDeformableBodyAPI', 'PhysxSurfaceDeformableBodyAPI',
    'OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI',
    'OmniPhysicsSurfaceDeformableMaterialAPI',
)


def config():
    return json.loads(Path(__file__).with_name('config.json').read_text())


def grid():
    """Centred, metre-scale +Z-facing triangles, shared by visual and physics."""
    c = config()
    nx, ny = c['cells_x'], c['cells_y']
    points, uv, faces = [], [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            u, v = i / nx, j / ny
            points.append(((u - .5) * c['width_m'], (v - .5) * c['length_m'], 0.))
            uv.append((u, v))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, d = a + 1, a + nx + 1
            faces.extend(((a, b, d + 1), (a, d + 1, d)))
    return points, uv, faces


def author(stage, *, center_world_m, yaw_rad):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    center = tuple(float(v) for v in center_world_m)
    if len(center) != 3 or not all(math.isfinite(v) for v in (*center, yaw_rad)):
        raise ValueError('Require finite centre XYZ and yaw')
    if not stage.GetEditTarget().GetLayer().anonymous:
        raise RuntimeError('Only an anonymous episode layer may be edited')
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise RuntimeError('Require metre-scale Z-up stage; will not change stage metadata')
    if not stage.GetPrimAtPath('/World/Assets'):
        raise RuntimeError('Expected existing /World/Assets parent')
    if stage.GetPrimAtPath(ROOT):
        raise RuntimeError('Cloth path already exists; will not overwrite it')
    for schema in SCHEMAS:
        if not Usd.SchemaRegistry.GetTypeFromSchemaTypeName(schema):
            raise RuntimeError('Missing native schema: ' + schema)
    c = config()
    points, uv, faces = grid()
    # All new opinions live below ROOT, including both material bindings.
    root = UsdGeom.Xform.Define(stage, ROOT)
    root.AddTranslateOp().Set(Gf.Vec3d(*center))
    root.AddRotateZOp().Set(math.degrees(yaw_rad))
    rp = root.GetPrim()
    rp.ApplyAPI('OmniPhysicsDeformableBodyAPI')
    rp.ApplyAPI('PhysxSurfaceDeformableBodyAPI')
    rp.GetAttribute('omniphysics:kinematicEnabled').Set(False)
    rp.GetAttribute('physxDeformableBody:selfCollision').Set(True)
    rp.SetCustomDataByKey('acceptanceStatus', c['status'])
    mesh = UsdGeom.Mesh.Define(stage, ROOT + '/Surface')
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for tri in faces for v in tri])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(uv)
    mp = mesh.GetPrim()
    mp.ApplyAPI('OmniPhysicsSurfaceDeformableSimAPI')
    mp.ApplyAPI('OmniPhysicsDeformablePoseAPI', 'default')
    UsdPhysics.CollisionAPI.Apply(mp)
    mp.GetAttribute('omniphysics:restShapePoints').Set(points)
    mp.GetAttribute('omniphysics:restTriVtxIndices').Set([Gf.Vec3i(*tri) for tri in faces])
    mp.GetAttribute('omniphysics:restBendAnglesDefault').Set('restShapeDefault')
    mp.CreateAttribute('deformablePose:default:omniphysics:purposes', Sdf.ValueTypeNames.TokenArray).Set(['bindPose'])
    mp.CreateAttribute('deformablePose:default:omniphysics:points', Sdf.ValueTypeNames.Point3fArray).Set(points)
    mat = UsdShade.Material.Define(stage, ROOT + '/PhysicsMaterial')
    for schema in SCHEMAS[-3:]:
        mat.GetPrim().ApplyAPI(schema)
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    values = {
        'density': c['areal_density_kg_m2'] / c['thickness_m'],
        'surfaceThickness': c['thickness_m'], 'youngsModulus': c['youngs_modulus_pa'],
        'poissonsRatio': c['poissons_ratio'], 'staticFriction': c['static_friction'],
        'dynamicFriction': c['dynamic_friction'],
    }
    for name, value in values.items():
        if not mat.GetPrim().GetAttribute('omniphysics:' + name).Set(value):
            raise RuntimeError('Could not author material ' + name)
    UsdShade.MaterialBindingAPI.Apply(rp).Bind(mat,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
    visual = UsdShade.Material.Define(stage, ROOT + '/VisualMaterial')
    shader = UsdShade.Shader.Define(stage, ROOT + '/VisualMaterial/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((.65, .61, .52))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.9)
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mp).Bind(visual)
    return {
        'root': ROOT, 'simulation_mesh': str(mesh.GetPath()),
        'vertices': len(points), 'triangles': len(faces),
        'mass_from_area_kg': c['width_m'] * c['length_m'] * c['areal_density_kg_m2'],
        'status': c['status'],
    }


def preflight_runtime(stage, physics_dt, *, deformable_beta_enabled):
    """Checks only; caller must not treat a passed guard as physical validation."""
    if not deformable_beta_enabled:
        raise RuntimeError('Deformable beta is disabled; shared runtime setup requires separate approval')
    if not math.isclose(physics_dt, 1 / 240, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError('Expected existing 240 Hz physics; refusing to override it')
    scene = stage.GetPrimAtPath('/World/PhysicsScene')
    if not scene or scene.GetAttribute('physxScene:enableGPUDynamics').Get() is not True:
        raise RuntimeError('GPU dynamics unavailable; shared runtime setup requires separate approval')
    if scene.GetAttribute('physxScene:broadphaseType').Get() != 'GPU':
        raise RuntimeError('GPU broadphase unavailable; refusing to change the shared scene')
    return True
