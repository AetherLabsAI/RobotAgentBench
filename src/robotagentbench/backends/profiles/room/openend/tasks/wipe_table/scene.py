"""Compose the original brush pose and an episode-seeded stain on its table.

No robot, furniture, asset physics, light or camera configuration writes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .registration import TOOL, STAIN, SUPPORT, INSTRUCTION


def compose(stage, data, root, existing_rows, sha):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    requests = data['requirements'].get('wipe_task_requirements', [])
    if not requests:
        return [], [], {}
    if len(requests) != 1:
        raise ValueError('One wiping task per room')
    if any(t['type'] == 'cloth_clear' for t in data['request']['tasks']):
        raise ValueError('Wiping replaces OpenEnd cloth clearing; cannot combine both')
    reference = Usd.Stage.Open(str(Path(data['layout']['scene']).parent/'ID_01_LIVING_STRAIGHT.usdc'))
    table = '/World/Assets/CoffeeTable'
    def points(s):
        mesh = UsdGeom.Mesh(s.GetPrimAtPath(SUPPORT))
        if not mesh:
            raise ValueError('Missing canonical physical tabletop mesh')
        p = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
        return (np.c_[p, np.ones(len(p))] @ matrix)[:, :3]
    a, b = points(reference), points(stage)
    if a.shape != b.shape or not len(a):
        raise ValueError('Furniture geometry differs from approved sampler: CoffeeTable')
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    rotation = u @ vt
    translation = bc-ac @ rotation
    if (np.linalg.det(rotation) < 0 or not np.allclose(rotation[2], [0,0,1], atol=1e-7)
            or np.max(abs(a @ rotation + translation - b)) > 1e-5):
        raise ValueError('Furniture geometry differs from approved sampler: CoffeeTable')
    transform = np.eye(4)
    transform[:3, :3], transform[3, :3] = rotation, translation
    delta = Gf.Matrix4d(*transform.ravel().tolist())
    library = root/'api-protocol/all_assets'
    files = {key: library/key/'asset.usda' for key in ('wipe_tool_hand_brush','table_stain_grid')}
    layout = library/'table_stain_grid/asset_layout.json'
    for path in (*files.values(), layout):
        if not path.is_file():
            raise ValueError('Missing wiping asset dependency: '+str(path))
    # Reuse the original shape ranges and grid/clearing contract. Only this
    # episode's stain appearance changes; source assets remain read-only.
    from .random_stain import generate
    generated=generate(root,Path(stage.GetRootLayer().realPath).parent/'wipe_stain',data['request']['seed'])
    files['table_stain_grid']=generated['asset'];layout=generated['layout']
    # Remove the old cloth only in this new episode. Never edit its source.
    removed = []
    for row in existing_rows:
        if row['id'] == 'tablecloth_dirty':
            stage.OverridePrim(row['path']).SetActive(False)
            removed.append(row['path'])
    # The accepted brush position and stain REGION remain fixed relative to
    # the table; blobs inside the region now follow the original seeded sampler.
    rows = []
    for asset, path, xyz in (('wipe_tool_hand_brush', TOOL, (-2.19,-.07,.45)),
                              ('table_stain_grid', STAIN, (-2.1294,-.4077,.45))):
        if stage.GetPrimAtPath(path):
            raise ValueError('Duplicate wiping prim')
        prim = stage.DefinePrim(path)
        prim.GetReferences().AddReference(str(files[asset]))
        # Preserve authored scale/orientation by moving the asset via a parent.
        current = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        if asset == 'wipe_tool_hand_brush':
            current.SetTranslateOnly(Gf.Vec3d(*xyz))
        else:
            current.SetTranslateOnly(Gf.Vec3d(*xyz))
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(current * delta)
        if asset == 'wipe_tool_hand_brush':
            body = UsdPhysics.RigidBodyAPI(prim)
            if not body or body.GetKinematicEnabledAttr().Get() is True:
                raise ValueError('Brush must retain original dynamic rigid body')
        rows.append(dict(id=asset,path=path,entry=str(files[asset]),source='coffee_table.wipe',
                         task_id=requests[0]['id'],task_role='wipe_tool' if path==TOOL else 'wipe_stain'))
    # Conservative real geometry check: do not silently move unrelated objects.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render'], useExtentsHint=False)
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(STAIN)).ComputeAlignedRange()
    lo, hi = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    for row in existing_rows:
        if row.get('role') == 'workspace_support' or row['path'] in removed:
            continue
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(row['path'])).ComputeAlignedRange()
        p, q = np.asarray(box.GetMin()), np.asarray(box.GetMax())
        if np.all(q[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] < hi[2]+.01 and q[2] > lo[2]:
            raise ValueError('Wiping surface occupied by selected object: '+row['path'])
    tasks = [dict(id=requests[0]['id'],type='wipe_table',name='擦拭茶几污渍',description=INSTRUCTION,
                  target_path=TOOL,object_paths=[TOOL, STAIN],source='coffee_table.wipe',target_count=1,
                  status='scene_bound',score=None,runtime_started=False)]
    group = dict(tool_path=TOOL,stain_path=STAIN,support_path=SUPPORT,
                 layout_file=str(layout),layout_sha256=sha(layout),removed_cloth_paths=removed,
                 initial_profile='atomic_st18_1_seeded_stain_furniture_relative',
                 table_transform=transform.tolist(),maximum_score=6,
                 physics_parameters_changed=False,robot_initialization_changed=False,
                 randomization=generated['record'],randomization_file=str(generated['receipt']),
                 randomization_sha256=sha(generated['receipt']))
    # Keep the existing region polygons/frequency. Only calibrate them through
    # the same furniture transform and bind the stain instead of the old cloth.
    profile = json.loads((root/'openend/tasks/coffee_table_workspaces/local_regions.json').read_text())
    cache_xform = UsdGeom.XformCache()
    base_table = cache_xform.GetLocalToWorldTransform(reference.GetPrimAtPath(table))
    table_inverse = cache_xform.GetLocalToWorldTransform(stage.GetPrimAtPath(table)).GetInverse()
    remap = base_table * delta * table_inverse
    for region in profile['regions']:
        region['polygon_local_m'] = [list(remap.Transform(Gf.Vec3d(*p))) for p in region['polygon_local_m']]
        if 'anchor_center_local_m' in region:
            region['anchor_center_local_m'] = list(remap.Transform(Gf.Vec3d(*region['anchor_center_local_m'])))
        if region['id'] == 'table_cloth':
            region['anchor_asset_id'] = 'table_stain_grid'
            region['anchor_center_local_m'] = list(table_inverse.Transform((bounds.GetMin()+bounds.GetMax())*.5))
    profile['provenance']['wipe_table'] = 'same approved polygons; rigid furniture calibration; stain anchor'
    return rows, tasks, dict(wipe_table=group,coffee_table_workspaces=profile)
