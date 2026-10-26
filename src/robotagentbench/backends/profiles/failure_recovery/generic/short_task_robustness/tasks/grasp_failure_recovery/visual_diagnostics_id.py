"""Read-only evidence for native gripper visibility; never an acceptance bypass.

No camera, renderer, mesh, joint or physical attribute is modified. RGB remains
the original observation. Instance IDs report actual renderer output, not an
inferred guarantee that every finger is visible.
"""
import math


def json_value(value):
    """Bounded metadata serialization, not image-data serialization."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, 'item'):
        return json_value(value.item())
    return str(value)


def instance_summary(payload, expected_shape=(480, 640), robot_root='/World/Robot'):
    """Inspect renderer IDs and path metadata without treating missing IDs as 0."""
    import numpy as np
    if not isinstance(payload, dict) or 'data' not in payload:
        raise ValueError('Instance annotator did not return data and metadata')
    data = np.asarray(payload['data'])
    if data.shape != expected_shape or data.dtype.kind not in 'ui':
        raise ValueError('Invalid instance-ID frame shape or dtype: '+str((data.shape, data.dtype)))
    ids, counts = np.unique(data, return_counts=True)
    info = payload.get('info', {})
    labels = info.get('idToLabels', {}) if isinstance(info, dict) else {}
    if not isinstance(labels, dict):
        raise ValueError('Instance idToLabels is not a mapping')
    rows, unresolved = [], []
    for ident, count in zip(ids, counts):
        key = str(int(ident))
        label = labels.get(key, labels.get(int(ident)))
        row = dict(id=int(ident), pixel_count=int(count), label=json_value(label))
        # The current annotator normally gives a prim-path string. Preserve other
        # metadata verbatim and do not guess names from numeric IDs or RGB.
        paths = [label] if isinstance(label, str) else []
        if isinstance(label, dict):
            paths += [v for k, v in label.items() if k in ('primPath', 'prim_path', 'path')
                      and isinstance(v, str)]
        row['renderer_robot_paths'] = [v for v in paths
                                      if v == robot_root or v.startswith(robot_root+'/')]
        if label is None:
            unresolved.append(int(ident))
        rows.append(row)
    robot_rows = [r for r in rows if r['renderer_robot_paths']]
    return dict(status='observed_ids_with_path_metadata' if labels else 'unmapped_instance_ids',
        observation_only=True, visual_acceptance=False, shape=list(data.shape), dtype=str(data.dtype),
        metadata=json_value(info), rows=rows, unmapped_visible_ids=unresolved,
        robot_paths_observed=bool(robot_rows), robot_pixel_count=sum(r['pixel_count'] for r in robot_rows),
        robot_paths=sorted({v for r in robot_rows for v in r['renderer_robot_paths']}),
        no_robot_ids_is_not_proof_mesh_absent=True)


def attach_head_instance(rep, product):
    """Attach only to the already-existing head RGB render product."""
    ann = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast')
    ann.attach([product])
    return ann


def inventory(stage, measured_joints, camera_path='/World/Robot/head_link2/Head_Camera'):
    """Inventory USD instance proxies and measured-joint FK in original camera.

    Both USD-readback and measured-coordinate FK bounds are recorded. A mismatch
    is evidence to investigate, never repaired here by writing link transforms.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from .render_initials import forward_frames
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = stage.GetPrimAtPath('/World/Robot')
    camera = stage.GetPrimAtPath(camera_path)
    if not root or not camera or not camera.IsA(UsdGeom.Camera):
        raise ValueError('Original robot and head camera are required for visual diagnostics')
    usd_camera = np.array(cache.GetLocalToWorldTransform(camera)).T

    def rigid_ancestor(prim):
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
            prim = prim.GetParent()
        return None

    def mat(prim):
        return np.array(cache.GetLocalToWorldTransform(prim)).T

    base = stage.GetPrimAtPath('/World/Robot/base_link')
    frames = forward_frames(stage, dict(measured_joints), mat(base))
    camera_body = rigid_ancestor(camera)
    if not camera_body or str(camera_body.GetPath()) not in frames:
        raise ValueError('Head camera lacks a resolved measured-FK rigid ancestor')
    fk_camera = frames[str(camera_body.GetPath())] @ np.linalg.inv(mat(camera_body)) @ usd_camera

    def bounds(points):
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('Non-finite or empty visual geometry')
        return dict(min=points[:, :3].min(axis=0).tolist(), max=points[:, :3].max(axis=0).tolist())

    rows = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        # Collider meshes remain listed separately so guide geometry cannot be
        # mistaken for actual rendered finger visuals.
        imageable = UsdGeom.Imageable(prim)
        body = rigid_ancestor(prim)
        row = dict(path=path, active=prim.IsActive(), defined=prim.IsDefined(),
            instance_proxy=prim.IsInstanceProxy(), instance=prim.IsInstance(),
            instanceable=prim.IsInstanceable(), visibility=str(imageable.ComputeVisibility()),
            purpose=str(imageable.ComputePurpose()), rigid_body=str(body.GetPath()) if body else None,
            errors=[])
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot() and not ancestor.IsInstance():
            ancestor = ancestor.GetParent()
        if ancestor and ancestor.IsInstance():
            prototype = ancestor.GetPrototype()
            row.update(instance_root=str(ancestor.GetPath()),
                       prototype=str(prototype.GetPath()) if prototype else None)
            if prim.IsInstanceProxy():
                row['prim_in_prototype'] = str(prim.GetPrimInPrototype().GetPath())
        try:
            pts = prim.GetAttribute('points').Get()
            row['point_count'] = len(pts) if pts is not None else 0
            points = np.asarray(pts, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError('Mesh points are unavailable')
            homogeneous = np.column_stack([points, np.ones(len(points))])
            usd_world = (mat(prim) @ homogeneous.T).T
            row['usd_world_bounds_m'] = bounds(usd_world)
            row['usd_camera_space_bounds_m'] = bounds((np.linalg.inv(usd_camera) @ usd_world.T).T)
            if not body or str(body.GetPath()) not in frames:
                raise ValueError('Visual mesh lacks resolved measured-FK body')
            fk_world = (frames[str(body.GetPath())] @ np.linalg.inv(mat(body)) @ mat(prim) @ homogeneous.T).T
            row['measured_fk_world_bounds_m'] = bounds(fk_world)
            row['measured_fk_camera_space_bounds_m'] = bounds((np.linalg.inv(fk_camera) @ fk_world.T).T)
            row['max_usd_vs_measured_fk_vertex_distance_m'] = float(
                np.linalg.norm(usd_world[:, :3]-fk_world[:, :3], axis=1).max())
        except Exception as error:
            row['errors'].append(type(error).__name__+': '+str(error))
        rows.append(row)
    camera_attributes = {str(a.GetName()): json_value(a.Get()) for a in camera.GetAttributes()}
    return dict(schema='grasp_failure_recovery.visual_inventory.v1', observation_only=True,
        visual_acceptance=False, camera_path=camera_path, camera_attributes=camera_attributes,
        usd_camera_world_matrix=usd_camera.tolist(), measured_fk_camera_world_matrix=fk_camera.tolist(),
        measured_joint_coordinates=dict(measured_joints),
        camera_negative_z_is_forward=True, meshes=rows,
        errors=[dict(path=r['path'], errors=r['errors']) for r in rows if r['errors']],
        physical_or_visual_state_written=False)
