"""Make active Family-3 scene geometry physical before the first physics reset.

Static visual meshes use exact triangles, preserving shelf and tray openings.
Existing dynamic compound proxies are retained. Retail props that have no body
or collider receive a body and one convex hull of their authored visual mesh.
No activation, visibility, pose, asset-layer, or robot opinions are changed.
This is collider coverage, not proof of geometric agreement or valid layout.
"""
from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

STOCK_PARENTS = ("/World/Assets/ShelfItems", "/World/Assets/Tier2Items")


def align_shelf_support_geometry(stage):
    """Realize the intended raised shelf as actual visible collision geometry.

    The legacy scene supported stock on invisible slabs 101/52.5 mm above
    the boards. Historical videos raised boards and brackets to those slabs.
    Use the same fixed fixture definition before physics, and retire only the
    redundant invisible support colliders after installing the real surfaces.
    This function assumes the caller has selected a session edit target.
    """
    prims = [p for p in stage.Traverse()
             if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith('/World/Assets/ShelvingUnit/')
             and any(n in str(p.GetPath()) for n in ('shelf_boards_0', 'horizontal_brackets_1'))]
    boards = [p for p in prims if 'shelf_boards_0' in str(p.GetPath())]
    if not boards:
        return {"changed_meshes": [], "replaced_hidden_supports": []}
    cache = UsdGeom.XformCache()
    bands = [(0.4, .9, '/World/Assets/Tier2Overlay', .751),
             (.95, 1.3, '/World/Assets/Tier3Overlay', 1.1825),
             (1.35, 1.65, None, 1.482)]
    shifts, measurements = [], []
    for lo, hi, support_path, nominal_top in bands:
        support = stage.GetPrimAtPath(support_path) if support_path else None
        if support:
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','guide','proxy']).ComputeWorldBound(support).ComputeAlignedRange()
            desired = bounds.GetMax()[2]
        else:
            desired = nominal_top
        z = [cache.GetLocalToWorldTransform(p).Transform(Gf.Vec3d(v))[2]
             for p in boards for v in UsdGeom.Mesh(p).GetPointsAttr().Get()]
        z = [v for v in z if lo <= v <= hi]
        if not z:
            raise RuntimeError(f'Missing authored shelf tier between {lo} and {hi} m')
        top = max(z)
        shifts.append((lo, hi, desired-top))
        measurements.append(dict(band_world_z_m=[lo,hi], before_top_world_m=top,
                                 desired_top_world_m=desired, shift_m=desired-top,
                                 legacy_support_path=support_path))
    changed = []
    for prim in prims:
        mesh = UsdGeom.Mesh(prim)
        world = cache.GetLocalToWorldTransform(prim)
        inv = world.GetInverse()
        output, count = [], 0
        for point in mesh.GetPointsAttr().Get():
            p = world.Transform(Gf.Vec3d(point))
            dz = next((dz for lo,hi,dz in shifts if lo <= p[2] <= hi),0.)
            if abs(dz) > 1e-7:
                p[2] += dz
                output.append(Gf.Vec3f(inv.Transform(p)))
                count += 1
            else:
                output.append(point)
        if count:
            mesh.GetPointsAttr().Set(output)
            # The source's authored extent must track changed vertices, or
            # renderer culling and subsequent geometry bounds can be stale.
            extent = [Gf.Vec3f(*(min(p[i] for p in output) for i in range(3))),
                      Gf.Vec3f(*(max(p[i] for p in output) for i in range(3)))]
            mesh.GetExtentAttr().Set(extent)
            changed.append(dict(path=str(prim.GetPath()), shifted_vertices=count))
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    replaced = []
    for _,_,support_path,_ in bands:
        if not support_path:
            continue
        support = stage.GetPrimAtPath(support_path)
        if not support:
            continue
        if UsdGeom.Imageable(support).ComputeVisibility() != 'invisible':
            raise RuntimeError(f'Legacy support unexpectedly visible: {support_path}')
        if support.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(support).CreateCollisionEnabledAttr(False)
            replaced.append(support_path)
    return dict(changed_meshes=changed, measured_tiers=measurements,
                replaced_hidden_supports=replaced,
                physical_support='Original visible boards use exact static triangle collision; hidden slabs are redundant and disabled.')


def _visible_shape(prim):
    return (prim.IsA(UsdGeom.Gprim)
            and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible"
            and UsdGeom.Imageable(prim).ComputePurpose() not in ("proxy", "guide"))


def _body(prim):
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        prim = prim.GetParent()
    return None


def _enabled_colliders(root):
    return [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]


def apply_scene_collision_profile(stage, target_path=None, target_hull_vertices=None,
                                  preserve_authored_shelf=False):
    changes = {"enabled_static_geometry": [], "added_stock_bodies": [],
               "added_stock_hulls": [], "retained_dynamic_bodies": [],
               "added_dynamic_mesh_colliders": []}
    previous = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        # Editable local instance opinions are necessary for original geometry
        # that was authored as an instance. The referenced asset is unchanged.
        for prim in list(stage.Traverse()):
            if str(prim.GetPath()).startswith(("/World/Assets/", "/World/Background/")) and prim.IsInstance():
                prim.SetInstanceable(False)
        # Imported fixtures already provide aligned visible boards and matching
        # support colliders. Legacy correction must not move their geometry.
        # Collider coverage below remains enabled in either mode.
        changes["shelf_support_geometry"] = (
            {"mode": "preserve_authored", "changed_meshes": [],
             "replaced_hidden_supports": []}
            if preserve_authored_shelf else align_shelf_support_geometry(stage))
        for parent_path in STOCK_PARENTS:
            parent = stage.GetPrimAtPath(parent_path)
            if not parent:
                continue
            for root in parent.GetChildren():
                shapes = [p for p in Usd.PrimRange(root) if _visible_shape(p)]
                if not shapes:
                    continue
                if not root.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = UsdPhysics.RigidBodyAPI.Apply(root)
                    body.CreateRigidBodyEnabledAttr(True)
                    body.CreateKinematicEnabledAttr(False)
                    # Conservative ordinary retail-object mass; visual volume
                    # alone cannot recover the mass of its contents.
                    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.25)
                    changes["added_stock_bodies"].append(str(root.GetPath()))
                if _enabled_colliders(root):
                    continue
                cache = UsdGeom.XformCache()
                inv = cache.GetLocalToWorldTransform(root).GetInverse()
                points, indices, counts = [], [], []
                for prim in shapes:
                    if not prim.IsA(UsdGeom.Mesh):
                        raise RuntimeError(f"Missing stock collider has nonmesh geometry: {prim.GetPath()}")
                    mesh = UsdGeom.Mesh(prim)
                    transform = cache.GetLocalToWorldTransform(prim) * inv
                    start = len(points)
                    points.extend(Gf.Vec3f(transform.Transform(Gf.Vec3d(p))) for p in mesh.GetPointsAttr().Get())
                    counts.extend(mesh.GetFaceVertexCountsAttr().Get())
                    indices.extend(start + i for i in mesh.GetFaceVertexIndicesAttr().Get())
                proxy = UsdGeom.Mesh.Define(stage, root.GetPath().AppendChild("full_scene_collision"))
                proxy.GetPointsAttr().Set(points)
                proxy.GetFaceVertexCountsAttr().Set(counts)
                proxy.GetFaceVertexIndicesAttr().Set(indices)
                proxy.GetPurposeAttr().Set("proxy")
                UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).CreateApproximationAttr("convexHull")
                changes["added_stock_hulls"].append(str(proxy.GetPath()))
        if target_hull_vertices is not None:
            # Controlled cooking-resolution experiment: the exact same input
            # mesh, pose and collision coverage are retained. Only the selected
            # target's generated convex hull receives a larger vertex budget.
            from pxr import PhysxSchema
            if not isinstance(target_hull_vertices, int) or not 4 <= target_hull_vertices <= 255:
                raise ValueError('Target convex hull vertex limit must be an integer in [4, 255]')
            if not target_path or not target_path.startswith(tuple(p + '/' for p in STOCK_PARENTS)):
                raise ValueError('Target hull override requires an explicit stock target')
            prim = stage.GetPrimAtPath(target_path + '/full_scene_collision')
            if (not prim or not prim.IsA(UsdGeom.Mesh)
                    or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexHull'):
                raise ValueError('Target hull override requires its enabled generated convex hull')
            api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            attr = api.GetHullVertexLimitAttr()
            before = attr.Get()
            api.CreateHullVertexLimitAttr(target_hull_vertices)
            changes['target_hull_cooking_override'] = dict(
                path=str(prim.GetPath()), before_vertex_limit=before,
                after_vertex_limit=attr.Get(),
                input_points=len(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                note='Cooking budget only; actual cooked geometry and trajectory still require validation.')
        dynamic_roots = {}
        for prim in list(stage.Traverse()):
            path = str(prim.GetPath())
            if not path.startswith(("/World/Assets/", "/World/Background/")) or not _visible_shape(prim):
                continue
            owner = _body(prim)
            if owner:
                dynamic_roots[str(owner.GetPath())] = owner
                continue
            had_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            collision = UsdPhysics.CollisionAPI.Apply(prim)
            was_enabled = had_collision and collision.GetCollisionEnabledAttr().Get()
            collision.CreateCollisionEnabledAttr(True)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            if was_enabled is not True:
                changes["enabled_static_geometry"].append(path)
        for path, root in dynamic_roots.items():
            if _enabled_colliders(root):
                changes["retained_dynamic_bodies"].append(path)
                continue
            for prim in Usd.PrimRange(root):
                if not _visible_shape(prim):
                    continue
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                changes["added_dynamic_mesh_colliders"].append(str(prim.GetPath()))
        # At 60 Hz a dropped 66 mm can can move farther than its diameter in
        # one step. A physical zero-thickness floor therefore requires swept
        # CCD on both the scene and each dynamic retail body.
        changes["ccd_stock_bodies"] = []
        for path, root in dynamic_roots.items():
            if not path.startswith(tuple(p+'/' for p in STOCK_PARENTS)):
                continue
            root.AddAppliedSchema('PhysxRigidBodyAPI')
            root.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
            changes["ccd_stock_bodies"].append(path)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                prim.AddAppliedSchema('PhysxSceneAPI')
                prim.CreateAttribute('physxScene:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
        # Group filtering is symmetric in PhysX. The legacy basket group
        # filtered the robot even though the robot group's own relation was
        # empty; that would silently defeat full robot/world collision.
        changes["removed_robot_world_group_filters"] = []
        groups = [p for p in stage.Traverse() if p.IsA(UsdPhysics.CollisionGroup)]
        robot_groups = {str(p.GetPath()) for p in groups
                        if any(str(x).startswith('/World/Robot') for x in p.GetRelationship('collection:colliders:includes').GetTargets())}
        for prim in groups:
            relation = UsdPhysics.CollisionGroup(prim).GetFilteredGroupsRel()
            old = relation.GetTargets()
            new = [] if str(prim.GetPath()) in robot_groups else [p for p in old if str(p) not in robot_groups]
            if list(old) != list(new):
                relation.SetTargets(new)
                changes["removed_robot_world_group_filters"].append(str(prim.GetPath()))
        return {"schema": "family3.full_scene_collision_profile.v1", **changes,
                "limitations": ["Existing dynamic proxy dimensions require separate geometry audit.",
                                "Missing retail masses use 0.25 kg pending measured asset data.",
                                "Collision coverage does not certify initial or trajectory clearance."]}
    finally:
        stage.SetEditTarget(previous)
