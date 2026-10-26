"""Read-only scene geometry for every manifest-bound cloth-clear target."""
import numpy as np
from openend.tasks.remote_transfer.native_geometry import meshes, hull, top_surface
from evaluation.tasks.shelf_pick.geometry import rotation
from .registration import resolve
from openend.tasks.scene_requirements.source_geometry import floor_paths


class SceneGeometry:
    def __init__(self, stage, manifest):
        from pxr import Usd, UsdGeom, UsdPhysics
        from evaluation.tasks.tabletop_pick.soft_cloth.clearance_transition import ClothSurface
        from evaluation.tasks.tabletop_pick.book_cloth.source_cloth_geometry import triangles
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Metre-scale Z-up room required')
        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1 or scenes[0].GetAttribute('physxScene:enableGPUDynamics').Get() is not False:
            raise ValueError('Existing CPU physics scene required')
        record = resolve(dict(type='cloth_clear', instance_id='cloth_clear', parameters={}), manifest)
        self.targets = tuple(record['target_paths'])
        self.target = self.cloth = record['cloth_path']  # host anchor, not scoring target
        self.table = record['table_path']
        self.bodies, self.local = {}, {}
        cache = UsdGeom.XformCache()
        for target in self.targets:
            bodies = [p for p in Usd.PrimRange(stage.GetPrimAtPath(target)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if (len(bodies) != 1 or str(bodies[0].GetPath()) != target + '/Asset'
                    or bodies[0].GetAttribute('physics:rigidBodyEnabled').Get() is False
                    or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True):
                raise ValueError('Exactly one dynamic manifest-bound body required: ' + target)
            rows = meshes(stage, target)
            if not any(c for _, _, c, _ in rows):
                raise ValueError('Missing target collider ' + target)
            m = cache.GetLocalToWorldTransform(bodies[0])
            q = m.RemoveScaleShear().ExtractRotationQuat()
            xyz = np.asarray(m.ExtractTranslation())
            self.local[target] = rotation((q.GetReal(), *q.GetImaginary())).inv().apply(
                np.vstack([p for _, p, _, _ in rows]) - xyz)
            self.bodies[target] = str(bodies[0].GetPath())
        self.top = top_surface(stage, self.table)
        # Cloth has its source mesh's slight vertical variation: use its actual
        # projected outline, not the rigid-table planar-top extractor.
        self.cloth_polygon = hull(np.vstack([p for _, p, _, _ in meshes(stage, self.cloth)])[:, :2])
        # Triangles are from actual static cloth, not a bounding-box support test.
        rows = [r for r in meshes(stage, self.cloth) if r[2]]
        if len(rows) != 1:
            raise ValueError('One actual cloth collider required')
        prim, points, _, _ = rows[0]
        mesh = UsdGeom.Mesh(prim)
        topology = triangles(list(mesh.GetFaceVertexCountsAttr().Get()), list(mesh.GetFaceVertexIndicesAttr().Get()))
        self.surface = ClothSurface(str(prim.GetPath()), tuple(tuple(map(float,p)) for p in points),
                                    tuple(tuple(map(int,t)) for t in topology))
        if not any(c for _, _, c, _ in meshes(stage, self.table)):
            raise ValueError('Table lacks collider')
        self.exclusions = tuple(hull(np.vstack([p for _, p, _, _ in meshes(stage, path)])[:, :2])
                                for path in record['exclusion_paths'])
        self.floors = []
        for path in floor_paths(stage):
            p = stage.GetPrimAtPath(path)
            if p and any(c.HasAPI(UsdPhysics.CollisionAPI) and c.GetAttribute('physics:collisionEnabled').Get() is not False
                         for c in Usd.PrimRange(p)):
                self.floors.append(path)
        if len(self.floors) != 1:
            raise ValueError('Unique physical floor required')
        self.fixed = {}
        self.fixed_points = {}
        for path in (*self.top['mesh_paths'], self.cloth, *record['exclusion_paths'], *self.floors):
            for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if any('Deformable' in str(a) for a in p.GetAppliedSchemas()):
                    raise ValueError('Static reader cannot observe deformable support')
                if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False:
                    if p.GetAttribute('physics:kinematicEnabled').Get() is not True:
                        raise ValueError('Dynamic support/exclusion requires native geometry reader: ' + str(p.GetPath()))
                if p.IsA(UsdGeom.Xformable):
                    self.fixed[str(p.GetPath())] = cache.GetLocalToWorldTransform(p)
                if p.IsA(UsdGeom.Mesh) and path == self.cloth:
                    self.fixed_points[str(p.GetPath())] = tuple(map(tuple, UsdGeom.Mesh(p).GetPointsAttr().Get()))

    def footprint(self, target, pose):
        return hull((rotation(pose['quaternion_wxyz']).apply(self.local[target]) + pose['position_m'])[:, :2])

    def validate_static(self, stage):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.static_cache import for_stage
        cache = for_stage(stage)
        for path, m in self.fixed.items():
            p = stage.GetPrimAtPath(path)
            if not p or cache.matrix(path) != m:
                raise RuntimeError('Bound support or exclusion moved: ' + path)
        for path, points in self.fixed_points.items():
            if cache.mesh_points(path) != points:
                raise RuntimeError('Static cloth changed')
