"""Read-only fixed CPU bag geometry. Legacy insertion predicate, not full containment."""
import numpy as np
from openend.tasks.remote_transfer.native_geometry import meshes, hull, top_surface
from evaluation.tasks.shelf_pick.geometry import rotation
from evaluation.tasks.basket_place.cavity_geometry import OpenCavity
from evaluation.tasks.basket_place.geometric_placement import insertion


class SceneGeometry:
    def __init__(self, stage, manifest):
        from pxr import Usd, UsdGeom, UsdPhysics
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Metres Z-up required')
        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1 or scenes[0].GetAttribute('physxScene:enableGPUDynamics').Get() is not False:
            raise ValueError('Existing explicit CPU physics scene required; never switch engines here')
        rows = [o for o in manifest['objects'] if o['id'] == 'wallet']
        if len(rows) != 1: raise ValueError('Exactly one scene-owned wallet required')
        self.target = rows[0]['path']
        self.source = None  # Native initial contacts identify support, never a location whitelist.
        self.destination = '/World/Assets/Task5PackStation/ToteSpawn/Tote'
        self.cabinet = '/World/Assets/Task5PackStation/Collider'
        body = [p for p in Usd.PrimRange(stage.GetPrimAtPath(self.target))
                if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
        if len(body) != 1 or body[0].GetAttribute('physics:kinematicEnabled').Get() is True:
            raise ValueError('Exactly one dynamic wallet body required')
        self.body = str(body[0].GetPath())
        cache = UsdGeom.XformCache()
        m = cache.GetLocalToWorldTransform(body[0]); q = m.RemoveScaleShear().ExtractRotationQuat()
        self.initial_pose = (tuple(m.ExtractTranslation()), (q.GetReal(), *q.GetImaginary()))
        self.parts = [(str(p.GetPath()), rotation(self.initial_pose[1]).inv().apply(points - self.initial_pose[0]))
                      for p, points, colliding, _ in meshes(stage, self.target) if colliding]
        if not self.parts: raise ValueError('Missing target collider')
        self.local = np.vstack([points for _, points in self.parts])
        self.source_top = None
        from openend.tasks.scene_requirements.source_geometry import floor_paths
        self.floors = floor_paths(stage)
        if len(self.floors) != 1: raise ValueError('Unique floor required')
        sim = stage.GetPrimAtPath(self.destination + '/Simulation')
        if (not sim or not sim.HasAPI(UsdPhysics.CollisionAPI)
                or sim.GetAttribute('physics:collisionEnabled').Get() is False
                or sim.GetAttribute('physics:approximation').Get() != 'none'):
            raise ValueError('Fixed open concave CPU collision variant required')
        for p in Usd.PrimRange(stage.GetPrimAtPath(self.destination)):
            if p.HasAPI(UsdPhysics.RigidBodyAPI) or any('Deformable' in str(a) for a in p.GetAppliedSchemas()):
                raise ValueError('Bag must stay fixed, non-deformable CPU geometry')
        mesh = UsdGeom.Mesh(sim)
        if any(n != 3 for n in mesh.GetFaceVertexCountsAttr().Get()): raise ValueError('Triangle bag required')
        self.bag_points = next(points for p, points, c, _ in meshes(stage, self.destination) if str(p.GetPath()) == str(sim.GetPath()) and c)
        self.cavity = OpenCavity(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), len(self.bag_points))
        # Source support is verified again from the full native initial window;
        # path naming alone never establishes support.
        self.fixed = {}
        for path in (self.destination, self.cabinet, str(sim.GetPath()), *self.floors):
            p = stage.GetPrimAtPath(path)
            if not p: raise ValueError('Missing bound furniture ' + path)
            self.fixed[path] = cache.GetLocalToWorldTransform(p)
        self.bag_mesh = str(sim.GetPath())
        self.bag_authored_points = tuple(tuple(p) for p in mesh.GetPointsAttr().Get())
        self.furniture = (self.cabinet, self.destination)
        self.exclusions = []
        self.top = dict(semantics='bag_cavity_not_cabinet', polygon=hull(self.bag_points[:, :2]))

    def footprint(self, pose):
        return hull((rotation(pose.quaternion_wxyz).apply(self.local) + pose.position_m)[:, :2])

    def insertion_evidence(self, pose):
        parts = [rotation(pose.quaternion_wxyz).apply(v) + pose.position_m for _, v in self.parts]
        points = np.vstack(parts)
        # Proven disjoint bounds: cheap rejection only, never AABB success.
        if np.any(points.max(0) < self.bag_points.min(0)) or np.any(points.min(0) > self.bag_points.max(0)):
            return False, False
        witnesses = np.vstack((pose.position_m, points[np.argmin(points[:, 2])], [p.mean(0) for p in parts]))
        diagnostic = dict(witness_names=['target_origin', 'lowest_authored_collider_vertex'] + [p for p, _ in self.parts],
            witness_world_positions=witnesses.tolist(), cavity=self.cavity.inspect(self.bag_points, witnesses))
        if any(x is None for x in diagnostic['cavity']['witness_inside']):
            return False, True  # ambiguous boundary withholds placement, not success
        return insertion(diagnostic)

    def validate_static(self, stage):
        from pxr import UsdGeom
        from openend.tasks.meal_tray.nonmotion_closeout_v1.static_cache import for_stage
        cache = for_stage(stage)
        for p, m in self.fixed.items():
            if cache.matrix(p) != m:
                raise RuntimeError('Bound furniture changed ' + p)
        if cache.mesh_points(self.bag_mesh) != self.bag_authored_points:
            raise RuntimeError('Fixed bag collision geometry changed')
