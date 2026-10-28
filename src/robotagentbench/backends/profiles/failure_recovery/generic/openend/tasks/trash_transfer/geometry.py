"""Scene-parameterized garbage reader with original full-bin geometry checks."""
from types import SimpleNamespace
import numpy as np
from evaluation.tasks.shelf_pick.geometry import rotation
from openend.tasks.all_objects_integration.geometry import Geometry, _source_object_roots
from openend.tasks.scene_requirements.source_geometry import floor_paths
from .shapes import shape_points
from .registration import resolve_object


class TrashGeometry(Geometry):
    def __init__(self, stage, manifest, parameters):
        from pxr import Usd, UsdGeom, UsdPhysics
        from evaluation.tasks.trash_disposal.bin_geometry import SceneGeometry
        obj = resolve_object(parameters, manifest)
        self.stage = stage; self.mode = 'trash'
        self.target = obj['path']; self.destination = parameters['destination_path']
        self._source_object_roots = _source_object_roots(manifest, self.target, 'trash')
        self.cabinet = self.destination; self.source = None; self.source_top = None
        self.top = None; self.exclusions = []; self.extra_bodies = {}
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Metre Z-up scene required')
        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1 or scenes[0].GetAttribute('physxScene:enableGPUDynamics').Get() is not False:
            raise ValueError('Existing CPU scene required')
        target = stage.GetPrimAtPath(self.target)
        if not target or not target.IsActive():raise ValueError('Missing active garbage prim')
        bodies = [p for p in Usd.PrimRange(target) if p.HasAPI(UsdPhysics.RigidBodyAPI)
                  and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
        if len(bodies) != 1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:
            details=[dict(path=str(p.GetPath()),kinematic=p.GetAttribute('physics:kinematicEnabled').Get()) for p in bodies]
            raise ValueError('One dynamic garbage body required: '+self.target+'; actual='+str(details))
        self.body = str(bodies[0].GetPath())
        cache = UsdGeom.XformCache(); m = cache.GetLocalToWorldTransform(bodies[0])
        q = m.RemoveScaleShear().ExtractRotationQuat()
        self.initial_pose = (tuple(m.ExtractTranslation()), (q.GetReal(), *q.GetImaginary()))
        self.parts = []; self.cube_parts = set(); visible = []
        for prim,points,colliding,seen in shape_points(stage,self.target):
            local = rotation(self.initial_pose[1]).inv().apply(points-self.initial_pose[0])
            if colliding:
                self.parts.append((str(prim.GetPath()),local))
                if prim.IsA(UsdGeom.Cube):self.cube_parts.add(str(prim.GetPath()))
            if seen:visible.append(local)
        if not self.parts:raise ValueError('Missing garbage collision geometry')
        self.local=np.vstack([points for _,points in self.parts])
        self.visible=np.vstack(visible) if visible else self.local
        self.floors=floor_paths(stage)
        if len(self.floors)!=1:raise ValueError('Unique existing floor required')
        dest=stage.GetPrimAtPath(self.destination)
        if not dest or not dest.IsActive() or not any(p.HasAPI(UsdPhysics.CollisionAPI)
                and p.GetAttribute('physics:collisionEnabled').Get() is not False for p in Usd.PrimRange(dest)):
            raise ValueError('Missing declared bin colliders: '+self.destination)
        self.fixed={p:cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in (*self.floors,self.destination)}
        self.bin=SceneGeometry(stage,SimpleNamespace(target=self.target,destination=self.destination))
        # Whole visible/collider envelope in actual rigid-body coordinates.
        frame=cache.GetLocalToWorldTransform(target).RemoveScaleShear(); q=frame.ExtractRotationQuat()
        world=rotation((q.GetReal(),*q.GetImaginary())).apply(self.bin.target_vertices)+np.asarray(frame.ExtractTranslation())
        self.bin.target_vertices=rotation(self.initial_pose[1]).inv().apply(world-self.initial_pose[0])
