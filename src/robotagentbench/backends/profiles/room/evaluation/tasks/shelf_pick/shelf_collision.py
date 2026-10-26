"""Read-only shelf identity and unchanged contact-point support exemption."""
import numpy as np
from .initial_state import SUPPORT

SHELF = '/World/Assets/ShelvingUnit'
RULE_VERSION = 'arena.shelf_pick.process.v4'


class SupportFace:
    """The original slab's top face, measured in its own frame.

    A loaded, upward-facing contact inside that face supports a target whose
    measured geometry centre is above it. Edges, underside and other slabs do
    not qualify. The epsilon excludes the exact rim, not a broad safety band.
    """
    def __init__(self, world_transform, size, tolerance_m=.003, support=SUPPORT):
        self.world = np.asarray(world_transform, dtype=float)
        if self.world.shape != (4,4) or not np.isfinite(self.world).all():
            raise ValueError('Invalid support transform')
        self.inverse = np.linalg.inv(self.world)
        self.scale = np.linalg.norm(self.world[:3,:3], axis=1)
        if min(self.scale) <= 0 or not np.isfinite(size) or size <= 0:
            raise ValueError('Invalid support extent')
        self.half = float(size)/2
        self.tolerance_m = tolerance_m
        self.support = support

    def classify(self, point, normal_on_target, target_center_world):
        if target_center_world is None:
            raise RuntimeError('Missing current target pose for support exemption')
        point = np.asarray([*point,1.]) @ self.inverse
        center = np.asarray([*target_center_world,1.]) @ self.inverse
        normal = np.asarray(normal_on_target) @ (self.world[:3,:3]/self.scale[:,None]).T
        norm = np.linalg.norm(normal)
        if not np.isfinite(np.r_[point,center,normal]).all() or norm <= 0:
            raise RuntimeError('Invalid support contact geometry')
        normal /= norm
        facts = dict(point_local=point[:3].tolist(), normal_on_target_local=normal.tolist(),
                     target_center_local=center[:3].tolist(),
                     on_original_top=bool(abs(point[2]-self.half)*self.scale[2] <= self.tolerance_m),
                     inside_face=bool(all(abs(point[i])*self.scale[i] < self.half*self.scale[i]-1e-6 for i in (0,1))),
                     upward_load=bool(normal[2] > .5),
                     target_above_face=bool(center[2] > self.half))
        exempt = all(facts[k] for k in ('on_original_top','inside_face','upward_load','target_above_face'))
        return exempt, facts

    def description(self):
        return dict(support=self.support,world_transform=self.world.tolist(),size=2*self.half,
                    face_tolerance_m=self.tolerance_m,edge_epsilon_m=1e-6,
                    normal_min_dot=.5,scope='target only, original top-face load with target geometry centre above face')


def bind_shelf(stage, tolerance_m, support=SUPPORT):
    from pxr import UsdGeom, UsdPhysics
    from arena_api.scene_obstacles import inventory
    rows = [r for r in inventory(stage) if r['category']=='shelf']
    if not rows: raise ValueError('Missing shelf collision inventory')
    transforms = {}; components = []
    for row in rows:
        prim = stage.GetPrimAtPath(row['path'])
        enabled = prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False
        if enabled != row['enabled']: raise ValueError('Shelf differs from current public collision policy: '+row['path'])
        components.append(dict(path=row['path'],enabled=enabled,excluded_reason=row['excluded_reason'],type=row['type']))
        if enabled: transforms[row['path']] = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    if support not in transforms: raise ValueError('Original support not in shelf colliders: '+support)
    support_prim = stage.GetPrimAtPath(support)
    if not support_prim.IsA(UsdGeom.Cube): raise ValueError('Unsupported original support geometry')
    face = SupportFace(transforms[support],UsdGeom.Cube(support_prim).GetSizeAttr().Get(),tolerance_m,support)
    return tuple(transforms), face, dict(rule_version=RULE_VERSION,billing_object=SHELF,original_support=support,components=components,
        normal_support_exemption=face.description(),physics_modified=False)
