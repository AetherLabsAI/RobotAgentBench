"""Explicit, episode-owned cloth assistance; never used for ordinary rigid props.

Trusted contact readers supply current opposing-pad geometry evidence. A command
to close is insufficient. Native constraints act on dynamic robot rigid links;
this module never writes cloth vertices, robot poses, gripper targets or scores.
Native mid-simulation constraint updates still require GPU acceptance testing.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PinchEvidence:
    hand: str
    cloth_mesh: str | None
    vertex: int | None
    measured_closed: bool
    opening: bool
    opposing_contacts: bool
    pad_interpenetration: bool


class ClothGraspAssist:
    """One instance for a full episode, shared across all composite subtasks.

    backend must implement attach(hand, mesh, vertex), release(hand).
    A task switch must NOT recreate/close this component. reset/episode close
    releases all constraints. Sources of PinchEvidence are trusted physics code,
    never API request fields or task success predicates.
    """
    def __init__(self, *, allowed_meshes, hands, backend):
        self.allowed = frozenset(allowed_meshes)
        self.hands = frozenset(hands)
        if not self.allowed or not self.hands:
            raise ValueError('Cloth assistance must be explicitly scoped')
        self.backend = backend
        self.held = {}
        self.events = []
        self.last_step = -1
        self.last_time = -1.
        self.closed = False

    def update(self, *, physics_step, time_s, samples):
        if (self.closed or type(physics_step) is not int or physics_step <= self.last_step
                or not math.isfinite(time_s) or time_s < 0 or time_s <= self.last_time):
            raise ValueError('Require fresh increasing physics evidence')
        samples = tuple(samples)
        if len(samples) != len(self.hands) or {s.hand for s in samples} != self.hands:
            raise ValueError('Require exactly one fresh sample per registered hand')
        for s in samples:
            if not all(type(v) is bool for v in (
                    s.measured_closed, s.opening, s.opposing_contacts, s.pad_interpenetration)):
                raise ValueError('Contact predicates must be measured booleans')
            if s.cloth_mesh is not None and s.cloth_mesh not in self.allowed:
                raise ValueError('Object is not registered for cloth assistance')
            if s.vertex is not None and (type(s.vertex) is not int or s.vertex < 0):
                raise ValueError('Invalid cloth vertex')
        for s in samples:
            if s.hand in self.held:
                if s.opening or not s.measured_closed:
                    self.backend.release(s.hand)
                    old = self.held.pop(s.hand)
                    self.events.append(dict(event='release', hand=s.hand, mesh=old[0],
                        vertex=old[1], physics_step=physics_step, time_s=time_s))
                # An existing ideal pinch is not re-anchored to other vertices.
                continue
            if (s.cloth_mesh is not None and s.vertex is not None and s.measured_closed
                    and not s.opening and s.opposing_contacts and not s.pad_interpenetration):
                self.backend.attach(s.hand, s.cloth_mesh, s.vertex)
                self.held[s.hand] = (s.cloth_mesh, s.vertex)
                self.events.append(dict(event='attach', hand=s.hand, mesh=s.cloth_mesh,
                    vertex=s.vertex, physics_step=physics_step, time_s=time_s,
                    model='assisted_single_vertex_native_constraint'))
        self.last_step, self.last_time = physics_step, time_s

    def close(self):
        if self.closed:
            return
        for hand in tuple(self.held):
            self.backend.release(hand)
            self.held.pop(hand)
        self.closed = True


class NativeVertexAttachments:
    """USD native two-way attachment authoring, not a verified live backend yet."""
    ROOT = '/World/ArenaClothGraspAssist'

    def __init__(self, stage, *, hands, allowed_meshes):
        from pxr import Usd, UsdGeom, UsdPhysics
        if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
            raise RuntimeError('Cloth attachments require episode session layer')
        if stage.GetPrimAtPath(self.ROOT):
            raise RuntimeError('Cloth attachment scope already exists')
        if not Usd.SchemaRegistry.GetTypeFromSchemaTypeName('OmniPhysicsVtxXformAttachment'):
            raise RuntimeError('Native vertex attachment schema unavailable')
        if not hands or not allowed_meshes:
            raise ValueError('Require explicit cloth and hand registrations')
        for hand, path in hands.items():
            if hand not in ('left', 'right'):
                raise ValueError('Unknown hand identifier')
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    or prim.GetAttribute('physics:rigidBodyEnabled').Get() is False
                    or prim.GetAttribute('physics:kinematicEnabled').Get() is True):
                raise ValueError('Attachment hand must be a dynamic rigid link')
        for path in allowed_meshes:
            prim = stage.GetPrimAtPath(path)
            if not UsdGeom.Mesh(prim) or not prim.HasAPI('OmniPhysicsSurfaceDeformableSimAPI'):
                raise ValueError('Only surface deformable simulation meshes may be registered')
        self.stage, self.layer = stage, stage.GetSessionLayer()
        self.hands, self.allowed = dict(hands), frozenset(allowed_meshes)
        self.attachments = {}
        stage.DefinePrim(self.ROOT, 'Scope')
        for hand, path in self.hands.items():
            prim = stage.DefinePrim(self.ROOT + '/' + hand, 'OmniPhysicsVtxXformAttachment')
            prim.GetAttribute('omniphysics:attachmentEnabled').Set(False)
            prim.GetRelationship('omniphysics:src1').SetTargets([path])
            self.attachments[hand] = prim

    def attach(self, hand, mesh, vertex):
        from pxr import Gf, Usd, UsdGeom
        if hand not in self.hands or mesh not in self.allowed:
            raise ValueError('Attachment outside explicit registration')
        points = UsdGeom.Mesh(self.stage.GetPrimAtPath(mesh)).GetPointsAttr().Get()
        if type(vertex) is not int or not 0 <= vertex < len(points):
            raise ValueError('Attachment vertex outside simulation mesh')
        cache = UsdGeom.XformCache()
        world = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(mesh)).Transform(Gf.Vec3d(points[vertex]))
        local = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.hands[hand])).GetInverse().Transform(world)
        if not all(math.isfinite(x) for x in local):
            raise ValueError('Invalid attachment anchor')
        with Usd.EditContext(self.stage, self.layer):
            prim = self.attachments[hand]
            if prim.GetAttribute('omniphysics:attachmentEnabled').Get():
                raise RuntimeError('Cannot re-anchor a held cloth')
            prim.GetRelationship('omniphysics:src0').SetTargets([mesh])
            prim.GetAttribute('omniphysics:vtxIndicesSrc0').Set([vertex])
            prim.GetAttribute('omniphysics:localPositionsSrc1').Set([Gf.Vec3f(local)])
            prim.GetAttribute('omniphysics:attachmentEnabled').Set(True)

    def release(self, hand):
        from pxr import Usd
        with Usd.EditContext(self.stage, self.layer):
            self.attachments[hand].GetAttribute('omniphysics:attachmentEnabled').Set(False)
