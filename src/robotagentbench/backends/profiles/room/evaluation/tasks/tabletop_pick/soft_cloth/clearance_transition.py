"""One-way cloth activation policy; no Isaac imports, USD writes or scoring.

Trusted physics readers supply ALL scene objects, not just the task target.
This module requests a boundary transaction; it does not pretend that a USD
attribute write proves a live PhysX actor was created. See TRANSITION.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
from typing import Protocol


CLEAR_HOLD_S = 0.5  # User-approved uninterrupted clear interval.


def _finite_vector(value, size):
    return (len(value) == size and all(type(x) in (int, float) and math.isfinite(x)
                                      for x in value))


@dataclass(frozen=True)
class Bounds:
    low: tuple
    high: tuple

    def __post_init__(self):
        object.__setattr__(self, 'low', tuple(self.low))
        object.__setattr__(self, 'high', tuple(self.high))
        if not (_finite_vector(self.low, 3) and _finite_vector(self.high, 3)
                and all(a <= b for a, b in zip(self.low, self.high))):
            raise ValueError('Invalid world-space collision bounds')


@dataclass(frozen=True)
class ClothSurface:
    mesh: str
    points: tuple
    triangles: tuple

    def __post_init__(self):
        object.__setattr__(self, 'points', tuple(tuple(p) for p in self.points))
        object.__setattr__(self, 'triangles', tuple(tuple(t) for t in self.triangles))
        if not self.mesh or not self.points or not self.triangles:
            raise ValueError('Require the actual complete cloth surface')
        if not all(_finite_vector(p, 3) for p in self.points):
            raise ValueError('Invalid world-space cloth points')
        for tri in self.triangles:
            if (len(tri) != 3 or len(set(tri)) != 3
                    or any(type(i) is not int or not 0 <= i < len(self.points) for i in tri)):
                raise ValueError('Invalid cloth triangle')

    @cached_property
    def fingerprint(self):
        # Exact snapshot identity, not a physical tolerance or asset version.
        return hashlib.sha256(json.dumps((self.mesh, self.points, self.triangles),
                                        allow_nan=False).encode()).hexdigest()

    @cached_property
    def bounds(self):
        return Bounds(tuple(min(p[i] for p in self.points) for i in range(3)),
                      tuple(max(p[i] for p in self.points) for i in range(3)))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def overlaps_surface(surface, bounds, clearance_m):
    """Conservative collision-AABB / triangle SAT, including edge touching.

    World AABBs may over-report occupancy for rotated/concave objects; never
    label this a contact-force test. clearance_m is supplied from the existing
    effective collision envelope by the reader, not tuned by this policy.
    """
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('Invalid existing collision envelope')
    if any(bounds.high[i] + clearance_m < surface.bounds.low[i]
           or bounds.low[i] - clearance_m > surface.bounds.high[i] for i in range(3)):
        return False
    center = tuple((lo+hi)/2 for lo, hi in zip(bounds.low, bounds.high))
    half = tuple((hi-lo)/2 + clearance_m for lo, hi in zip(bounds.low, bounds.high))
    basis = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    for indices in surface.triangles:
        vertices = [tuple(x-c for x, c in zip(surface.points[i], center)) for i in indices]
        edges = [tuple(vertices[(i+1) % 3][j]-vertices[i][j] for j in range(3))
                 for i in range(3)]
        axes = (*basis, _cross(edges[0], edges[1]),
                *(_cross(e, axis) for e in edges for axis in basis))
        for axis in axes:
            radius = sum(h*abs(x) for h, x in zip(half, axis))
            projected = [_dot(v, axis) for v in vertices]
            if min(projected) > radius or max(projected) < -radius:
                break
        else:
            return True
    return False


@dataclass(frozen=True)
class ObjectEvidence:
    object_id: str
    bounds: Bounds
    touching_cloth: bool | None
    supported_by_cloth: bool | None


@dataclass(frozen=True)
class ClearanceFrame:
    episode_id: str
    physics_step: int
    time_s: float
    surface: ClothSurface
    inventory_ids: frozenset
    objects: tuple
    inventory_complete: bool
    contacts_complete: bool
    robot_clear: bool | None
    at_physics_boundary: bool


@dataclass(frozen=True)
class ActivationRequest:
    episode_id: str
    physics_step: int
    time_s: float
    clear_since_s: float
    surface: ClothSurface


@dataclass(frozen=True)
class ActivationReceipt:
    episode_id: str
    physics_step: int
    time_s: float
    preserved_surface_fingerprint: str
    static_collider_count: int
    dynamic_cloth_count: int
    live_physics_verified: bool
    scene_reset: bool
    other_bodies_unchanged: bool


class ActivationBackend(Protocol):
    def activate(self, request: ActivationRequest) -> ActivationReceipt:
        """Atomically replace only cloth physics at THIS boundary; no step/reset.

        Preserve current world geometry, topology, visual material, and approved
        soft-body material. No interval may have both colliders or neither.
        A failed transaction must not allow simulation to resume. Receipt must
        use live physics readback, not USD attribute reads alone.
        """
        ...


class ClearToDynamic:
    """Episode-owned, not recreated on composition subtask changes.

    Frame evidence is required each physical step. Missing steps restart the
    hold; invalid/stale evidence invalidates the episode. Furniture supporting
    the cloth is exempt only by explicit instance ID. Never exempt task props
    or robot links. The reader must certify the whole scene inventory and all
    robot/cloth contacts (not only hand or target contacts).
    """
    def __init__(self, *, episode_id, cloth_mesh, physics_dt_s,
                 collision_clearance_m, support_ids):
        if not episode_id or not cloth_mesh:
            raise ValueError('Require episode and cloth identity')
        if not math.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError('Require the existing physics timestep')
        if not math.isfinite(collision_clearance_m) or collision_clearance_m < 0:
            raise ValueError('Require the existing collision envelope')
        self.episode_id, self.cloth_mesh = episode_id, cloth_mesh
        self.dt, self.clearance = physics_dt_s, collision_clearance_m
        self.support_ids = frozenset(support_ids)
        if any(not isinstance(i, str) or not i or i == cloth_mesh for i in self.support_ids):
            raise ValueError('Invalid support exemption')
        self.state = 'static'
        self.reason = 'not_observed'
        self.clear_since = None
        self.last_step = self.last_time = None
        self.geometry = None
        self.inventory = None
        self.pending = None
        self.events = []

    def _fail(self, reason):
        self.state, self.reason, self.pending = 'error', reason, None
        self.clear_since = None
        raise RuntimeError(reason)

    def _blocked(self, reason):
        self.state, self.reason, self.clear_since = 'static', reason, None
        return None

    def observe(self, frame):
        if self.state in ('error', 'closed'):
            raise RuntimeError('Transition controller is ' + self.state)
        self.pending = None  # A request can never survive a newer physics frame.
        if (frame.episode_id != self.episode_id or frame.surface.mesh != self.cloth_mesh
                or type(frame.physics_step) is not int or frame.physics_step < 0
                or not math.isfinite(frame.time_s) or frame.time_s < 0):
            self._fail('Invalid episode/mesh/physical clock evidence')
        if self.last_step is not None:
            delta = frame.physics_step - self.last_step
            if (delta <= 0 or frame.time_s <= self.last_time
                    or not math.isclose(frame.time_s-self.last_time, delta*self.dt,
                                        rel_tol=1e-8, abs_tol=1e-9)):
                self._fail('Stale or inconsistent physical clock')
            if delta != 1:
                self.clear_since = None
        self.last_step, self.last_time = frame.physics_step, frame.time_s
        if self.state == 'dynamic':
            return None  # One-way: replacing objects never freezes the cloth.
        if not all(type(v) is bool for v in (frame.inventory_complete,
                frame.contacts_complete, frame.at_physics_boundary)):
            self._fail('Completeness flags must be measured booleans')
        if frame.robot_clear is not None and type(frame.robot_clear) is not bool:
            self._fail('Robot clearance must be measured')
        ids = [o.object_id for o in frame.objects]
        if (len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids)
                or set(ids) != set(frame.inventory_ids) or self.cloth_mesh in ids):
            self._fail('Incomplete or duplicated scene inventory')
        if not frame.inventory_complete or not frame.contacts_complete:
            return self._blocked('incomplete_scene_or_contact_evidence')
        fingerprint = frame.surface.fingerprint
        if self.geometry is not None and fingerprint != self.geometry:
            return self._blocked('static_cloth_geometry_changed')
        self.geometry = fingerprint
        if self.inventory is not None and self.inventory != frame.inventory_ids:
            self.clear_since = None
        self.inventory = frozenset(frame.inventory_ids)
        blockers = []
        for obj in frame.objects:
            if obj.object_id in self.support_ids:
                continue
            flags = (obj.touching_cloth, obj.supported_by_cloth)
            if not all(v is None or type(v) is bool for v in flags):
                self._fail('Contact/support predicates must be measured booleans')
            if any(v is None for v in flags):
                return self._blocked('unknown_object_evidence:' + obj.object_id)
            if (any(flags) or overlaps_surface(frame.surface, obj.bounds, self.clearance)):
                blockers.append(obj.object_id)
        if blockers:
            return self._blocked('occupied:' + ','.join(sorted(blockers)))
        if frame.robot_clear is not True:
            return self._blocked('robot_contact_or_unknown')
        if not frame.at_physics_boundary:
            return self._blocked('not_at_physics_boundary')
        if self.clear_since is None:
            self.clear_since = frame.time_s
        self.state, self.reason = 'waiting_clear', 'clear_interval_in_progress'
        if frame.time_s - self.clear_since + 1e-12 < CLEAR_HOLD_S:
            return None
        self.pending = ActivationRequest(self.episode_id, frame.physics_step,
                                        frame.time_s, self.clear_since, frame.surface)
        self.state, self.reason = 'ready_to_switch', 'clear_interval_complete'
        return self.pending

    def activate(self, request, backend: ActivationBackend):
        if (self.state != 'ready_to_switch' or request is not self.pending
                or request is None):
            raise RuntimeError('Require the current one-use boundary request')
        self.pending = None
        self.state = 'switching'
        try:
            receipt = backend.activate(request)
            if not (isinstance(receipt, ActivationReceipt)
                    and receipt.episode_id == request.episode_id
                    and receipt.physics_step == request.physics_step
                    and receipt.time_s == request.time_s
                    and receipt.preserved_surface_fingerprint == request.surface.fingerprint
                    and type(receipt.static_collider_count) is int and receipt.static_collider_count == 0
                    and type(receipt.dynamic_cloth_count) is int and receipt.dynamic_cloth_count == 1
                    and receipt.live_physics_verified is True
                    and receipt.scene_reset is False and receipt.other_bodies_unchanged is True):
                raise RuntimeError('Invalid or unverified live activation receipt')
        except Exception as exc:
            self._fail('Activation failed; stop episode: ' + str(exc))
        self.state, self.reason = 'dynamic', 'activation_verified'
        self.events.append(dict(event='cloth_became_dynamic', episode_id=self.episode_id,
            physics_step=request.physics_step, time_s=request.time_s,
            clear_since_s=request.clear_since_s, mesh=self.cloth_mesh,
            surface_fingerprint=request.surface.fingerprint))
        return receipt

    def close(self):
        # Never mutate physics to freeze the cloth as part of scoring cleanup.
        self.pending = None
        self.clear_since = None
        self.state, self.reason = 'closed', 'episode_closed'
