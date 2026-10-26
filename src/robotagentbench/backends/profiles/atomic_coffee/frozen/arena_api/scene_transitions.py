"""Opt-in, trusted cloth-only boundary service. No network-selected callbacks.

This is not a scene loader/reset API. Task-owned evidence decides when; the
approved cloth backend performs and verifies the single physical transaction.
Other tasks return None and do not run this service or any new per-step reads.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PhysicsBoundary:
    episode_id: str
    physics_step: int
    time_s: float


@dataclass(frozen=True)
class ClothSwitchReceipt:
    boundary: PhysicsBoundary
    cloth_root: str
    changed_prim_paths: tuple
    static_colliders: int
    dynamic_bodies: int
    geometry_preserved: bool
    visual_material_preserved: bool
    physics_material_preserved: bool
    other_bodies_unchanged: bool
    live_physics_verified: bool
    world_reset: bool


class ClothTransitionSession:
    """Trusted server-side injection only; own one instance for the episode.

    poll(boundary) is read-only and returns None or an opaque one-use request.
    switch(request, boundary) must return live, not USD-only, verification.
    Backend exceptions permanently invalidate this session and must stop the
    episode. This class does not implement a native PhysX switching backend.
    """
    def __init__(self, *, episode_id, cloth_root, poll, switch):
        parts = cloth_root.split('/') if isinstance(cloth_root, str) else []
        if (not isinstance(episode_id, str) or not episode_id
                or len(parts) < 4 or parts[:3] != ['', 'World', 'Assets']
                or any(not p or p in ('.', '..') for p in parts[3:])
                or not callable(poll) or not callable(switch)):
            raise ValueError('Require a specific cloth subtree and trusted handlers')
        self.episode_id, self.cloth_root = episode_id, cloth_root
        self.poll, self.switch = poll, switch
        self.state = 'waiting'
        self.last_step = self.last_time = None
        self.events = []
        self.error = None

    def after_physics(self, *, episode_id, physics_step, time_s):
        if self.state in ('closed', 'error'):
            raise RuntimeError('Cloth transition session is ' + self.state)
        try:
            if (episode_id != self.episode_id or type(physics_step) is not int or physics_step < 0
                    or isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < 0
                    or (self.last_step is not None and (physics_step <= self.last_step
                                                       or time_s <= self.last_time))):
                raise ValueError('Stale or cross-episode physics boundary')
            self.last_step, self.last_time = physics_step, time_s
            if self.state == 'dynamic':
                return
            boundary = PhysicsBoundary(episode_id, physics_step, time_s)
            request = self.poll(boundary)
            if request is None:
                return
            self.state = 'switching'
            receipt = self.switch(request, boundary)
            if not isinstance(receipt, ClothSwitchReceipt):
                raise ValueError('Expected a cloth-only switch receipt')
            if (receipt.boundary != boundary or receipt.cloth_root != self.cloth_root
                    or not receipt.changed_prim_paths
                    or any(not isinstance(p, str) or not (
                        p == self.cloth_root or p.startswith(self.cloth_root + '/'))
                        or any(x in ('.', '..', '') for x in p.split('/')[1:])
                        for p in receipt.changed_prim_paths)
                    or type(receipt.static_colliders) is not int or receipt.static_colliders != 0
                    or type(receipt.dynamic_bodies) is not int or receipt.dynamic_bodies != 1
                    or receipt.geometry_preserved is not True
                    or receipt.visual_material_preserved is not True
                    or receipt.physics_material_preserved is not True
                    or receipt.other_bodies_unchanged is not True
                    or receipt.live_physics_verified is not True or receipt.world_reset is not False):
                raise ValueError('Cloth-only physical transaction was not verified')
            self.events.append(asdict(receipt))
            self.state = 'dynamic'
        except Exception as exc:
            self.state, self.error = 'error', str(exc)
            raise

    def report(self):
        return dict(kind='cloth_static_to_dynamic.v1', episode_id=self.episode_id,
                    cloth_root=self.cloth_root, state=self.state,
                    events=list(self.events), error=self.error)

    def close(self):
        if self.state != 'error':
            self.state = 'closed'


def for_task(task, *, episode_id, stage):
    """Called once after physics initialization; default is a zero-work no-op.

    A task declaring this hook MUST register an explicitly scoped cloth mesh
    before the first step. There is no generic scene replacement operation.
    """
    hook = getattr(task, 'scene_transitions', None)
    session = hook() if hook is not None else None
    if session is None:
        return None
    if not isinstance(session, ClothTransitionSession) or session.episode_id != episode_id:
        raise ValueError('Invalid trusted cloth transition session')
    root = stage.GetPrimAtPath(session.cloth_root)
    if not root or root.GetCustomDataByKey('arena:transitionRole') != 'cloth':
        raise ValueError('Transition subtree must be explicitly declared as cloth')
    return session
