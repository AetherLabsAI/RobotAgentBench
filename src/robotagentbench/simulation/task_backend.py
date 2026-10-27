"""Trusted task-owner bindings, never constructed from agent RPC arguments."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class NativeBindings:
    mobile_controller: Callable
    collision_guard: Callable
    robot_alignment: object

    def __post_init__(self):
        if not callable(self.mobile_controller) or not callable(self.collision_guard):
            raise ValueError('Original native controller and collision guard are required')
        if any(not callable(getattr(self.robot_alignment, name, None))
               for name in ('apply', 'verify', 'verify_native_forces')):
            raise ValueError('Original robot parameter application and verification are required')


@dataclass(frozen=True)
class TaskBackend:
    registry: object
    verify_scene_sources: Callable
    build_short_session: Callable
    load_native_bindings: Callable
    admit_room: Callable | None = None
    build_room: Callable | None = None
    accept_room: Callable | None = None
    wrap_coding_host: Callable | None = None
    exec_special_owner: Callable | None = None
    public_context: Callable | None = None
    scoring_output: Callable | None = None

    def __post_init__(self):
        for name in ('verify_scene_sources', 'build_short_session', 'load_native_bindings'):
            if not callable(getattr(self, name)):
                raise ValueError('A trusted callable is required: ' + name)
        for name in ('admit_room', 'build_room', 'accept_room', 'wrap_coding_host', 'exec_special_owner', 'public_context', 'scoring_output'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError('A trusted callable is required: ' + name)
