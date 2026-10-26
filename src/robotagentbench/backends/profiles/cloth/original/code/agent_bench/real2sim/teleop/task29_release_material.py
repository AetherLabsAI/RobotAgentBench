"""Frozen Task29 contact policy for release and thin-floor landing.

The authored ``GripMat`` gives both the PaperCup and distal pads compliant
contact stiffness/damping.  Task29 evidence showed that this spring contact
can retain the cup on one fully opened pad.  This contract changes no friction
or restitution: it removes only compliant contact, applies before the first
physics reset, and is scoped to the Task29 target plus the active hand's two
link4 collision shapes.  The 6 mm geometry-derived bin floor is thinner than
one impact-time discrete step, so the same pre-reset policy also requires the
already CCD-enabled physics scene and enables CCD on the PaperCup rigid body.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


SCHEMA = "real2sim.task29_release_material.v2"


@dataclass(frozen=True)
class Task29ReleaseMaterialContract:
    schema: str = SCHEMA
    task_id: str = "29"
    target_body_path: str = "/World/Assets/PaperCup"
    material_path: str = "/World/PhysicsMaterials/Task29CupReleaseMat"
    static_friction: float = 1.2
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    friction_combine_mode: str = "max"
    compliant_contact: bool = False
    physics_scene_path: str = "/World/PhysicsScene"
    scene_ccd_required: bool = True
    target_ccd_enabled: bool = True
    landing_collision_policy: str = (
        "target_ccd_against_bottom_lod_before_first_physics_reset")
    binding_scope: str = (
        "target_enabled_collisions_and_active_hand_link4_collisions")
    activation_policy: str = "before_first_physics_reset"
    release_time_rebinding_forbidden: bool = True
    velocity_override_forbidden: bool = True

    def __post_init__(self) -> None:
        values = (
            self.static_friction,
            self.dynamic_friction,
            self.restitution,
        )
        if (
            self.schema != SCHEMA
            or self.task_id != "29"
            or self.target_body_path != "/World/Assets/PaperCup"
            or self.material_path
            != "/World/PhysicsMaterials/Task29CupReleaseMat"
            or not all(math.isfinite(value) for value in values)
            or self.static_friction != 1.2
            or self.dynamic_friction != 1.0
            or self.restitution != 0.0
            or self.friction_combine_mode != "max"
            or self.compliant_contact is not False
            or self.physics_scene_path != "/World/PhysicsScene"
            or self.scene_ccd_required is not True
            or self.target_ccd_enabled is not True
            or self.landing_collision_policy
            != "target_ccd_against_bottom_lod_before_first_physics_reset"
            or self.binding_scope
            != "target_enabled_collisions_and_active_hand_link4_collisions"
            or self.activation_policy != "before_first_physics_reset"
            or self.release_time_rebinding_forbidden is not True
            or self.velocity_override_forbidden is not True
        ):
            raise ValueError("invalid Task29 release-material contract")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

CONTRACT = Task29ReleaseMaterialContract()


__all__ = [
    "CONTRACT",
    "SCHEMA",
    "Task29ReleaseMaterialContract",
]
