"""Versioned wire models; never infer joint order or silently convert units."""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, field_validator, model_serializer

from . import robot_profiles as rp

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
StepIndex = Annotated[int, Field(strict=True, ge=0)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Randomization(WireModel):
    appearance: StrictBool = False
    layout: StrictBool = False
    microwave_pose: StrictBool = False
    left_arm_pose: StrictBool = False


RobotProfileId = Literal[rp.JOINT16, rp.JOINT20, rp.MOBILE22]


class EpisodeConfig(WireModel):
    robot_profile: RobotProfileId = rp.DEFAULT
    task_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")]
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)] = 0
    max_steps: Annotated[int, Field(strict=True, ge=1, le=60000)] | None = None
    randomization: Randomization = Field(default_factory=Randomization)
    instruction: Annotated[str, Field(max_length=1000)] = ""
    task_parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)


class BaseVelocity(WireModel):
    base_forward_m_s: Number
    base_yaw_rate_rad_s: Number


class BaseState(WireModel):
    position_world_m: Annotated[list[Number], Field(min_length=3, max_length=3)]
    orientation_world_xyzw: Annotated[list[Number], Field(min_length=4, max_length=4)]
    yaw_world_unwrapped_rad: Number
    linear_velocity_world_m_s: Annotated[list[Number], Field(min_length=3, max_length=3)]
    angular_velocity_world_rad_s: Annotated[list[Number], Field(min_length=3, max_length=3)]


class JointPositionAction(WireModel):
    mode: Literal["joint_position"] = "joint_position"
    unit: Literal["rad", "native"] = "rad"
    targets: Annotated[dict[str, Number], Field(min_length=1, max_length=128)]
    base_velocity: BaseVelocity | None = None

    @model_serializer(mode='wrap')
    def compatible_wire(self, handler):
        result = handler(self)
        if self.base_velocity is None:
            result.pop('base_velocity', None)
        return result


class StepRequest(WireModel):
    request_id: UUID
    expected_step_index: StepIndex
    action: JointPositionAction
    physics_hz: Annotated[int, Field(strict=True)] | None = None

    @field_validator('physics_hz')
    @classmethod
    def supported_physics_rate(cls, value):
        from .physics_rate import validate_hz
        return None if value is None else validate_hz(value)


class Observation(WireModel):
    episode_id: UUID
    step_index: StepIndex
    time_s: Number
    time_basis: Literal["simulation_time"] = "simulation_time"
    joint_positions_rad: dict[str, Number]
    joint_velocities_rad_s: dict[str, Number] = Field(default_factory=dict)
    joint_positions_m: dict[str, Number] = Field(default_factory=dict)
    joint_velocities_m_s: dict[str, Number] = Field(default_factory=dict)
    images: dict[str, str] = Field(default_factory=dict)
    instruction: str = ""
    source: Literal["isaac_physics"] = "isaac_physics"
    base_state: BaseState | None = None

    @model_serializer(mode='wrap')
    def compatible_wire(self, handler):
        result = handler(self)
        if self.base_state is None:
            result.pop('base_state', None)
        return result


class TaskEvaluation(WireModel):
    """Privileged server result, NEVER part of the model observation."""
    task_id: str
    episode_id: UUID
    step_index: StepIndex
    time_s: Number
    evaluator_id: str
    valid: StrictBool
    score: Number | None = None
    maximum_score: Number | None = None
    success: StrictBool | None = None
    terminated: StrictBool = False
    truncated: StrictBool = False
    reason: str | None = None
    finalized: StrictBool = False
    details: dict = Field(default_factory=dict)


class EpisodeState(WireModel):
    robot_profile: RobotProfileId = rp.JOINT16
    episode_id: UUID
    task_id: str
    backend: Literal["isaac_physics"] = "isaac_physics"
    evaluation_valid: StrictBool = False
    seed: int
    task_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    status: Literal["running", "paused", "terminated", "truncated", "closed", "error"]
    terminated: bool = False
    truncated: bool = False
    stop_reason: str | None = None
    observation: Observation
    # Controller evidence is not sensor input and is never sent to the policy.
    execution: dict | None = None
    randomization_applied: Randomization = Field(default_factory=Randomization)
    evaluation: TaskEvaluation | None = None
    reward: Number | None = None
    success: StrictBool | None = None


class EpisodeResult(WireModel):
    robot_profile: RobotProfileId = rp.JOINT16
    episode_id: UUID
    task_id: str
    backend: Literal["isaac_physics"] = "isaac_physics"
    status: str
    steps: int
    seed: int
    task_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    evaluation_valid: StrictBool = False
    scoring_status: Literal["not_evaluated", "scoring", "finalized", "invalid"] = "not_evaluated"
    success: StrictBool | None = None
    score: Number | None = None
    maximum_score: Number | None = None
    termination_reason: str | None = None
    evaluation: TaskEvaluation | None = None
    reason: str = "Physical execution only; task reset/scoring unavailable."
    metrics: dict = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class ErrorDetail(WireModel):
    code: str
    message: str


class ErrorResponse(WireModel):
    error: ErrorDetail
