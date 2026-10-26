"""Provider-side codec for the H2 G1 checkpoints, NOT a universal GO-1 convention.

Arena n = outer_joint_rad / (pi/4): 0 closed, 1 open.
This checkpoint m = -closure_command: -1 closed, 0 open.
Thus model input m=n-1 and Arena output n=m+1 (not simply -m).
Neither conversion clips, replans, nor changes the other fourteen coordinates.

Training used commanded effector values in its state slots. Measured joint
opening is an explicit approximation, especially under contact; this codec
does not establish task-level or physical-aperture calibration.
"""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .policy_contract import validate_chunk

ENCODING = "g1_negative_closure_v1"
GRIPPERS = (7, 15)


def to_native_state(state):
    if not isinstance(state, list) or len(state) != 1:
        raise ValueError("Expected one measured state row")
    validate_chunk([state[0]] * 30)
    result = [list(state[0])]
    for index in GRIPPERS:
        if not 0 <= result[0][index] <= 1:
            raise ValueError("Arena gripper state outside [0,1]")
        result[0][index] -= 1.
    return result


def to_arena_actions(native):
    validate_chunk(native)
    result = [list(row) for row in native]
    for row in result:
        for index in GRIPPERS:
            row[index] += 1.
    # Keep finite out-of-range predictions intact. The evaluator's unchanged
    # strict limits reject them; silently clipping would change model behavior.
    return result


class NegativeClosurePolicy:
    """Wrap a provider-local runtime; keep native and converted evidence locally."""
    def __init__(self, native, audit_dir):
        self.native = native
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = {**native.metadata, "gripper_codec": ENCODING,
                         "gripper_input": "arena_n_minus_1", "gripper_output": "native_m_plus_1",
                         "state_gripper_source": "measured_joint_opening_approximation_of_training_command",
                         "contact_state_equivalence_verified": False,
                         "task_evaluation_valid": False}
        self.run_id = None
        self.index = 0
        self._save("provider.json", self.metadata)

    def _save(self, name, record):
        with (self.audit_dir / name).open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def reset(self, seed):
        self.native.reset(seed)
        self.run_id, self.index = str(uuid4()), 0
        self._save(f"{self.run_id}_reset.json", {"model_rng_seed": seed, "codec": ENCODING})

    def infer(self, instruction, state, images):
        if self.run_id is None:
            raise ValueError("Reset required before inference")
        native_state = to_native_state(state)
        prefix = f"{self.run_id}_{self.index:04d}"
        self._save(prefix + "_input.json", {
            "instruction": instruction, "arena_state": state, "native_state": native_state,
            "rgb_array_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest()
                                 for key, value in images.items()}, "codec": ENCODING})
        native_actions = self.native.infer(instruction, native_state, images)
        # Preserve native output before adapting or checking execution limits.
        self._save(prefix + "_native.json", {"native_action_chunk": native_actions})
        actions = to_arena_actions(native_actions)
        self._save(prefix + "_arena.json", {"action_chunk": actions, "clipping_applied": False})
        self.index += 1
        return actions

    def close(self):
        self.native.close()
