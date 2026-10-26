"""Standalone physical identities; no meal destination or predecessor."""
from dataclasses import dataclass
from evaluation.tasks.microwave_take.initial_state import (
    MICROWAVE, PLATE, STAND, ROBOT, bindings_for,
)

KNOB = MICROWAVE+"/Asset/StartKnob"
KIND = "microwave_heat_take"
INSTANCE = "heat_take"


@dataclass(frozen=True)
class HeatBinding:
    episode_id: str
    instance_id: str
    payload_variant: str

    def __post_init__(self):
        if not self.episode_id or self.instance_id != INSTANCE:
            raise ValueError("Exact standalone microwave identity required")
        bindings_for(self.payload_variant)


def validate_parameters(parameters):
    if (not isinstance(parameters, dict) or set(parameters) != {"payload_variant"}
            or parameters["payload_variant"] not in ("beef_box", "leftover_box")):
        raise ValueError("One original payload variant required")


def bind_manifest(parameters, manifest):
    validate_parameters(parameters)
    tasks = manifest.get("tasks", [])
    if (len(tasks) != 1 or tasks[0].get("id") != INSTANCE or tasks[0].get("type") != KIND
            or tasks[0].get("parameters") != parameters or tasks[0].get("target_path") != KNOB):
        raise ValueError("One exact independent microwave scene task required")
    paths = [r.get("path") for r in manifest.get("objects", [])]
    required = (MICROWAVE, PLATE, STAND, bindings_for(parameters["payload_variant"])["payload"])
    if any(paths.count(p) != 1 for p in required):
        raise ValueError("Unique original microwave, stand, plate and food box required")
    return tasks[0]
