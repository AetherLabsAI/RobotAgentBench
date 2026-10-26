"""Exact state required by the unmodified original native contact reader."""
from types import SimpleNamespace
from evaluation.tasks.microwave_take.initial_state import PAYLOAD, LEFTOVER_PAYLOAD


def reader_state(stage, interface, payload):
    if payload not in (PAYLOAD, LEFTOVER_PAYLOAD):
        raise ValueError("Exact selected original payload path required")
    return SimpleNamespace(stage=stage, interface=interface, payload=payload,
                           active={}, collider_paths={})
