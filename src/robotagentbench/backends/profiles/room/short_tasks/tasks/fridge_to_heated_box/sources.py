"""Reject component drift before authoring a task 07 scene or native reader."""
import hashlib
import json
from pathlib import Path


def verify_sources(receipt=None):
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    if receipt is None:
        receipt = json.loads((here / "component_sources.json").read_text())
    changed = []
    for relative, expected in receipt["files_sha256"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    if changed:
        raise ValueError("Task 07 pinned component sources changed: " + ", ".join(changed))
    return dict(verified=True, component_count=len(receipt["files_sha256"]),
                upstream_commit=receipt["upstream_commit"], native_physics_verified=False)
