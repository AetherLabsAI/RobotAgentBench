"""Version-pinned public person memory; no private target enumeration."""
import hashlib
import json
from pathlib import Path

MEMORY = Path(__file__).resolve().parent / "person_context_v4"
PINS = {
    "context.txt": "d5d4f52564612d1b7215a2b9dd6dcfb5a5a4466aae83ff93ca136887538ea5ca",
    "prompts.json": "d98df2f9fb041c2ab62da3f9cdd85707f1df63e16431a37958fe029e09c2da55",
}
DISCOVERY_SUFFIX = (
    "For this evaluation, do not carry out the requested household tasks. "
    "Instead, use the household memory, the request above and what you observe "
    "in the room to identify the objects that need to be handled. "
    "Touch each of those objects with either gripper. "
    "You do not need to grasp, lift, move or place them, operate appliances, "
    "or finish cleaning. Touching a destination or a piece of background "
    "furniture does not replace touching a task object."
)
PROTECTED_DISCOVERY_DIRECTIVE = "Touch every task object with either gripper; do not perform the tasks."


def wire_instruction(public):
    """Protect the complete request/directive before any native right truncation.

    Preserve the full fixed memory as the trailing, truncatable field. Never
    shorten the request or silently filter memory using the private target list.
    """
    suffix = "\n\n" + DISCOVERY_SUFFIX
    if not public['prompt'].endswith(suffix):
        raise ValueError('Pinned scenario request and discovery suffix required')
    request = public['prompt'][:-len(suffix)]
    protected = PROTECTED_DISCOVERY_DIRECTIVE + "\n" + request
    return protected + "\n\nHousehold context:\n" + public['context'] + suffix


def model_input(prompt_id):
    contents = {}
    for name, sha256 in PINS.items():
        raw = (MEMORY / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError("Person-aware input changed; explicit new revision required")
        contents[name] = raw.decode("utf-8")
    rows = [r for r in json.loads(contents["prompts.json"])["prompts"] if r["id"] == prompt_id]
    if len(rows) != 1:
        raise ValueError("One existing person-aware scenario prompt required")
    return dict(context=contents["context.txt"],
                prompt=rows[0]["prompt"] + "\n\n" + DISCOVERY_SUFFIX)
