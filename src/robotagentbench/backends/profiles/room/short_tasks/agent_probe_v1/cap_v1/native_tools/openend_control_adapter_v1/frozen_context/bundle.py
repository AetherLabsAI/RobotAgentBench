"""Read frozen model-visible text; never infer or register tasks."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def model_input(prompt_id):
    if prompt_id == 'outing_generic_v3':
        # Explicit versioned task-owned input; legacy household texts stay frozen.
        path = HERE / 'outing_v3.json'
        public = json.loads(path.read_text(encoding='utf-8'))
        if set(public) != {'context', 'prompt'} or not all(isinstance(v, str) and v for v in public.values()):
            raise ValueError('Invalid frozen outing public input')
        return public
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if name not in ("context.txt", "prompts.json"):
            raise ValueError("Unexpected frozen input file")
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Frozen input changed: " + name)
    bank = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    matches = [row for row in bank["prompts"] if row["id"] == prompt_id]
    if len(matches) != 1:
        raise ValueError("Expected one known prompt ID: " + prompt_id)
    return {
        "context": (HERE / "context.txt").read_text(encoding="utf-8"),
        "prompt": matches[0]["prompt"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_id")
    args = parser.parse_args()
    print(json.dumps(model_input(args.prompt_id), ensure_ascii=False, indent=2))
