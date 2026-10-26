"""Existing pinned atomic-coffee environment; no alternative scoring or physics."""
import argparse
import json
import sys
from pathlib import Path
from ..common import COFFEE_SOURCE


def main():
    from evaluation.tasks.coffee_machine import initial_state
    if not Path(initial_state.__file__).resolve().is_relative_to(COFFEE_SOURCE.resolve()):
        raise ValueError("Atomic coffee dependency must load from its original pinned package")
    if sys.argv[1] == "native":
        from ..closeout_v1.public_adapter import exec_native
        del sys.argv[1]
        return exec_native()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare",))
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(args.request_json)
    if request["task_name"] != "failure_recovery_coffee_bag":
        raise ValueError("Coffee dependency worker accepts only its trusted task")
    from ..closeout_v1.public_adapter import prepare_registered
    prepare_registered(request, None, None, args.output)


if __name__ == "__main__":
    main()
