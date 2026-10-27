"""Inspect saved diagnostics; this command never generates or runs a scene."""
import argparse
import json
from . import load, model_input


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario')
    parser.add_argument('--model-input', action='store_true')
    args = parser.parse_args(argv)
    function = model_input if args.model_input else load
    print(json.dumps(function(args.bank, args.task, args.seed, scenario=args.scenario), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
