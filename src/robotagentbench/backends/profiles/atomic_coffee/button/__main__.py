import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Task-local atomic button lighting variant')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('describe')
    sub.add_parser('test')
    build = sub.add_parser('prepare')
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--seed', type=int, required=True)
    build.add_argument('--role', choices=('source', 'random'), required=True)
    args = parser.parse_args()
    if args.command == 'describe':
        print((Path(__file__).parent/'definition.json').read_text())
    elif args.command == 'prepare':
        from .initial import build
        print(json.dumps(build(args.output, seed=args.seed, role=args.role), indent=2))
    else:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_lighting')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
