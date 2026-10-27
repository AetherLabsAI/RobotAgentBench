"""Catalog queries, dependency checks and explicit trusted native operators."""
import argparse
import json
import sys

from . import __version__
from .tasks.catalog import CATALOG_NAMES, CatalogError, load_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotagentbench",
        description="RobotAgentBench household-agent evaluation tooling.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    instances = commands.add_parser('instances', help='Inspect or select a saved task instance without randomizing or running it')
    instances.add_argument('operation', choices=('list', 'verify', 'select'))
    instances.add_argument('plan', help='Installed, hash-pinned saved-instance source plan')
    instances.add_argument('--task', help='Task identifier from instances list')
    instances.add_argument('--episode', type=int, help='Saved episode ordinal 1–30; select only')
    catalog = commands.add_parser("catalog", help="List registered task or component identifiers")
    catalog.add_argument("name", choices=CATALOG_NAMES)
    catalog.add_argument("--format", choices=("text", "json"), default="text")
    dependencies = commands.add_parser("check-dependencies", help="Verify files against an operator-reviewed dependency configuration")
    dependencies.add_argument("config", help="Runtime dependency JSON configuration")
    dependencies.add_argument("--format", choices=("text", "json"), default="text")
    backends = commands.add_parser('backends', help='Inspect or verify bundled task implementations without Isaac Sim')
    backends.add_argument('--profile', choices=('room', 'atomic_coffee', 'failure_recovery', 'cloth'))
    backend_mode = backends.add_mutually_exclusive_group()
    backend_mode.add_argument('--verify', action='store_true')
    backend_mode.add_argument('--root', action='store_true', help='Print the embedded source directory')
    assets = commands.add_parser('assets', help='Download, install or verify a pinned asset release')
    assets.add_argument('operation', choices=('fetch', 'install', 'verify', 'configure'))
    assets.add_argument('manifest', help='Versioned asset release manifest supplied with the code release')
    assets.add_argument('--cache', help='Archive cache; required for fetch/install')
    assets.add_argument('--destination', help='Asset directory; required for install/verify')
    assets.add_argument('--runtime', help='Fresh runtime directory; only with configure')
    assets.add_argument('--profile', choices=('room', 'atomic_coffee', 'failure_recovery', 'cloth'), help='Task backend; only with configure')
    cloth = commands.add_parser("cloth", help="Inspect or explicitly run a configured native cloth task")
    cloth.add_argument("operator", help="Reviewed native cloth operator JSON")
    cloth.add_argument("--task", required=True, choices=("spread_tablecloth", "collect_tablecloth", "long_task_18"))
    mode = cloth.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--prepare", metavar="FRESH_LAUNCH")
    mode.add_argument("--execute", metavar="PREPARED_LAUNCH")
    cloth.add_argument("--output", help="Fresh output path; only with explicit --execute")
    specialized = commands.add_parser("specialized", help="Inspect or explicitly run a configured recovery or atomic-coffee owner")
    specialized.add_argument("operator", help="Reviewed specialized native operator JSON")
    specialized.add_argument("--family", required=True, choices=("atomic_coffee", "failure_recovery"))
    specialized.add_argument("--task", required=True)
    mode = specialized.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--prepare", metavar="FRESH_DISPATCH_DESCRIPTOR")
    mode.add_argument("--execute", action="store_true")
    for name in ("launch", "output", "dispatch"):
        specialized.add_argument("--" + name, help="Explicit execute-only " + name)
    return parser


def _unique_config_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate dependency configuration field')
        value[key] = item
    return value


def _check_dependencies(config):
    # Operator CLI only: never import a task implementation or native runtime.
    from .simulation.dependencies import RuntimeDependencies
    with open(config, encoding='utf-8') as stream:
        value = json.load(stream, object_pairs_hook=_unique_config_keys)
    if not isinstance(value, dict) or not isinstance(value.get('resources'), dict):
        raise ValueError('Explicit resource mapping required')
    return RuntimeDependencies.from_config(value).verify()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'instances':
        from .simulation import saved_instances
        if args.operation == 'list' and (args.task is not None or args.episode is not None):
            parser.error('instances list does not accept --task/--episode')
        if args.operation == 'verify' and args.episode is not None:
            parser.error('instances verify checks a task bank; --episode is select-only')
        if args.operation == 'select' and (args.task is None or args.episode is None):
            parser.error('instances select requires --task and --episode')
        try:
            result = (saved_instances.list_tasks(args.plan) if args.operation == 'list' else
                      saved_instances.verify(args.plan, args.task) if args.operation == 'verify' else
                      saved_instances.select(args.plan, args.task, args.episode))
        except (OSError, ValueError, KeyError, TypeError):
            print(f'{parser.prog}: saved-instance identity or artifact verification failed; no task was started', file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if args.command == 'backends':
        from .backends import describe, root, verify
        if (args.verify or args.root) and args.profile is None:
            parser.error('--verify/--root requires --profile')
        try:
            result = (str(root(args.profile)) if args.root else verify(args.profile)
                      if args.verify else describe())
        except (OSError, ValueError, KeyError, TypeError):
            print(f'{parser.prog}: embedded backend source failed verification', file=sys.stderr)
            return 2
        print(result if args.root else json.dumps(result, indent=2))
        return 0
    if args.command == 'assets':
        from . import resources
        if args.operation in ('fetch', 'install') and not args.cache:
            parser.error('asset fetch/install requires --cache')
        if args.operation in ('install', 'verify', 'configure') and not args.destination:
            parser.error('asset install/verify/configure requires --destination')
        if args.operation == 'fetch' and args.destination:
            parser.error('asset fetch does not accept --destination')
        if args.operation in ('verify', 'configure') and args.cache:
            parser.error('asset verify/configure does not accept --cache')
        if args.operation == 'configure' and (not args.runtime or not args.profile):
            parser.error('asset configure requires --runtime and --profile')
        if args.operation != 'configure' and (args.runtime or args.profile):
            parser.error('--runtime/--profile are configure-only arguments')
        try:
            manifest = resources.read_manifest(args.manifest)
            if args.operation == 'fetch':
                result = resources.fetch(manifest, args.cache)
            elif args.operation == 'install':
                result = resources.install(manifest, args.cache, args.destination)
            elif args.operation == 'verify':
                result = resources.verify(manifest, args.destination)
            else:
                from .installation import configure
                result = configure(manifest, args.destination, args.runtime, args.profile)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            # Error type is useful without leaking private paths or file content.
            print(f'{parser.prog}: asset {args.operation} failed ({type(error).__name__}); no native task was started', file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0
    if args.command == 'specialized':
        from .simulation.specialized.operator import main as specialized_main
        from .simulation.source_tree import SourceDependencyError
        forwarded = [args.operator, '--family', args.family, '--task', args.task]
        for flag in ('verify', 'execute'):
            if getattr(args, flag):
                forwarded.append('--' + flag)
        for flag in ('prepare', 'launch', 'output', 'dispatch'):
            if getattr(args, flag) is not None:
                forwarded.extend(('--' + flag, getattr(args, flag)))
        try:
            return specialized_main(forwarded)
        except (OSError, ValueError, TypeError, KeyError, SourceDependencyError):
            print(f'{parser.prog}: specialized configuration or a required native resource failed verification', file=sys.stderr)
            return 2
    if args.command == 'cloth':
        from .simulation.cloth.launcher import main as cloth_main
        from .simulation.source_tree import SourceDependencyError
        forwarded = [args.operator, '--task', args.task]
        if args.verify:
            forwarded.append('--verify')
        for flag in ('prepare', 'execute', 'output'):
            if getattr(args, flag) is not None:
                forwarded.extend(('--' + flag, getattr(args, flag)))
        try:
            return cloth_main(forwarded)
        except (OSError, ValueError, TypeError, SourceDependencyError):
            print(f'{parser.prog}: cloth configuration or a required native resource failed verification', file=sys.stderr)
            return 2
    if args.command == 'check-dependencies':
        from .simulation.source_tree import SourceDependencyError
        try:
            result = _check_dependencies(args.config)
        except (OSError, ValueError, TypeError, SourceDependencyError):
            print(f'{parser.prog}: dependency configuration is invalid or a required file failed verification', file=sys.stderr)
            return 2
        if args.format == 'json':
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print(f"Verified {result['file_count']} configured resources")
        return 0
    try:
        catalog = load_catalog(args.name)
    except (CatalogError, OSError) as exc:
        # No traceback, input contents, or local installation paths in CLI errors.
        print(f"{parser.prog}: {type(exc).__name__}: packaged catalog is unavailable or invalid", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(catalog.as_dict(), indent=2, allow_nan=False))
    else:
        print(f"{catalog.name}: {len(catalog.identifiers)} registered {catalog.entry_kind}")
        print("\n".join(catalog.identifiers))
    return 0
