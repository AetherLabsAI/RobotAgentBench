"""One task-local administrative entrypoint; no new robot action protocol."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def describe():
    return json.loads((HERE / 'definition.json').read_text())


def prepare(output, seed, role, variant='machine_only'):
    from .variants import get_variant
    get_variant(variant)
    if variant == 'bag_only':
        raise ValueError('Bag-only keeps its separate coffee_bag_interference preparation owner')
    if variant == 'bag_then_machine':
        from .build_combined import build
        return build(output, seed=seed, role=role)
    from .build_initial import build
    # Existing builder owns scope, source reset, randomization and physics audit.
    return build(output, seed=seed, role=role)


def preview(args):
    from .isolated_preflight import preflight
    # Reject wrong/source-incompatible initials BEFORE starting Isaac/GPU.
    preflight(args.initial)
    from .native_diagnostic import main as native_main
    command = ['--initial', str(args.initial), '--output', str(args.output),
               '--gpu', str(args.gpu), '--atomic-source-replay']
    if args.authorized_shared_gpu2:
        command.append('--authorized-shared-gpu2')
    if args.reuse_cache is not None:
        command.extend(['--reuse-cache', str(args.reuse_cache)])
    if args.hd:
        command.append('--hd-presentation')
    return native_main(command)


def verify(output, initial):
    for path in (output, initial):
        if not Path(path).resolve(strict=True).is_relative_to(HERE):
            raise ValueError('Verification inputs and report must remain inside this task')
    from .verify_atomic_replay import verify as native_verify
    result = native_verify(output, initial)
    # This command verifies the accepted INTERFERENCE, not benchmark success.
    # The original always-missing-pour stress result remains in the returned
    # evidence, even when false. No scoring rule or evidence is overwritten.
    return (0 if result['real_trigger_checks_passed']
            and all(result['mechanism_checks'].values())
            and all(result['farther_push_policy_checks'].values()) else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    from .variants import VARIANT_NAMES, describe_variant
    description = commands.add_parser('describe', help='Task definition and actual integration status')
    description.add_argument('--variant', choices=VARIANT_NAMES,
                             help='Describe a preset, not a registered runnable task')
    commands.add_parser('variants', help='List the three approved interference presets and status')
    setup = commands.add_parser('prepare', help='Existing atomic source/random initial sampler')
    setup.add_argument('--output', type=Path, required=True)
    setup.add_argument('--seed', type=int, required=True)
    setup.add_argument('--role', choices=('source', 'random'), required=True)
    setup.add_argument('--variant', choices=('machine_only', 'bag_then_machine'), default='machine_only',
                       help='Bag-only retains its existing separate task; combined requires paired/random initial')
    replay = commands.add_parser('preview', help='Unscored native source replay, not model evaluation')
    replay.add_argument('--initial', type=Path, required=True)
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--gpu', type=int, required=True)
    replay.add_argument('--authorized-shared-gpu2', action='store_true')
    replay.add_argument('--reuse-cache', type=Path)
    replay.add_argument('--hd', action='store_true')
    check = commands.add_parser('verify', help='Accepted interference evidence; not task score/success')
    check.add_argument('--initial', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    commands.add_parser('test', help='CPU rules/source/reader/presentation/entrypoint regressions')
    args = parser.parse_args(argv)
    if args.command == 'describe':
        print(json.dumps(describe_variant(args.variant) if args.variant else describe(),
                         indent=2, ensure_ascii=False))
        return 0
    if args.command == 'variants':
        print(json.dumps([describe_variant(name) for name in VARIANT_NAMES], indent=2))
        return 0
    if args.command == 'prepare':
        plan = prepare(args.output, args.seed, args.role, args.variant)
        print(json.dumps({k: plan[k] for k in ('scene', 'seed', 'role', 'source_api', 'source_episode')}, indent=2))
        return 0
    if args.command == 'preview':
        return preview(args)
    if args.command == 'verify':
        return verify(args.output, args.initial)
    import unittest
    modules = ('test_rules', 'test_usd', 'test_reader', 'test_replay_actions',
               'test_hd_presentation', 'test_entrypoint', 'test_variants', 'test_combined', 'test_native_bag',
               'test_gpu_admission', 'test_isolated_preflight')
    suite = unittest.defaultTestLoader.loadTestsFromNames([__package__ + '.' + name for name in modules])
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
