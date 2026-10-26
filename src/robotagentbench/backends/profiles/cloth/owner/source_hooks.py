"""Explicit task-local extension of pinned original source, not an SDK patch.

All original lines remain byte-for-byte present. The added branch hands the
already initialized native objects to the trusted owner, and exposes exactly
the original apply_action + step(render=False) statements. The diagnostic
replay/repair loop remains available when no owner is installed.
"""
import ast
import hashlib


SOURCE_SHA256 = {
    'native_removal_continuous_transport_v54.py':
        '6071a1f888b6c721d7d5a60b6da07fdeebcd020a4e2ef869e5e62dcc525297ce',
    'native_measured_mirrored_lay_v50.py':
        'af277146a6797013e0677251fe656e8db1e9fb14adb528580cd828c861284428',
}
MARKER = '    adaptive_brush = None\n'
PREINIT_MARKER = '    sim = SimulationContext(physics_dt=1/(fps*args.substeps), rendering_dt=1/fps,\n'
PREINIT_HOOK = '''    # Native cameras must be registered before this source parses physics.
    observation_owner = globals().get('NATIVE_OWNER')
    if observation_owner is not None:
        observation_owner.prepare_observations(stage, app)
'''
NATIVE_STEP = '''                art.apply_action(ArticulationAction(
                    joint_positions=torch.as_tensor(targets, dtype=torch.float32, device='cuda:0'),
                    joint_indices=torch.as_tensor(indices, dtype=torch.int64, device='cuda:0')))
                sim.step(render=False)
'''
HOOK = '''    # Task-local native owner: no second World, reset, renderer or controller.
    native_owner = globals().get('NATIVE_OWNER')
    if native_owner is not None:
        def advance_original_substep(targets):
            art.apply_action(ArticulationAction(
                joint_positions=torch.as_tensor(targets, dtype=torch.float32, device='cuda:0'),
                joint_indices=torch.as_tensor(indices, dtype=torch.int64, device='cuda:0')))
            sim.step(render=False)
        try:
            return native_owner.run(locals(), advance_original_substep)
        finally:
            try:
                native_owner.close()
            finally:
                try:
                    sim.stop()
                except BaseException as exc:
                    native_owner.native_stop_failed(exc)
                    raise
                else:
                    native_owner.native_stopped()
'''


def extend_source(raw, filename, *, observations=False):
    """Return a separately saved extension; never overwrite the pinned source."""
    if filename not in SOURCE_SHA256:
        raise ValueError('Unknown native entry')
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[filename]:
        raise ValueError('Original entry bytes differ from the audited source')
    original = raw.decode('utf-8')
    if original.count(MARKER) != 1 or original.count(NATIVE_STEP) != 1:
        raise ValueError('Native hook or physical-step identity changed')
    if type(observations) is not bool:
        raise ValueError('Explicit trusted observation mode required')
    extended = original.replace(MARKER, HOOK + MARKER, 1)
    if observations:
        if original.count(PREINIT_MARKER) != 1:
            raise ValueError('Source physics initialization boundary changed')
        extended = extended.replace(PREINIT_MARKER, PREINIT_HOOK+PREINIT_MARKER, 1)
    # Reversibility is an explicit protection against hidden parameter edits.
    if extended.replace(HOOK, '', 1).replace(PREINIT_HOOK, '', 1) != original:
        raise AssertionError('Unexpected source edit outside the named hook')
    tree = ast.parse(extended)
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == 'advance_original_substep')
    expected = ast.parse('def step(targets):\n' + '\n'.join(
        line[12:] for line in NATIVE_STEP.rstrip().splitlines()))
    if ast.dump(ast.Module(body=function.body, type_ignores=[])) != ast.dump(
            ast.Module(body=expected.body[0].body, type_ignores=[])):
        raise AssertionError('Owner step differs from the original native statements')
    return extended
