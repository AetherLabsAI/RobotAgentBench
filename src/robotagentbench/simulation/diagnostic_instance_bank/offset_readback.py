"""Trusted, read-only native offset observation at original initial admission.

This supplies a TaskBackend factory wrapper, not a simulator or an action
source. It neither changes original modules nor creates/enables native shapes.
Use it only in a new operator-owned diagnostic with unchanged original inputs.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from .contact_offsets import EffectiveOffsetUnavailable, capture_existing_human_views

OWNER = {'owner': 'RobotAgentBench-diagnostic-offset-readback-v32'}
TASK = 'stock_wooden_tray_interference'


def _output(root):
    root = Path(root)
    if (not root.is_absolute() or root.resolve() != root
            or json.loads((root/'OWNER.json').read_text()) != OWNER):
        raise ValueError('Canonical fresh readback-owned output required')
    path = root/'effective_contact_offsets.json'
    if path.exists() or path.is_symlink():
        raise ValueError('Offset observation must not overwrite an old result')
    return path


class _InitialReadbackSession:
    def __init__(self, original, output, sdk_identity, source_identity):
        self._original = original
        self._output = output
        self._sdk_identity = sdk_identity
        self._source_identity = source_identity
        self._observed = False

    def __getattr__(self, name):
        return getattr(self._original, name)

    def accept_initial(self, *args, **kwargs):
        # Original acceptance, scorer state and arm operation execute first,
        # unchanged. A failed original admission is never swallowed.
        result = self._original.accept_initial(*args, **kwargs)
        if self._observed:
            raise RuntimeError('Readback admission is one-shot')
        self._observed = True
        session = self._original
        report = dict(status='effective_offset_unavailable',
                      original_initial_accepted=result.get('accepted') is True,
                      control_actions_sent=0, trigger_forced=False,
                      physics_or_scoring_modified=False)
        before = now = None
        try:
            if (result.get('accepted') is not True or not session.started
                    or session.closed or session.faulted or session.control_index != 0
                    or session.pending is not None):
                raise EffectiveOffsetUnavailable('Fresh original accepted initial state required')
            actor = session.environment_actor
            backend = actor.backend
            human = backend.human
            if (backend.phase != 'waiting' or actor.started_at is not None
                    or human.appearance_state != 'waiting'):
                raise EffectiveOffsetUnavailable('Existing actor must remain absent and untriggered')
            clock = session.clock
            now, before = clock.current_time, clock.snapshot()
            stage = backend.stage
            scene = stage.GetPrimAtPath('/World/PhysicsScene')
            def attribute(name):
                value = scene.GetAttribute(name).Get()
                return list(map(float, value)) if name.endswith('gravityDirection') and value is not None else value
            physics = dict(meters_per_unit=stage.GetMetadata('metersPerUnit'),
                           physics_hz=clock.hz,
                           gravity_magnitude=attribute('physics:gravityMagnitude'),
                           gravity_direction=attribute('physics:gravityDirection'),
                           gpu_dynamics=attribute('physxScene:enableGPUDynamics'),
                           step_counter_source='original_pending_control_interval_substeps_not_global_native_index')
            shape = dict(self._source_identity)
            shape['measured_asset_sha256'] = hashlib.sha256(backend.asset.path.read_bytes()).hexdigest()
            if shape.get('expected_asset_sha256') != shape['measured_asset_sha256']:
                raise EffectiveOffsetUnavailable('Existing native human asset differs from sealed source')
            report.update(capture_existing_human_views(
                views=human.views, links=human.links,
                physics_step=session.substeps, time_s=now,
                sdk_identity=self._sdk_identity, shape_identity=shape,
                scene_physics_identity=physics))
            report['status'] = 'measured'
        except Exception as error:
            # Unknown native offsets remain unknown. This diagnostic does not
            # alter the owner's result, enable collision, or rerun admission.
            report.update(status='effective_offset_unavailable',
                          error_type=type(error).__name__, error=str(error))
        finally:
            if before is not None:
                unchanged = (session.clock.current_time == now
                             and session.clock.snapshot() == before)
                report['native_state_and_clock_unchanged'] = unchanged
                if not unchanged:
                    report['status'] = 'invalid_readback_state_changed'
            with self._output.open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')
        return result


def wrap_backend(backend, *, output_root, sdk_identity, source_identity):
    """Copy the existing trusted TaskBackend with a read-only session proxy.

    All other callbacks are the original bound callables. Use the frozen
    v20 worker/source tree for the existing seed-8 input, not current generator
    code substituted under an old launch. This function starts nothing.
    """
    path = _output(output_root)
    if (not isinstance(sdk_identity, dict) or not sdk_identity
            or not isinstance(source_identity, dict)
            or not source_identity.get('expected_asset_sha256')):
        raise ValueError('Explicit sealed SDK and source identities required')
    original_factory = backend.build_short_session
    def factory(task, *args, **kwargs):
        if task != TASK:
            raise ValueError('Offset readback is restricted to the original shelf relocation owner')
        session = original_factory(task, *args, **kwargs)
        return _InitialReadbackSession(session, path, dict(sdk_identity), dict(source_identity))
    return replace(backend, build_short_session=factory)


def measured_uniform_arm_margin(receipt, *, expected_sdk, expected_shape):
    """Validate an operator-verified real receipt; never estimate a margin.

    The caller must independently verify original native completion, source
    pins and the receipt hash. This helper cannot turn a CPU fixture into native
    evidence. Unequal per-bone offsets are not reduced with max/mean/fallback.
    """
    if (receipt.get('status') != 'measured'
            or receipt.get('native_state_and_clock_unchanged') is not True
            or receipt.get('original_initial_accepted') is not True
            or receipt.get('sdk_identity') != expected_sdk
            or receipt.get('shape_identity') != expected_shape
            or receipt.get('physics_attributes_written') is not False
            or receipt.get('shapes_created_or_enabled') is not False):
        raise EffectiveOffsetUnavailable('Verified unchanged original native receipt required')
    physics = receipt['scene_physics_identity']
    if physics.get('meters_per_unit') != 1 or physics.get('physics_hz') != 60:
        raise EffectiveOffsetUnavailable('Original metre/60Hz scene identity required')
    rows = receipt['rows']
    arm = [rows[name]['contact_offset_stage_units'] for name in ('forearm.R', 'upper_arm.R')]
    # Python float retains all bits of the SDK float32 input. Exact hexadecimal
    # equality preserves that value; no tolerance, averaging or rounding.
    if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in arm) or arm[0].hex() != arm[1].hex():
        raise EffectiveOffsetUnavailable('Actual forearm and upper-arm offsets are not bit-identical')
    return arm[0]
