"""Long identity/completion around the unchanged native OpenEnd control owner.

No simulation step, World/reset, action producer, or scoring arithmetic here.
Original per-port numeric scores and dispatcher termination remain authoritative.
"""
from copy import deepcopy
import math

from .long_registry import compile_task, bind_compiled


def prepare_scene(compiled, stage, manifest, output):
    """Execute only existing selected initialization hooks before PhysX attach."""
    from importlib import import_module
    from pxr import Usd
    session = stage.GetSessionLayer()
    if not session.anonymous:
        raise ValueError('Anonymous pre-physics session required')
    before = session.ExportToString()
    result = deepcopy(manifest)
    try:
        with Usd.EditContext(stage, session):
            for plan in compiled['scene_preparations']:
                # This list comes from compile_task, never a public module string.
                allowed = {'short_tasks.tasks.sofa_to_coffee_table.scene.prepare',
                           'short_tasks.tasks.sofa_to_drawer.scene.prepare'}
                path = plan['scene_preparation']
                if path not in allowed:
                    raise ValueError('Unknown Long scene preparation')
                module, function = path.rsplit('.', 1)
                result = getattr(import_module(module), function)(plan, stage, result)
        if compiled['document_id'] == '20':
            from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.scenario_alignment_v1.runtime import prepare_fridge
            result = prepare_fridge(stage, result, compiled['runtime_request'], output)
    except BaseException:
        session.ImportFromString(before)
        raise
    return result


def native_factories(compiled):
    factories = {}
    for identity, role in compiled['native_factory_roles'].items():
        if role == 'approved_sofa_book_surface':
            from .book_surface import factory_override
            declaration = next(t for t in compiled['runtime_request']['tasks'] if t['instance_id'] == identity)
            plan = dict(asset='book_red', runtime_components=[declaration])
            factories[identity] = factory_override(plan, declaration)
        elif role == 'approved_wallet_drawer':
            from .wallet_lifecycle import wallet_port
            factories[identity] = wallet_port()
        elif role == 'approved_coffee_preparation':
            from .current_devices import coffee_port
            factories[identity] = coffee_port()
        elif role == 'approved_bread_manipulation_current':
            from .current_devices import bread_port
            factories[identity] = bread_port()
        else:
            raise ValueError('Unknown trusted Long native factory role')
    from .current_terminal import wrap_factories
    return wrap_factories(compiled, factories)


def _current_component(port, *, now, kind, settled):
    """Read existing evaluator predicates; never infer success from point totals."""
    score = port.score
    if not isinstance(score, dict) or type(score.get('success')) is not bool:
        return dict(available=False, met=None, reason='typed_current_native_predicate_missing')
    if settled and hasattr(port, '_long_terminal_current'):
        current = port._long_terminal_current
        if current.get('simulation_time_s') != now:
            return dict(available=False, met=None, reason='terminal_current_sample_not_current')
        return deepcopy(current)
    component = port.composite
    if kind == 'coffee':
        current = getattr(component, 'current_result', None)
        if not component.component.success:
            return dict(available=True, met=False, source='required_coffee_process_incomplete')
        observed = port.owner.service.latest['sample'].time_s
        if observed != now or getattr(port,'_long_current_observed_time',None) != now or not isinstance(current, dict) or type(current.get('valid')) is not bool:
            return dict(available=False, met=None, reason='current_coffee_predicate_missing_or_stale')
        return dict(available=True, met=current['valid'], source='approved_prepare_coffee_current_result',
                    simulation_time_s=now, original_predicate=deepcopy(current))
    if kind == 'fryer':
        if not component.success:
            return dict(available=True, met=False, source='required_fryer_process_incomplete')
        current = getattr(port, '_long_current_result', None)
        if not isinstance(current, dict) or current.get('simulation_time_s') != now:
            return dict(available=False, met=None, reason='current_fryer_predicate_missing_or_stale')
        return deepcopy(current)
    if kind in ('microwave_box_to_meal_tray', 'cutlery_to_meal_tray'):
        # These native finalize methods sample present transforms/contact state
        # at shared settlement. Their synthetic component-step clock is not the
        # global World clock and must not be mistaken for a stale sample.
        final = score.get('final_state')
        if settled and isinstance(final, dict) and type(final.get('condition_met')) is bool:
            return dict(available=True, met=bool(score['success'] and final['condition_met']),
                source='original_native_terminal_resampling', simulation_time_s=now,
                original_predicate=deepcopy(final))
        return dict(available=False, met=None, reason='native_terminal_resampling_required')
    last = getattr(component, 'last', None)
    observed = getattr(last, 'time_s', None)
    if type(observed) not in (float, int) or not math.isfinite(observed) or abs(observed-now) > 1e-6:
        # An untouched initial component cannot have completed its process.
        # Preserve a definite non-success without pretending its sample is new.
        if score['success'] is False and score.get('operation_complete') is False:
            return dict(available=True, met=False, source='required_operation_incomplete',
                        simulation_time_s=observed, current_placement_sample=False)
        return dict(available=False, met=None, reason='component_final_sample_not_current',
                    sample_time_s=observed, simulation_time_s=now)
    return dict(available=True, met=score['success'], source='original_current_component_success',
                simulation_time_s=observed)


def completion_evidence(room):
    """Conjunction specified by the user, separate from existing native scores."""
    spec = room._long_compiled
    now = room.clock.current_time
    declarations = {t['instance_id']:t for t in spec['runtime_request']['tasks']}
    rows = {}
    for identity, port in room.ports.items():
        declaration = declarations.get(identity)
        if declaration is None:
            raise ValueError('Unbound Long runtime instance')
        kind = declaration['type']
        if kind == 'fridge_to_microwave' and spec['document_id'] == '20':
            # Retrieval/loading is a causal prerequisite, not the final location:
            # after extraction it must not require the meal still be in the oven.
            rows[identity] = dict(available=True, met=bool(port.composite.success),
                source='original_fridge_handoff_causal_prerequisite')
        else:
            rows[identity] = _current_component(port, now=now, kind=kind,
                settled=room.dispatch.ended and hasattr(room,'terminal_result'))
    if set(rows) != set(declarations):
        raise ValueError('Missing Long component evidence')
    available = all(row['available'] for row in rows.values())
    return dict(schema='long_task.completion_evidence.v1', available=available,
        success=all(row['met'] for row in rows.values()) if available else None,
        components=rows, simulation_time_s=now,
        scoring_changed=False, terminal_rule_changed=False)


def build_room(request, stage, manifest, output, *, episode_id, sensor,
               evaluation_mode, compiled=None):
    """Called only after entry admits source hashes, scene, bindings and paths."""
    raw = manifest.get('long_task_request')
    if not isinstance(raw, dict) or set(raw) != {'document_id', 'parameters'}:
        raise ValueError('Manifest must retain the admitted Long request')
    expected = bind_compiled(compile_task(raw['document_id'], raw['parameters']), manifest)
    if compiled is not None and compiled != expected:
        raise ValueError('Compiler result differs from manifest-bound request')
    if evaluation_mode != expected['runtime_id'] or request != expected['runtime_request']:
        raise ValueError('Long native mode/request differs from trusted recipe')
    manifest = prepare_scene(expected, stage, manifest, output)
    from openend.room import MultiTaskRoom
    from openend.tasks.all_objects_integration.workspace_physics import from_scene
    room = MultiTaskRoom(stage, manifest, output, episode_id, request=request,
        sensor=sensor, workspace_selector=from_scene(stage, manifest, sensor),
        native_factories=native_factories(expected))
    room._long_compiled = deepcopy(expected)
    return room


def accepted_session(room, *, episode_id, instruction):
    from short_tasks.agent_probe_v1.cap_v1.native_tools.openend_control_adapter_v1.session_adapter import OpenEndControlSession

    class LongControlSession(OpenEndControlSession):
        def result(self):
            result = super().result()
            spec = self.room._long_compiled
            result.update(task_name=spec['runtime_id'], document_id=spec['document_id'],
                          rule_version=spec['rule_version'], score_policy=deepcopy(spec['score_policy']),
                          success=None)
            if result['status'] == 'finished' and result['evaluation_valid']:
                result['long_completion'] = completion_evidence(self.room)
                # Native scoring can finish correctly while required completion
                # evidence is unavailable. Do not turn that into a valid zero.
                if not result['long_completion']['available']:
                    result['evaluation_valid'] = False
                    result['completion_status'] = 'unavailable'
                else:
                    result['success'] = result['long_completion']['success']
                    result['completion_status'] = 'success' if result['long_completion']['success'] else 'not_successful'
            else:
                result['long_completion'] = None
            return result

    if instruction != room._long_compiled['instruction']:
        raise ValueError('Public instruction differs from admitted Long task')
    return LongControlSession(room, episode_id=episode_id, instruction=instruction)
