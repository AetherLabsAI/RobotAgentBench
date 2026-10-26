"""Task-owned Short Task stage reuse; no proximity/interaction envelope."""
import json
from pathlib import Path

from ..contact_credit_v1.runtime import ContactRoom
from ..contact_credit_v1.policy import RULE_VERSION as CONTACT_RULE_VERSION
from openend.tasks.meal_tray.preparation import Preparation
from short_tasks.tasks.prepare_coffee.evaluator import (
    _PreparationCoffeeTask, verify_component_sources,
    RULE_VERSION as SHORT_COFFEE_RULE,
)

RULE_VERSION = 'openend.short_stages.contact_only.v3'


def coffee_preparation(binding):
    """Reuse the same process implementation, stop at the original handoff.

    Do NOT instantiate the Short Task registry/interaction/navigation envelope.
    The receiving meal task remains the sole scorer for retrieval and serving.
    """
    verify_component_sources()
    component = _PreparationCoffeeTask(binding)
    component.source = 'native_physx'
    return Preparation(component, kind='coffee')


def prepare_fridge(stage, manifest, request, output):
    """Call unchanged Short Task hooks once, in the pre-PhysX session layer."""
    selected = [t for t in request['tasks'] if t['type'] == 'fridge_to_microwave']
    if not selected:
        return manifest
    if len(selected) != 1:
        raise ValueError('One explicitly bound refrigerator payload required')
    from pxr import Usd
    from short_tasks.tasks.fridge_to_heated_box.initial_boxes import restore
    from short_tasks.tasks.fridge_to_heated_box.interior_contents import add
    variant = selected[0]['parameters']['payload_variant']
    plan = dict(runtime_components=[dict(parameters=dict(payload_variant=variant))],
                scene_request=dict(seed=manifest['seed']))
    session = stage.GetSessionLayer()
    if not session.anonymous:
        raise ValueError('Anonymous pre-physics episode layer required')
    before = session.ExportToString()
    try:
        with Usd.EditContext(stage, session):
            result = add(plan, stage, restore(plan, stage, manifest))
    except BaseException:
        session.ImportFromString(before)
        raise
    receipt = dict(
        schema='openend.short_fridge_scene_reuse.v1',
        payload_variant=variant,
        fridge_boxes=result['fridge_box_initialization'],
        cabinet=result['fridge_interior_initialization'],
        layer='anonymous_episode_session_before_physics',
        extra_runtime_tasks=False, selected_target_moved=False,
        native_validation=False,
    )
    with (Path(output).parent/'fridge_scene_alignment.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    # The hook already pinned every referenced background dependency. Preserve
    # the other task bindings and exact source objects supplied by composition.
    return result


class AlignedContactRoom(ContactRoom):
    def accept_initial(self):
        accepted = super().accept_initial()
        aligned = {}
        for key, kind in self.contact_ledger.kinds.items():
            if kind != 'coffee':
                continue
            port = self.ports[key]
            old = port.composite
            if (not isinstance(old, Preparation) or old.kind != 'coffee'
                    or old.success or old.ended or port.index != 0
                    or old.component.last.step != 0 or old.summary()['score'] != 0):
                raise ValueError('Coffee alignment requires a fresh linked preparation')
            # Existing CoffeeShortPort follows the same reset-time scorer swap.
            # Reuse the EXACT accepted measured frame; no sensor/world reread,
            # counters reset, physical step, or synthetic identity is introduced.
            from evaluation.tasks.coffee_machine.openended_composite_v1.composite import Frame
            measured = old.component.last
            replacement = coffee_preparation(port.binding)
            score = replacement.observe(Frame(port.binding, measured))
            if score['score'] != 0 or score['maximum_score'] != old.summary()['maximum_score']:
                raise ValueError('Coffee initial points or handoff maximum changed')
            port.composite, port.score = replacement, score
            aligned[key] = dict(
                short_task_stage_rule=SHORT_COFFEE_RULE,
                preparation_phases=['1-1','1-2','1-3'],
                retrieval_scored_here=False,
                maximum_score=score['maximum_score'],
                handoff_owner='original_coffee_to_meal_tray',
                short_task_navigation_envelope_used=False,
                facts_unchanged=True, initial_score=score['score'],
            )
        with (self.output/'stage_alignment_initial.json').open('x') as f:
            json.dump(dict(rule_version=RULE_VERSION, coffee=aligned,
                contact_rule=CONTACT_RULE_VERSION,
                proximity_points_enabled=False, native_initial_acceptance=accepted), f, indent=2)
        return accepted
