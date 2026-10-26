"""Reward-only room selected ONLY by the ordinary OpenEnd factory branch."""
import json
from evaluation.tasks.refrigerator_take.process_rules import STAGE_POINTS
from ..scenario_alignment_v1.runtime import AlignedContactRoom
from .policy import PositiveLedger, RULE_VERSION


class PositiveOnlyRoom(AlignedContactRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ContactDispatcher's normal and terminal submissions, as well as
        # ContactRoom.accept_initial, all use this same explicit score ledger.
        ledger = PositiveLedger(self.contact_ledger, fridge_points=dict(STAGE_POINTS))
        self.contact_ledger = ledger
        self.dispatch.ledger = ledger
        contact_rule_path = self.output/'contact_rule.json'
        contact_rule = json.loads(contact_rule_path.read_text())
        contact_rule.update(rule_version=RULE_VERSION, numerical_deductions_enabled=False)
        contact_rule_path.write_text(json.dumps(contact_rule, indent=2)+'\n')
        with (self.output/'positive_score_rule.json').open('x') as stream:
            json.dump(dict(rule_version=RULE_VERSION, numerical_deductions_enabled=False,
                scope='ordinary_openend_only', contact_points_per_registered_instance=1,
                task_maxima_unchanged=True, physical_predicates_unchanged=True,
                physics_and_safety_guards_unchanged=True), stream, indent=2)
            stream.write('\n')
