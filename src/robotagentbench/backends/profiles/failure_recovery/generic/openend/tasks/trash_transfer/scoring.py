"""Approved cup pickup policy for four garbage classes; legacy rules untouched."""
from types import SimpleNamespace
from openend.tasks.object_transfers.binding import Binding, ContactSelection
from openend.tasks.object_transfers.scoring import Transfer, Pickup, Placement
from arena_api.policy_execution import PHYSICS_DT
from evaluation.tasks.tidy_sofa.specs import spec
from .registration import CATEGORIES


class TrashTransfer(Transfer):
    def __init__(self, binding, category, *, workspace_rates=False):
        if not isinstance(binding, Binding) or binding.key != 'trash_transfer':
            raise ValueError('Scene-bound garbage task required')
        if category not in CATEGORIES or type(workspace_rates) is not bool:
            raise ValueError('Approved garbage category/workspace mode required')
        self.workspace_rates = workspace_rates
        self.b = binding; self.category = category
        self.d = SimpleNamespace(mode='trash', height_reference='rigid_origin',
            placement_api=CATEGORIES[category][2])
        # 13-7 is a reused rule template, not an invented paper/snack/can API ID.
        self.pick = Pickup(binding.episode_id, PHYSICS_DT, spec('13-7'))
        self.place = Placement(self.d, binding.episode_id)
        self.selection = ContactSelection()
        self.last = None; self.permit = False; self.ended = False
        self.reason = None; self.finalized = False

    def summary(self):
        return dict(super().summary(), task_type='trash_transfer', category=self.category,
            pickup_rule_template='13-7_user_approved_reuse',
            disposal_component=CATEGORIES[self.category][2])
