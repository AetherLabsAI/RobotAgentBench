"""Task-owned combinations of existing pickup/drawer components.

Only the explicit sofa-to-table destination uses the approved new three-point
current-state rule. No source module, physics value or pickup rule is modified.
"""
from types import SimpleNamespace
from openend.tasks.object_transfers import scoring as old
from openend.tasks.object_transfers.binding import Binding, ContactSelection
from evaluation.tasks.tidy_sofa.specs import spec
from arena_api.policy_execution import PHYSICS_DT


class LegacyPickup:
    def __init__(self, asset, episode, target):
        if asset == 'remote_tv':
            from openend.tasks.remote_transfer import components
        elif asset == 'glasses_case':
            from openend.tasks.glasses_transfer import components
        else:
            from openend.tasks.wallet_transfer import components
        self.components = components
        self.core = components.RecoverablePick(episode, PHYSICS_DT, target)
        self.spec = SimpleNamespace(maximum_score=3)
        self.remote = asset == 'remote_tv'
        self.last = None

    def observe(self, f, *, active, pickup_phase):
        self.last = f
        p = self.components.pick.PhysicalFrame(f.episode_id, f.sample_index,
            f.time_s, self.core.target, f.target_grasped,
            not f.target_supported, f.target_on_floor)
        if self.remote:
            return self.core.observe_one(p, active, validated=True)
        return self.core.observe_one(p, active)

    def summary(self):
        return self.core.summary()


class CurrentSurface:
    """Three reversible points in the actual legal region, no placement bonus."""
    def __init__(self):
        self.current_final = False

    def observe(self, f, b, *, enabled, final_sample):
        self.current_final = bool(enabled and f.in_zone and not f.on_floor
            and f.supports == frozenset((b.destination,)))
        return bool(self.current_final and not f.held and not f.robot_contact and final_sample)

    def summary(self):
        points = 3 if self.current_final else 0
        return dict(earned=points, raw_score=points, collision_deduction=0,
            stages={'current_destination': points}, events=[],
            current_final=self.current_final, operation_complete=True)


class CurrentDrawer(old.Placement):
    def observe(self, f, b, *, enabled, final_sample):
        done=super().observe(f,b,enabled=enabled,final_sample=final_sample)
        # Historical close is a process milestone, not proof it is still closed.
        if f.drawer_travel_m > old.DRAWER_CLOSED_M:
            self.current_final=False
            self.awarded.pop('place',None)
            return False
        return done


class Transfer(old.Transfer):
    def __init__(self, binding, asset, mode, *, workspace_rates=False):
        if not isinstance(binding, Binding) or type(workspace_rates) is not bool:
            raise ValueError('Explicit bound scene and rate mode required')
        if mode not in ('surface', 'drawer'):
            raise ValueError('Explicit approved destination required')
        allowed = ('wallet', 'notebook_green') if mode == 'drawer' else (
            'book_red', 'remote_tv', 'wallet', 'glasses_case')
        if asset not in allowed:
            raise ValueError('Unsupported combination')
        self.b, self.workspace_rates = binding, workspace_rates
        self.d = SimpleNamespace(mode=mode, height_reference='rigid_origin',
            placement_api='16-1' if asset == 'notebook_green' else '16-2')
        if asset in ('book_red', 'notebook_green'):
            self.pick = old.Pickup(binding.episode_id, PHYSICS_DT,
                spec('15-1' if asset == 'book_red' else '13-6'))
        else:
            self.pick = LegacyPickup(asset, binding.episode_id, binding.target)
        self.place = CurrentDrawer(self.d, binding.episode_id) if mode == 'drawer' else CurrentSurface()
        self.selection = ContactSelection()
        self.last = None
        self.permit = self.ended = self.finalized = False
        self.reason = None
        self.asset = asset
        self.maximum = self.pick.spec.maximum_score + (spec(self.d.placement_api).maximum_score
            if mode == 'drawer' else 3)

    def finalize(self, reason):
        if not self.finalized and isinstance(self.pick, LegacyPickup) and not self.pick.core.ever_complete:
            self.pick.core.hold_deduction = self.pick.components.pick.HOLD_PENALTY
        return super().finalize(reason)

    def summary(self):
        p, q = self.pick.summary(), self.place.summary()
        final_key = 'place' if self.d.mode == 'drawer' else 'current_destination'
        final = q['stages'].get(final_key, 0)
        process = p['earned_points'] + q['earned'] - final
        deductions = p['earned_points'] - p['raw_score'] + q['earned'] - q['raw_score']
        return dict(instance_id=self.b.instance_id, asset=self.asset,
            target_path=self.b.target, destination_path=self.b.destination,
            maximum_score=self.maximum, score=max(0, process+final-deductions),
            raw_score=process+final-deductions, pickup=p, placement=q,
            scores=dict(process=process,current_final=final,deductions=deductions,
                total=max(0,process+final-deductions),maximum=self.maximum),
            operation_complete=bool(getattr(self,'pickup_complete',False) and q['operation_complete']),
            success=bool(getattr(self,'current_success',False) and self.last
                and not self.last.held and not self.last.robot_contact), ended=self.ended,
            finalized=self.finalized,reason=self.reason, timeout_owner='room',
            selection=self.selection.request(ended=self.ended),
            evidence_status='rule_core_not_native_reader')
