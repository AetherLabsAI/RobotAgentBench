"""Read-only components with Openend's approved single room timeout."""
from evaluation.tasks.coffee_machine.process_rules import Process as LegacyCoffeeProcess
from evaluation.tasks.air_fryer.continuous_openended_v1.composite import _HandoffProcess
from openend.tasks.holding_follow_v1.fryer import FollowValidation

class RoomTimeoutOnly:
    def _finish(self, facts, *, success=False, failure=None):
        # Called after physical failures, deductions, progress and success.
        # Do not mutate source Specs or alter the measured selected-time clock.
        if (not success and self._charging is not None and
                failure == 'stage%d_budget_exhausted' % self._charging):
            return
        return super()._finish(facts, success=success, failure=failure)

    def summary(self):
        result=super().summary()
        result['time_limit_s']=None
        result['timeout_owner']='openend_room_no_score_increase'
        result['stage_time_limits_enforced']=False
        result['budgets']={k:dict(used_s=v['used_s'],enforced=False)
                           for k,v in result['budgets'].items()}
        return result

class CoffeeProcess(RoomTimeoutOnly, LegacyCoffeeProcess):
    pass

class FryerProcess(FollowValidation, RoomTimeoutOnly, _HandoffProcess):
    pass
