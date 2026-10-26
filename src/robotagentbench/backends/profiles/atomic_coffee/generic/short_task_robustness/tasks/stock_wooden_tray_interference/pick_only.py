"""Existing shelf-pick rules only; never enter carry-to-tray scoring."""
from dataclasses import replace
from short_tasks.tasks.stock_wooden_tray.adapter import StockComposite


class PickOnlyComposite(StockComposite):
    def _advance(self,frame):
        self.pick.dt=frame.physics_dt_s
        # Same recoverable process and outside-workspace policy as the base.
        result=self.pick.observe_batch((replace(frame.pick,target_outside_workspace=False),))
        self.committed_target=self.pick.committed_target
        if result['failure']:
            self.failure=dict(phase='pick',reason=result['failure'])
            self.reason=result['failure'];self.ended=True
            self.pick.finalize(self.reason)
        elif result['success']:
            self.completed_pick=self.pick.finalize('success')
            self.success=True;self.ended=True;self.reason='success';self.phase='picked'
            self.transitions.append(dict(sample=frame.pick.sample_index,time_s=frame.pick.time_s,
                from_phase='pick',to_phase='picked',target=self.committed_target,
                scene_reset=False,initialization_called=False))
        self.last=frame

    def summary(self):
        out=super().summary()
        pick=out['pick']
        out.update(task_kind='shelf_pick_interference',maximum_score=4,
            short_task_rule_version='openend.short_task.stock_wooden_tray_interference.v1',
            score=pick['score'] if pick else 0,place=None,
            completion_policy='original_pick_success_only',
            phase_budgets_s={'pick':out['phase_budgets_s']['pick']})
        return out
