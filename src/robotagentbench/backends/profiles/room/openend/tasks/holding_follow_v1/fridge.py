"""Read-only room adapter for fridge -> microwave holding facts."""
from dataclasses import replace
from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.native_source import NativeSource
from evaluation.tasks.microwave_put.isaac_evidence import final_facts
from evaluation.tasks.microwave_put.openended_fridge_to_microwave_v1.fridge_rules import CLOSED_DEG
from .detection import GraspDetector, HANDS


class FollowFridgeSource(NativeSource):
    def __init__(self, scene, clock, view):
        super().__init__(scene, clock, view)
        from evaluation.tasks.coffee_machine.openended_composite_v1.native_source import _BoundBodies
        for path in HANDS.values():
            self.views[path] = _BoundBodies(view, [path])
        self.detector = GraspDetector((scene.target,))

    def after_physics(self, helper_report, phase):
        super().after_physics(helper_report, phase)
        row = self.latest
        held, details = self.detector.observe(row['time_s'], row['details']['poses'])
        target = self.scene.target
        row['held'] = held[target]
        row['facts'] = replace(row['facts'], held_by_robot=held[target],
            held_by_right=details[target+':r']['confirmed'],
            released=bool(row['facts'].released and not held[target]))
        row['take_hold'] = bool(held[target] and row['out'] and all(a <= CLOSED_DEG for a in row['angles']))
        row['put_hold'] = final_facts(row['facts'])
        row['details']['position_follow'] = details
