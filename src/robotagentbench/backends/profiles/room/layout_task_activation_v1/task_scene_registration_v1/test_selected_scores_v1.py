"""Synthetic selected-clock regressions; never proof of physical manipulation."""
import unittest
from unittest.mock import patch
from selected_scores_v1 import SelectedCoffee, SelectedFryer, interrupt_hold
from multi_task_room_v2 import validate_request
from evaluation.tasks.coffee_machine.openended_composite_v1 import test_composite as coffee
from evaluation.tasks.air_fryer.continuous_openended_v1 import test_composite as fryer

class Tests(unittest.TestCase):
    def coffee(self):
        with patch.object(coffee, 'CoffeeComposite', SelectedCoffee): return coffee.Run()
    def fryer(self):
        with patch.object(fryer, 'AirFryerComposite', SelectedFryer): return fryer.Feed()
    def test_coffee_full_synthetic_sequence(self):
        r=self.coffee();r.place();r.pour();r.brew();r.retrieve()
        self.assertEqual(r.c.summary()['score'],21)
    def test_fryer_full_synthetic_sequence(self):
        r=self.fryer();r.insert(True);r.pull();r.take()
        self.assertEqual(r.p.summary()['score'],12)
    def test_coffee_pause_retains_score_and_phase(self):
        r=self.coffee();r.place();r.feed()
        before=r.c.summary()['score'];identity=id(r.c)
        interrupt_hold(r.c)
        self.assertEqual(r.c.summary()['score'],before)
        self.assertEqual(r.c.phase,'1-2');self.assertEqual(id(r.c),identity)
        r.pour();r.brew();r.retrieve();self.assertEqual(r.c.summary()['score'],21)
    def test_fryer_pause_retains_baseline_and_score(self):
        r=self.fryer();r.insert(True);r.pull()
        before=r.p.summary()['score'];baseline=r.p.basket_baseline
        interrupt_hold(r.p)
        self.assertEqual(r.p.summary()['score'],before)
        self.assertEqual(r.p.basket_baseline,baseline)
        r.take();self.assertEqual(r.p.summary()['score'],12)
    def test_coffee_partial_hold_must_restart(self):
        r=self.coffee();r.feed(cup=coffee.obj((0,0,.1),held=True))
        r.feed(12,cup=coffee.obj((.3,0,0),coffee.TRAY),placement=coffee.CupPlacement(.004,True,False))
        interrupt_hold(r.c)
        r.feed(10);self.assertEqual(r.c.phase,'1-1')
        r.feed(12);self.assertEqual(r.c.phase,'1-2')
    def test_registration_subsets(self):
        base=dict(no_progress_timeout_s=120,selection_order='front_then_distance')
        for tasks in ([],[dict(instance_id='c',type='coffee',parameters={})],
                      [dict(instance_id='f',type='fryer',parameters={'bread_variant':'sugar'})]):
            self.assertEqual(validate_request(dict(base,tasks=tasks)),tasks)
    def test_unknown_adapter_rejected(self):
        with self.assertRaises(ValueError):validate_request(dict(tasks=[dict(instance_id='x',type='unknown',parameters={})],no_progress_timeout_s=120,selection_order='front_then_distance'))
    def test_coffee_original_stage_budget_failure_preserved(self):
        r=self.coffee()
        for _ in range(1202):
            if r.c.ended:break
            r.feed()
        self.assertEqual(r.c.failure['reason'],'stage1_budget_exhausted')
    def test_fryer_original_stage_budget_failure_preserved(self):
        r=self.fryer()
        for _ in range(int(r.p.process.spec.time_limit_s/.05)+2):
            if r.p.termination:break
            r.push(target=fryer.ON_PLATE)
        self.assertEqual(r.p.failure,'stage1_budget_exhausted')
    def test_fryer_partial_hold_must_restart(self):
        r=self.fryer();r.push(target=fryer.ON_PLATE,tr=fryer.TRAY_OPEN)
        r.push(target=fryer.IN_BASKET,tr=fryer.TRAY_OPEN)
        r.push(12,target=fryer.IN_BASKET)
        interrupt_hold(r.p)
        r.push(10,target=fryer.IN_BASKET);self.assertEqual(r.p.phase,'2.1')
        r.push(12,target=fryer.IN_BASKET);self.assertEqual(r.p.phase,'2.3')

if __name__=='__main__':unittest.main()
