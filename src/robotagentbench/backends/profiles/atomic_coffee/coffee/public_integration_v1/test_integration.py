"""Pure contract/gating regressions; not native task success evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from . import adapter


class AdmissionTests(unittest.TestCase):
    def test_four_explicit_tasks_one_scorer_each(self):
        from short_tasks import registry
        for name,(api,variant,_) in adapter.NAMES.items():
            with self.subTest(name=name):
                plan=registry.resolve(adapter.request_for(name))
                self.assertTrue(plan['available'])
                self.assertEqual(plan['public_task_count'],1)
                self.assertEqual(len(plan['runtime_components']),1)
                self.assertEqual(plan['source_api'],api)
                self.assertEqual(plan['variant'],variant)
                self.assertEqual(plan['score_policy']['maximum'],5 if api=='1-3' else 6)
                self.assertEqual(plan['no_progress_timeout_s'],120.)
                self.assertIsNone(plan['score_policy']['time_limit_s'])
                self.assertFalse(plan.get('interaction_policy'))
                self.assertFalse(plan.get('navigation_policy'))

    def test_no_fake_layout_or_ignored_inputs(self):
        from short_tasks import registry
        for name in adapter.NAMES:
            for override in ({'layout_id':30},{'options':{'threshold':0}},
                             {'targets':[dict(category='cup',quantity=2)]}):
                with self.subTest(name=name,override=override), self.assertRaises(ValueError):
                    registry.resolve(dict(adapter.request_for(name),**override))

    def test_wire_contract_is_import_only_copy(self):
        for name,old,new in (
            ('host.py','from arena_api import mobile_base','from .robot_api import mobile_base'),
            ('contract.py','from arena_api import mobile_base, robot_profiles as rp',
                           'from .robot_api import mobile_base, robot_profiles as rp')):
            expected=(adapter.ROOT/'short_tasks/agent_probe_v1'/name).read_text().replace(old,new)
            self.assertEqual((adapter.HERE/name).read_text(),expected)

    def test_same_public_names_and_dimensions(self):
        from .contract import ORDER, specification
        from .robot_api import robot_profiles as rp
        wire=specification({n:[-1.,1.] for n in ORDER[:20]})
        self.assertEqual(len(ORDER),22)
        self.assertEqual(ORDER[:20],list(rp.get(rp.MOBILE22).order))
        self.assertEqual(len(rp.NATIVE_JOINTS),34)
        self.assertEqual(wire['schema'],'short_tasks.gpt.robot22.v1')

    def test_output_cannot_escape_selected_task(self):
        for name in adapter.NAMES:
            with self.assertRaises(ValueError):
                adapter.owned(name,adapter.ROOT/'unrelated_output')


class GateTests(unittest.TestCase):
    def worker(self,variant):
        from .session import make_runtime
        from evaluation.tasks.coffee_machine.specs import episodes
        context=SimpleNamespace(task_id='1-2',task_parameters=dict(source_episode=episodes('1-2')[0],random_layout=False),
            output=adapter.MACHINE/'public_integration_v1/test_not_created')
        return make_runtime(context,dict(appearance={},scene='/unused'),variant)

    def test_real_hold_not_credited_before_relocation(self):
        from evaluation.tasks.coffee_machine.tests.test_process_rules import facts,HELD,SCENES
        for variant in ('bag_only','bag_then_machine'):
            worker=self.worker(variant)
            worker.armed=True
            worker.actor=SimpleNamespace(completed=False,retired=False,robot_process_scoring_enabled=False)
            p=worker.process
            for step in range(2):
                p.observe(facts(step,**dict(SCENES['1-2'],bag=HELD)))
            self.assertEqual(p.score,0)
            worker.actor.completed=True
            worker.actor.retired=True
            worker.actor.robot_process_scoring_enabled=True
            p.observe(facts(2,**dict(SCENES['1-2'],bag=HELD)))
            self.assertEqual(p.score,2)
            p.observe(facts(3,**dict(SCENES['1-2'],bag=HELD)))
            self.assertEqual(p.score,2)  # no repeated award, no second scorer
            self.assertEqual(p.summary()['budgets'],{})

    def test_gate_does_not_hide_failure_with_new_progress_deadline(self):
        from evaluation.tasks.coffee_machine.tests.test_process_rules import facts,SCENES,GROUND
        worker=self.worker('bag_only')
        p=worker.process
        p.observe(facts(0,**dict(SCENES['1-2'],bag=dict(ground_contact=True,supports=frozenset({GROUND})))))
        self.assertEqual(p.failure,'bag_on_ground')
        self.assertIsNone(p.spec.time_limit_s)

    def test_machine_variant_has_unchanged_process(self):
        from evaluation.tasks.coffee_machine.process_rules import Process
        self.assertIs(type(self.worker('machine_only').process),Process)


if __name__=='__main__':
    unittest.main()
