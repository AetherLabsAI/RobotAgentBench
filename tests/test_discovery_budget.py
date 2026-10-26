"""Task-local rule and existing-owner binding tests; no native physics mocks as scores."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from source_fixtures import SourceFixture
from robotagentbench.simulation.original_backend import bind_original_backend, ROOM_OWNER

from robotagentbench.tasks.discovery.policy import CoverageLedger, RULE_VERSION
from robotagentbench.tasks.discovery.binding import install, describe_plan, DiscoveryControlSession

LEFT = '/World/Robot/gripper_l_finger'
RIGHT = '/World/Robot/gripper_r_finger'


def ledger(n=2):
    return CoverageLedger(episode_id='test',
        targets={f't{i}': (f'/World/Target{i}', f'/World/TargetAlias{i}') for i in range(n)},
        non_targets={f'w{i}': (f'/World/Wrong{i}',) for i in range(7)}, fingers=(LEFT, RIGHT))


def pair(path, finger=LEFT, samples=True):
    return ((finger+'/collision', path), [dict(measured=True)] if samples else [])


class BudgetRules(unittest.TestCase):
    def setUp(self):
        self.value = ledger()
        self.value.arm(0.)
        self.time = 0.

    def touch(self, *pairs, accepting=True):
        self.time += .01
        return self.value.observe(pairs, episode_id='test', time_s=self.time, accepting_contacts=accepting)

    def test_unique_physical_aliases_both_hands_and_repeat_once(self):
        events = self.touch(pair('/World/Target0/mesh'), pair('/World/TargetAlias0/other', RIGHT))
        self.assertEqual(len(events), 1)
        self.touch(pair('/World/Target0/mesh', RIGHT))
        self.assertEqual(self.value.touch_count, 1)
        self.assertEqual(self.value.report()['score'], 1)
        self.assertEqual(self.value.touch_limit, 4)

    def test_wrong_targets_consume_budget_without_score_penalty(self):
        self.touch(pair('/World/Wrong0/mesh'))
        self.touch(pair('/World/Wrong0/mesh', RIGHT))
        self.assertEqual(self.value.report()['touch_budget'], dict(limit=4, used=1, remaining=3, exhausted=False))
        self.assertEqual(self.value.report()['score'], 0)

    def test_last_budget_contact_still_earns_coverage(self):
        for i in range(3): self.touch(pair(f'/World/Wrong{i}/mesh'))
        events = self.touch(pair('/World/Target0/mesh'))
        self.assertEqual(events[0]['touch_index'], 4)
        self.assertTrue(self.value.budget_exhausted)
        self.assertEqual(self.value.report()['coverage'], .5)
        self.assertEqual(self.touch(pair('/World/Target1/mesh')), [])
        self.assertEqual(self.value.report()['score'], 1)

    def test_two_mistakes_still_allow_complete_coverage(self):
        for p in ('/World/Wrong0','/World/Wrong1','/World/Target0','/World/Target1'):
            self.touch(pair(p))
        result = self.value.finalize(evaluation_valid=True, termination_reason='all_targets_contacted')
        self.assertTrue(result['success'])
        self.assertEqual(result['score'], 2)
        self.assertEqual(result['touch_budget']['used'], 4)

    def test_same_substep_keeps_source_order_and_never_exceeds_limit(self):
        value = CoverageLedger(episode_id='test', targets={'z': ('/World/Target',)},
            non_targets={x: (f'/World/{x}',) for x in 'abcd'}, fingers=(LEFT, RIGHT))
        value.arm(0.)
        events = value.observe([pair('/World/Target'), *(pair('/World/'+x) for x in 'dcba')],
            episode_id='test', time_s=.01, accepting_contacts=True)
        self.assertEqual([e['object_id'] for e in events], ['a', 'b', 'c'])
        self.assertEqual(value.touch_count, 3)
        self.assertEqual(value.report()['score'], 0)

    def test_nonfinger_self_and_empty_contact_do_not_count(self):
        self.touch(pair('/World/Target0', '/World/Robot/arm'),
            pair('/World/Robot/arm'), pair('/World/Target0', samples=False))
        self.assertEqual(self.value.touch_count, 0)

    def test_late_contact_not_banked_and_no_time_reset(self):
        self.touch(pair('/World/Target0'), accepting=False)
        self.assertEqual(self.value.touch_count, 0)
        with self.assertRaises(ValueError):
            self.value.observe([], episode_id='test', time_s=self.time, accepting_contacts=True)
        with self.assertRaises(ValueError):
            self.value.observe([], episode_id='other', time_s=1., accepting_contacts=True)

    def test_unbound_contact_is_infrastructure_error_not_free_touch(self):
        with self.assertRaises(ValueError): self.touch(pair('/World/Unbound'))
        self.assertEqual(self.value.touch_count, 0)

    def test_invalid_settlement_is_not_model_zero_and_finalization_idempotent(self):
        self.touch(pair('/World/Target0'))
        first = self.value.finalize(evaluation_valid=False, termination_reason='native_error')
        self.assertIsNone(first['score'])
        self.assertEqual(first['observed_coverage'], .5)
        self.assertEqual(first, self.value.finalize(evaluation_valid=False, termination_reason='native_error'))

    def test_ambiguous_catalogue_rejected(self):
        with self.assertRaises(ValueError):
            CoverageLedger(episode_id='test', targets={'x': ('/World/A',)},
                non_targets={'y': ('/World/A/mesh',)}, fingers=(LEFT,))


class BudgetBinding(unittest.TestCase):
    def room(self, root):
        return NS(discovery=ledger(), started=False, closed=False, faulted=False,
                  pending=None, output=root, dispatch=NS(ended=False, current='t0', termination_reason=None))

    def test_plan_only_changes_approved_task_rule(self):
        plan = dict(task_name='task_discovery', rule_version='original',
                    score_policy=dict(maximum=2, metric='unique_target_object_coverage'))
        updated = describe_plan(plan)
        self.assertEqual(updated['rule_version'], RULE_VERSION)
        self.assertEqual(plan['rule_version'], 'original')
        self.assertEqual(updated['score_policy']['maximum'], 2)
        with self.assertRaises(ValueError): describe_plan(dict(task_name='openend'))

    def test_install_only_before_native_acceptance(self):
        with TemporaryDirectory() as root:
            room = self.room(root)
            self.assertIs(install(room), room)
            self.assertTrue((Path(root)/'discovery_touch_budget_rule.json').is_file())
            room.discovery.arm(0.)
            with self.assertRaises(ValueError): install(room)

    def test_real_frame_finishes_and_scores_before_budget_settlement(self):
        with TemporaryDirectory() as root:
            room = self.room(root); install(room); room.discovery.arm(0.)
            room.started=True
            for index in range(3):
                room.discovery.observe([pair('/World/Wrong'+str(index))],
                    episode_id='test', time_s=(index+1)*.01, accepting_contacts=True)
            room.discovery.observe([pair('/World/Target0')], episode_id='test',time_s=.04,accepting_contacts=True)
            order=[]
            def settle():
                order.append('settle')
                self.assertEqual(room.dispatch.termination_reason, 'touch_budget_exhausted')
                room.terminal_result=room.discovery.finalize(evaluation_valid=True,
                    termination_reason=room.dispatch.termination_reason)
            room._settle_terminal=settle
            def finish(**kwargs):
                order.append('original_finish')
                self.assertEqual(kwargs, {'completed_substeps':3})
            original=NS(room=room, faulted=False, finish_control=finish,
                result=lambda:dict(evaluation_valid=room.dispatch.ended, status='finished' if room.dispatch.ended else 'running'),
                close=Mock(),begin_control=Mock(),after_physics=Mock())
            session=DiscoveryControlSession(original)
            result=session.finish_control(completed_substeps=3)
            self.assertEqual(order,['original_finish','settle'])
            self.assertEqual(result['discovery']['score'],1)
            self.assertTrue(result['evaluation_valid'])
            session.close();original.close.assert_called_once()

    def test_existing_terminal_reason_is_not_overwritten(self):
        room=NS(discovery=ledger(1),dispatch=NS(ended=True,termination_reason='all_targets_contacted'))
        room.discovery.arm(0.)
        original=NS(room=room,faulted=False,finish_control=Mock(),result=lambda:{},close=Mock())
        session=DiscoveryControlSession(original)
        session.finish_control(completed_substeps=3)
        self.assertEqual(room.dispatch.termination_reason,'all_targets_contacted')

    def test_formal_original_backend_binds_plan_room_and_session(self):
        with TemporaryDirectory() as folder:
            fixture=SourceFixture(folder)
            fixture.write(ROOM_OWNER.replace('.', '/')+'.py', '''
from types import SimpleNamespace as NS
def admit_launch(launch, output):
    return {}, {}, dict(task_name='task_discovery',rule_version='original',score_policy=dict(maximum=2))
def build_room(*args, **kwargs):
    return NS(started=False,closed=False,faulted=False,pending=None,output=args[0],
        discovery=NS(episode_id='cpu',origin=None,final_result=None,target_events={},wrong_events={},
            targets={'a':('/World/A',),'b':('/World/B',)},non_targets={},
            fingers=('/World/Robot/gripper_l_finger',)))
def accepted_session(room, **kwargs):
    return NS(room=room)
''')
            with bind_original_backend(fixture.tree()) as backend:
                request,manifest,plan=backend.admit_room({'task_name':'task_discovery'},folder)
                self.assertEqual(plan['rule_version'],RULE_VERSION)
                room=backend.build_room(folder,evaluation_mode='task_discovery')
                self.assertIsInstance(room.discovery,CoverageLedger)
                self.assertEqual(room.discovery.touch_limit,4)
                self.assertIsInstance(backend.accept_room(room),DiscoveryControlSession)


if __name__ == '__main__': unittest.main()
