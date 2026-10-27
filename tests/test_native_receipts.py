from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from robotagentbench.model_examples.native_receipts import collect, verify


def receipt_case(family, task, episode='fresh', output='/tmp/unique/native', *, shelf_api='4-3'):
    """CPU fixtures transcribed from original owners, not native-run evidence.

    Atomic: public_integration_v1/session.py::result + native_worker.py.
    Recovery: public_session.py::result, TaskLifecycle._snapshot, and
    room_owner_v2/v3.py with generic short_tasks/common/episode.py::finish.
    Cloth: owner/control_session.py::_finish + source_owner.py + supervisor.py.
    Only fields relevant to closure are retained; deliberately absent original
    fields (room Recovery evaluation_valid, generic shutdown flags) stay absent.
    """
    def lifecycle(api):
        return dict(task_id=api, episode_id=episode, valid=True, finalized=True,
                    score=0., maximum_score=6., success=False, terminated=False,
                    truncated=True, reason='client_closed')

    if family == 'atomic_coffee':
        api = '1-3' if task == 'coffee_button_lighting_interference' else '1-2'
        result = dict(schema='short_tasks.atomic_coffee.result.v1', status='completed',
                      task_name=task, episode_id=episode, evaluation_valid=True,
                      success=False, score=0., source_evaluation=lifecycle(api))
        native = dict(status='completed', task_name=task, cleanup_errors=[],
                      native_shutdown_complete=True, result=result)
        files = {'native_report.json': native}
    elif family == 'failure_recovery':
        variant = task.removeprefix('failure_recovery_')
        native = dict(status='completed', cleanup_errors=[], variant=variant,
                      original_native_initial_accepted=True)
        if variant in ('shelf_drink', 'coffee_bag'):
            api = shelf_api if variant == 'shelf_drink' else '1-2'
            evaluation = lifecycle(api)
            native.update(task_name=task, source_api=api, source_layers_unchanged=True,
                          task_result=dict(status='completed', evaluation_valid=True,
                              scoring_status='valid', score=0., success=False, evaluation=evaluation),
                          final_lifecycle_evaluation=deepcopy(evaluation))
        else:
            base = 'sofa_to_coffee_table' if variant == 'sofa' else 'clear_table_trash'
            native.update(task_local_robot22_connected=True, closed_result=dict(
                schema='openend.short_task.result.draft.v1', task_name=base, episode_id=episode,
                status='finished', success=False, termination_reason='client_closed',
                scores=dict(process=0., current_final=0., deductions=0., total=0., maximum=5.)))
        files = {'native_report.json': native}
    elif family == 'cloth':
        result = dict(status='finished', evaluation_valid=True, scoring_available=True,
                      task_name=task, episode_id=episode, success=False,
                      termination_reason='client_closed', scores={'total': 0., 'maximum': 5.})
        native = dict(schema='robotagentbench.original_cloth_owner.v18',
            status='source_native_session_closed', task_name=task, episode_id=episode,
            native_session_transport_accepted=True, native_shutdown_complete=True,
            render_process_shutdown_complete=True, final_result=result)
        formal = dict(schema='robotagentbench.formal_cloth_owner.v1', task_name=task,
            status='native_session_closed', physical_returncode=0, cleanup_errors=[],
            input_forward_errors=[], closed_ack_sent=True, lease_released=True,
            native_cleanup_accepted=True, final_result=deepcopy(result))
        files = {'source_owner_report.json': native, 'formal_owner_report.json': formal}
    else:
        raise ValueError('This fixture covers dedicated owners only')
    return dict(native_output=output, files=files)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.task = 'coffee_bag_interference'
        self.bundle = receipt_case('atomic_coffee', self.task)
        self.native = self.bundle['files']['native_report.json']

    def accepted(self, **kwargs):
        return verify('atomic_coffee', self.task, 'fresh', self.bundle,
                      closed_ack=kwargs.get('ack', True), worker_exit_code=kwargs.get('code', 0))

    def test_original_atomic_fields_do_not_require_invented_generic_flag(self):
        self.assertTrue(self.accepted())
        self.assertNotIn('source_layers_unchanged', self.native)

    def test_ack_exit_and_identity_are_all_required(self):
        for args in ({'ack': False}, {'code': 1}, {'code': False}):
            self.assertFalse(self.accepted(**args))
        self.native['result']['episode_id'] = 'previous'
        self.assertFalse(self.accepted())

    def test_partial_or_error_receipt_fails(self):
        for key, value in [('status', 'ready'), ('cleanup_errors', ['app']),
                           ('native_shutdown_complete', False), ('error', 'fault')]:
            before = deepcopy(self.native)
            self.native[key] = value
            self.assertFalse(self.accepted())
            self.native.clear(); self.native.update(before)

    def test_collector_reads_only_fixed_direct_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'native_report.json').write_text(json.dumps(self.native))
            (root/'private-unrelated.json').write_text('{"not_for_collection": true}')
            self.assertEqual(set(collect('atomic_coffee', root)['files']), {'native_report.json'})
            (root/'native_report.json').unlink()
            (root/'native_report.json').symlink_to(root/'private-unrelated.json')
            with self.assertRaises(ValueError): collect('atomic_coffee', root)

    def test_recovery_atomic_uses_original_nested_episode(self):
        bundle = receipt_case('failure_recovery', 'failure_recovery_coffee_bag')
        report = bundle['files']['native_report.json']
        args = ('failure_recovery', 'failure_recovery_coffee_bag', 'fresh', bundle)
        self.assertTrue(verify(*args, closed_ack=True, worker_exit_code=0))
        report['final_lifecycle_evaluation']['episode_id'] = 'other'
        self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))

    def test_all_four_atomic_lifecycles_are_finalized_and_task_bound(self):
        tasks = ('coffee_bag_interference', 'coffee_machine_push_interference',
                 'coffee_bag_then_machine_interference', 'coffee_button_lighting_interference')
        for task in tasks:
            with self.subTest(task=task):
                bundle = receipt_case('atomic_coffee', task)
                args = ('atomic_coffee', task, 'fresh', bundle)
                self.assertTrue(verify(*args, closed_ack=True, worker_exit_code=0))
                result = bundle['files']['native_report.json']['result']
                for key, value in (('task_id', 'wrong'), ('episode_id', 'old'),
                                   ('finalized', False), ('valid', False)):
                    saved = deepcopy(result['source_evaluation'])
                    result['source_evaluation'][key] = value
                    self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))
                    result['source_evaluation'] = saved
                result['source_evaluation'] = None
                self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))

    def test_all_five_recovery_original_owner_schemas(self):
        variants = ('shelf_drink', 'coffee_bag', 'sofa', 'table_waste_empty_grasp', 'table_waste_tipped_cup')
        for variant in variants:
            task = 'failure_recovery_' + variant
            with self.subTest(task=task):
                bundle = receipt_case('failure_recovery', task)
                args = ('failure_recovery', task, 'fresh', bundle)
                self.assertTrue(verify(*args, closed_ack=True, worker_exit_code=0))
                report = bundle['files']['native_report.json']
                self.assertNotIn('native_shutdown_complete', report)
                if variant not in ('shelf_drink', 'coffee_bag'):
                    self.assertNotIn('task_name', report)
                    self.assertNotIn('source_layers_unchanged', report)
                    self.assertNotIn('evaluation_valid', report['closed_result'])
                    result = report['closed_result']
                    for key, value in (('task_name', task), ('episode_id', 'old'),
                                       ('schema', 'unknown'), ('status', 'completed')):
                        saved = deepcopy(result)
                        result[key] = value
                        self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))
                        result.clear(); result.update(saved)
                report['cleanup_errors'] = ['original owner cleanup error']
                self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))

    def test_shelf_accepts_each_original_beverage_api_not_only_old_blue_bottle(self):
        task = 'failure_recovery_shelf_drink'
        for api in ('4-' + str(n) for n in range(1, 9)):
            bundle = receipt_case('failure_recovery', task, shelf_api=api)
            args = ('failure_recovery', task, 'fresh', bundle)
            self.assertTrue(verify(*args, closed_ack=True, worker_exit_code=0))
            report = bundle['files']['native_report.json']
            report['task_result']['evaluation']['task_id'] = '1-2'
            report['final_lifecycle_evaluation']['task_id'] = '1-2'
            self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))

    def test_all_three_cloth_task_receipts_match_both_native_owners(self):
        for task in ('spread_tablecloth', 'collect_tablecloth', 'long_task_18'):
            with self.subTest(task=task):
                bundle = receipt_case('cloth', task)
                args = ('cloth', task, 'fresh', bundle)
                self.assertTrue(verify(*args, closed_ack=True, worker_exit_code=0))
                formal = bundle['files']['formal_owner_report.json']
                for key, value in (('physical_returncode', False), ('physical_returncode', 1),
                                   ('closed_ack_sent', False), ('lease_released', False),
                                   ('input_forward_errors', ['forward error'])):
                    saved = deepcopy(formal)
                    formal[key] = value
                    self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))
                    formal.clear(); formal.update(saved)
                formal['final_result']['status'] = 'running'
                self.assertFalse(verify(*args, closed_ack=True, worker_exit_code=0))

    def test_malformed_nested_receipts_fail_closed_instead_of_raising(self):
        cases = [('failure_recovery', 'failure_recovery_coffee_bag', 'native_report.json', 'task_result'),
                 ('failure_recovery', 'failure_recovery_coffee_bag', 'native_report.json', 'final_lifecycle_evaluation'),
                 ('cloth', 'collect_tablecloth', 'source_owner_report.json', 'final_result')]
        for family, task, name, field in cases:
            for invalid in (None, [], 'malformed'):
                bundle = receipt_case(family, task)
                bundle['files'][name][field] = invalid
                self.assertFalse(verify(family, task, 'fresh', bundle, closed_ack=True, worker_exit_code=0))

    def test_unknown_or_extra_receipt_fails(self):
        self.bundle['files']['operator_claim.json'] = {'passed': True}
        self.assertFalse(self.accepted())


if __name__ == '__main__': unittest.main()
