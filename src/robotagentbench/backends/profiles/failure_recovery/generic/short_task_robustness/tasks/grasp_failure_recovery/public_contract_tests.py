"""Full-source robot22 integration tests; synthetic state, no physics/model."""
import argparse
import ast
import base64
from copy import deepcopy
import io
import json
import unittest
from uuid import uuid4
from arena_api import bimanual, policy_execution as pe
from . import public_profiles as rp
from arena_api.task_runtime import WorkerContext, WorkerTask, TaskEvidence, TaskScore, TaskRegistration
from arena_api.task_lifecycle import TaskLifecycle
from short_tasks import registry
from .public_host import SessionHost, InputRejected
from .common import HERE, CURRENT, write_json
from .public_adapter import NAMES, request_for, verify_sources_for_plan
from .public_session import RecoverySession


class Registered(unittest.TestCase):
    def test_exact_two_names_and_source_ids(self):
        self.assertEqual(set(registry.FAILURE_RECOVERY_TASKS), set(NAMES))
        self.assertEqual({v[1] for v in NAMES.values()}, {'4-3','1-2'})
        for name in NAMES:
            self.assertIn(name, registry.TASK_NAMES)
            self.assertEqual(registry.task_directory(name), HERE)

    def test_score_goals_and_no_generic_bonus_changes(self):
        for name,(variant,api,maximum) in NAMES.items():
            plan = registry.resolve(request_for(name))
            self.assertEqual(plan['score_policy']['maximum'], maximum)
            self.assertEqual(plan['source_api'], api)
            self.assertEqual(plan['failure_recovery_variant'], variant)
            self.assertIsNone(plan['no_progress_timeout_s'])
            self.assertNotIn('navigation_progress', json.dumps(plan))
            self.assertTrue(plan['instruction'].isascii())

    def test_original_source_records_are_verified(self):
        for name in NAMES:
            result = registry.verify_sources(registry.resolve(request_for(name)))
            self.assertTrue(result['verified'])
            self.assertGreater(result['file_count'], 10)

    def test_rejects_semantic_or_scene_overrides(self):
        for name in NAMES:
            for key,value in [('layout_id',1),('options',{'new':True}),
                              ('targets',[{'category':'other','quantity':1}])]:
                with self.subTest(name=name,key=key), self.assertRaises(Exception):
                    registry.resolve(dict(request_for(name), **{key:value}))

    def test_existing_task_names_are_preserved(self):
        values = ast.parse((CURRENT/'short_tasks/registry.py').read_text())
        assignment = next(n for n in values.body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='TASK_NAMES' for t in n.targets))
        names = ast.literal_eval(assignment.value)
        self.assertEqual(len(names), len(set(names)))
        for name in ('stock_wooden_tray','prepare_coffee','select_drink','coffee_bag_interference'):
            self.assertIn(name,names)

    def test_original_mobile_controller_parameters_are_not_modified(self):
        old = ast.parse((CURRENT/'arena_api/mobile_base_native.py').read_text())
        new = ast.parse((HERE/'public_mobile.py').read_text())
        # Only a relative import is rebased for the frozen atomic API package.
        old.body = [n for n in old.body if not (isinstance(n,ast.ImportFrom) and n.level==1)]
        new.body = [n for n in new.body if not (isinstance(n,ast.ImportFrom) and n.level==1)]
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_shared_prepare_and_owner_delegate_before_generic_bootstrap(self):
        for name in ('prepare_task.py','native_owner.py'):
            text = (CURRENT/'short_tasks/agent_probe_v1'/name).read_text()
            self.assertIn('registry.FAILURE_RECOVERY_TASKS',text)
            self.assertIn('registry.failure_recovery_adapter()',text)

    def test_robot22_reuse_changes_imports_only_not_execution(self):
        class ImportsOnly(ast.NodeTransformer):
            def visit_ImportFrom(self,node): return None
        files=[('arena_api/robot_profiles.py','public_profiles.py'),
               ('arena_api/mobile_base.py','public_base.py'),
               ('short_tasks/agent_probe_v1/contract.py','public_contract.py'),
               ('short_tasks/agent_probe_v1/host.py','public_host.py'),
               ('short_tasks/agent_probe_v1/native_driver.py','public_driver.py')]
        for old,new in files:
            with self.subTest(new=new):
                first=ImportsOnly().visit(ast.parse((CURRENT/old).read_text()))
                second=ImportsOnly().visit(ast.parse((HERE/new).read_text()))
                self.assertEqual(ast.dump(first),ast.dump(second))


class Worker(WorkerTask):
    settle_steps = 0
    def __init__(self,context,world):
        self.context,self.world,self.index=context,world,0
    def validate_initial(self,metadata): pass
    def collect_evidence(self):
        value=TaskEvidence(self.context.episode_id,self.context.task_id,self.index,self.world.current_time,{})
        self.index+=1
        return value
    def evaluate(self,evidence): return TaskScore(0,4,{})
    def check_success(self,evidence,score): return False


class FakeDriver:
    def __init__(self,session,world):
        from PIL import Image
        self.session,self.world=session,world
        self.calls,self.advances=0,0
        self.limits={n:[-10.,10.] for n in rp.get(rp.MOBILE22).order}
        self.joints={n:0. for n in rp.NATIVE_JOINTS}
        stream=io.BytesIO();Image.new('RGB',(640,480)).save(stream,format='PNG')
        self.images={n:base64.b64encode(stream.getvalue()).decode() for n in ('head','left','right')}
        self.base=dict(position_world_m=[0.,0.,0.],orientation_world_xyzw=[0.,0.,0.,1.],
            linear_velocity_world_m_s=[0.,0.,0.],angular_velocity_world_rad_s=[0.,0.,0.],yaw_world_unwrapped_rad=0.)
    def validate_targets(self,targets):
        if set(targets)!=set(self.limits):raise ValueError('Wrong joints')
    def base_state(self):return deepcopy(self.base)
    def observe(self):
        return dict(episode_id=self.session.episode_id,step=self.session.control_index,
            time_s=self.world.current_time,instruction=self.session.plan['instruction'],
            joints=self.joints,base_state=self.base,image_step=self.session.control_index,images=self.images)
    def apply_action(self,targets,base):
        self.calls+=1
        self.last_base=dict(base)
    def advance(self,dt):
        self.advances+=1
        self.world.current_time+=dt
        return dict(events=[],docked_object_count=0,tray_dock_enabled=False)


class Robot22(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.world=SimpleNamespace(current_time=0.)
        context=WorkerContext('contract-only','synthetic',HERE/'synthetic-output',1,1200,{})
        worker=Worker(context,self.world)
        reg=TaskRegistration('synthetic','test',max_steps=1200,default_steps=1200,evaluator_id='original')
        life=TaskLifecycle(worker,context,reg,bimanual.CONTROL_DT)
        life.prepare_scene(HERE/'public_adapter.py');life.attach(None)
        life.before_physics_reset(self.world);life.robot_initialization();life.initialize(self.world)
        life.accept_initial({},0.)
        self.session=RecoverySession(life,self.world,'Pick up the bottle.')
        self.driver=FakeDriver(self.session,self.world)
        self.host=SessionHost(self.session,self.driver)
    def message(self,rows):
        return dict(op='execute_actions',request_id=str(uuid4()),episode_id=self.session.episode_id,
                    expected_step=self.session.control_index,actions=rows)
    def test_contract_has_twenty_two_physical_channels(self):
        contract=self.host.ready()['contract']
        self.assertEqual(len(contract['action_order']),22)
        self.assertEqual(contract['control_hz'],20)
        self.assertEqual(contract['action_order'][-2:],['base_forward_m_s','base_yaw_rate_rad_s'])
    def test_hold_routes_all_controls_through_original_lifecycle(self):
        reply=self.host.execute(self.message([[0.]*22 for _ in range(20)]))
        self.assertEqual(reply['actions_executed'],20)
        self.assertEqual(reply['observation']['step'],20)
        self.assertEqual(self.driver.advances,20*pe.SUBSTEPS)
        self.assertAlmostEqual(self.world.current_time,1.)
        self.assertEqual(self.session.result()['score'],0)
        self.assertEqual(set(self.driver.last_base),{'base_forward_m_s','base_yaw_rate_rad_s'})
    def test_invalid_later_row_does_not_execute_earlier_rows(self):
        for invalid in ([0.]*21,[0.]*23,[float('nan')]*22):
            with self.subTest(invalid=len(invalid)),self.assertRaises(InputRejected):
                self.host.execute(self.message([[0.]*22,invalid]))
        self.assertEqual((self.driver.calls,self.driver.advances),(0,0))
    def test_request_retry_is_idempotent(self):
        request=self.message([[0.]*22])
        first=self.host.execute(request)
        self.assertEqual(first,self.host.execute(request))
        self.assertEqual(self.driver.calls,1)
    def test_stale_request_does_not_move(self):
        request=self.message([[0.]*22]);request['expected_step']=1
        with self.assertRaises(InputRejected):self.host.execute(request)
        self.assertEqual(self.driver.calls,0)
    def test_observe_is_read_only_and_truth_filtered(self):
        first=self.host.observe();second=self.host.observe()
        self.assertEqual(first,second)
        self.assertEqual(self.world.current_time,0.)
        self.assertEqual(set(first),{'episode_id','step','time_s','instruction','joints','base_state','image_step','images'})
    def test_close_has_no_bonus_or_source_progress(self):
        self.host.close();self.host.close()
        result=self.session.result()
        self.assertTrue(result['evaluation_valid'])
        self.assertEqual(result['score'],0)
        self.assertEqual(result['reason'],'client_closed')
        self.assertFalse(result['success'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Registered,Robot22))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(args.output,dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        passed=result.wasSuccessful(),tests_only=True,physics_run=False,models_called=False))
    raise SystemExit(not result.wasSuccessful())
