"""Focused CPU-only mocks: no Codex process, model, Docker, GPU or Isaac."""
import base64,copy,json,os,stat,struct,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from robotagentbench.model_examples.codex import app_server,bridge,session
from robotagentbench.model_examples.codex.__main__ import run
from robotagentbench.interfaces import coding_contract,robot_profiles

ROOT=Path(os.environ.get('ROBOTAGENTBENCH_TEST_TMPDIR',str(Path(__file__).resolve().parent/'private')))

def png(w,h):
    return base64.b64encode(b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',w,h)+b'\0'*9).decode()

def observation():
    return dict(episode_id='mock-episode',step=0,time_s=0.,instruction='Mock task',
        joints={n:0. for n in robot_profiles.NATIVE_JOINTS},image_step=0,
        images={name:png(640,480)for name in ('head','left','right')},
        tcp={'left':None,'right':None},context='Public household memory',
        base_state=dict(position_world_m=[0.,0.,0.],orientation_world_xyzw=[0.,0.,0.,1.],
            linear_velocity_world_m_s=[0.,0.,0.],angular_velocity_world_rad_s=[0.,0.,0.],yaw_world_unwrapped_rad=0.),
        private_target_bindings={'must_not':'leak'},score=999)

class FakeWorker:
    def __init__(self,*args,**kwargs):self.calls=[];self.closed=False
    def read(self):
        c=coding_contract.specification({n:[-4.,4.]for n in coding_contract.ORDER[:20]})
        c['observation_fields']=[key for key in observation()if key not in ('private_target_bindings','score')]
        return dict(op='ready',policy_schema='short_tasks.coding_robot22.v1',contract=c,observation=observation())
    def exchange(self,row):
        self.calls.append(row)
        if row=={'op':'close'}:return dict(op='closed',result=dict(status='stopped',evaluation_valid=True,success=False,score=2.,scores={'total':2.,'maximum':5.}))
        return dict(op='result',actions_executed=0,episode_finished=False,observation=observation())
    def close(self):self.closed=True;return 0

class FakeServer:
    def __init__(self,*args,**kwargs):self.sent=[];self.events=0
    def request(self,name,args,**kwargs):
        self.sent.append((name,args))
        if name=='turn/start':return {'turn':{'id':'turn'}}
        return {}
    def start_thread(self,model,instructions,tools):return dict(model=model,thread={'id':'thread'})
    def next_event(self,*args):
        self.events+=1
        if self.events==1:return {'method':'item/completed','params':{'threadId':'thread','item':{'type':'reasoning','text':'PRIVATE_REASONING_DO_NOT_STORE'}}}
        return {'id':1,'method':'item/tool/call','params':dict(threadId='thread',turnId='turn',namespace='arena',callId='done1',tool='done',arguments={'reason':'mock stop'})}
    def send(self,row):self.sent.append(row)
    def close(self):pass

class Contracts(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT,prefix='mock-only-')
    def tearDown(self):self.tmp.cleanup()
    def test_public_filter_dedup_and_no_score(self):
        w=FakeWorker();b=bridge.RobotBridge(w,Path(self.tmp.name)/'obs')
        self.assertNotIn('private_target_bindings',b.observed);self.assertNotIn('score',b.observed)
        first=b.dispatch('one','observe',{});self.assertEqual(first,b.dispatch('one','observe',{}))
        self.assertEqual(len(w.calls),1)
        with self.assertRaises(RuntimeError):b.dispatch('one','done',{'reason':'different'})
        self.assertIn('200 model turns',b.instructions)
    def test_uncertain_ack_never_replayed(self):
        w=FakeWorker();b=bridge.RobotBridge(w,Path(self.tmp.name)/'obs')
        w.exchange=lambda row:dict(op='result',actions_executed=1,observation=observation())
        with self.assertRaises(RuntimeError):b.dispatch('one','observe',{})
        self.assertIsNotNone(b.pending)
        with self.assertRaises(RuntimeError):b.dispatch('two','observe',{})
    def test_original_thread_wire_isolation(self):
        server=app_server.AppServer.__new__(app_server.AppServer);server.transport=None
        server.scratch=SimpleNamespace(name='/mock/isolated');calls=[]
        def request(name,args):
            calls.append((name,args))
            if name=='account/read':return {'account':{'type':'chatgpt'}}
            if name=='config/read':return {'config':{'mcp_servers':{'private':{}}}}
            return {'model':'gpt-6-astra','thread':{'id':'thread'}}
        server.request=request;server.start_thread('gpt-6-astra','prompt',bridge.tool_specs())
        params=calls[-1][1]
        self.assertFalse(params['allowProviderModelFallback']);self.assertEqual(params['environments'],[])
        self.assertEqual(params['dynamicTools'][0]['name'],'arena')
        self.assertEqual(params['config']['mcp_servers']['private'],{'enabled':False})
    def test_one_command_mock_cleanup_and_owner_result(self):
        out=Path(self.tmp.name)/'run'
        value=dict(output=str(out),model='gpt-6-astra',effort='medium',max_model_responses=200,
            codex_executable='mock-codex',expected_cli_version='mock-only-version',transport=None,
            worker_argv=['mock-worker'],prepare_argv=[])
        with patch.object(app_server,'AppServer',FakeServer),patch('robotagentbench.model_examples.codex.protocol.Worker',FakeWorker),patch('subprocess.run',return_value=SimpleNamespace(stdout='mock-only-version\n')):
            self.assertEqual(run(value),0)
        result=json.loads((out/'operator_result.json').read_text())
        self.assertTrue(result['native_cleanup_verified']);self.assertTrue(result['evaluation_valid'])
        self.assertFalse(result['success']);self.assertEqual(result['score'],2.)
        report=(out/'observations/codex_report.json').read_text()
        self.assertNotIn('PRIVATE_REASONING_DO_NOT_STORE',report)
    def test_namespace_same_path_stdio_and_claim_once(self):
        from robotagentbench.model_examples.codex import namespace
        root=Path(self.tmp.name);runtime=root/'Runtime';runtime.mkdir()
        for name in ('code','sdk','locks','inputs'):(root/name).mkdir()
        value=dict(schema='robotagentbench.codex_namespace.v1',runtime=str(runtime),image='mock-image',name='mock',gpu=0,
            lock_root=str(root/'locks'),sdk_site=str(root/'sdk'),package_root=str(root/'code'),
            entry_argv=['python3','-m','reviewed.raw_worker'],readonly_roots=[],readonly_aliases={'/bank':str(root/'inputs')},
            docker_cli=None,docker_host=None,intent=str(root/'dispatch.json'))
        path=root/'namespace.json';path.write_text(json.dumps(value))
        base=['docker','run','--interactive','--name','mock-owned-container','mock-image','unused-original-example']
        with patch('robotagentbench.namespace.command',return_value=base),patch('robotagentbench.example.runtime',return_value=(runtime,runtime,{})),patch('subprocess.run',return_value=SimpleNamespace(returncode=0))as run:
            self.assertEqual(namespace.main([str(path)]),0);run.assert_not_called()
            self.assertFalse((root/'dispatch.json').exists())
            self.assertEqual(namespace.main([str(path),'--execute']),0)
            cmd=run.call_args.args[0]
            self.assertIn('--interactive',cmd);self.assertNotIn('--tty',cmd)
            self.assertEqual(cmd[-3:],['python3','-m','reviewed.raw_worker'])
            self.assertIn('type=bind,src='+str(runtime)+',dst='+str(runtime),cmd)
            self.assertIn('type=bind,src='+str(root/'inputs')+',dst=/bank,readonly',cmd)
            with self.assertRaises(FileExistsError):namespace.main([str(path),'--execute'])
            self.assertEqual(run.call_count,1)
            value['readonly_aliases']={'/bank/../escape':str(root/'inputs')}
            with self.assertRaises(ValueError):namespace.command(value)
    def test_stage_describe_and_existing_destination_reject(self):
        from robotagentbench.model_examples.codex import stage_runtime
        root=Path(self.tmp.name);provider=root/'provider';provider.mkdir()
        with patch('subprocess.run')as run:
            self.assertEqual(stage_runtime.main(['--runtime',str(provider),'--destination',str(root/'new')]),0)
            run.assert_not_called();self.assertFalse((root/'new').exists())
        with patch('robotagentbench.example.runtime',return_value=(provider,provider,{'source_files':{}})):
            with self.assertRaises(ValueError):stage_runtime.stage(provider,provider)
    def test_docker_group_only_preserves_existing_host_access(self):
        from robotagentbench.model_examples.codex.namespace import docker_socket_group
        info=SimpleNamespace(st_mode=stat.S_IFSOCK|0o660,st_uid=0,st_gid=77)
        with patch('os.stat',return_value=info),patch('os.getuid',return_value=2005),patch('os.getgid',return_value=2005),patch('os.getgroups',return_value=[77,2005]):
            self.assertEqual(docker_socket_group('/mock/operator.sock'),77)
            with patch('os.getgroups',return_value=[2005]):
                with self.assertRaises(ValueError):docker_socket_group('/mock/operator.sock')
            info.st_uid=2005
            self.assertIsNone(docker_socket_group('/mock/operator.sock'))

if __name__=='__main__':unittest.main()
