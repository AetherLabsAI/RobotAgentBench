"""CPU synthetic contracts and stub HTTP only; never real model inference."""
import base64
from copy import deepcopy
import io
import json
import struct
import sys
import threading
import unittest
import zlib

from robotagentbench.interfaces.action_contract import ORDER, specification
from robotagentbench.interfaces import robot_profiles as rp, mobile_base
from .codec import actions_response, action_request, inference_request, state
from .http import Client, serve
from .bridge import run
from .config import validate_control_schema


def png(width,height):
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
    return base64.b64encode(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+
        chunk(b'IDAT',zlib.compress((b'\0'+bytes(width*3))*height))+chunk(b'IEND',b'')).decode()


def observation(step=0):
    return dict(episode_id='synthetic-test-episode',step=step,time_s=step*.05,
        instruction='Synthetic contract fixture, not a benchmark/model episode',
        joints=dict.fromkeys(rp.NATIVE_JOINTS,0.),image_step=step,
        images=dict.fromkeys(('head','left','right'),png(640,480)),context='Public synthetic context',
        base_state=dict(position_world_m=[1.,2.,0.],orientation_world_xyzw=[0.,0.,0.,1.],
            yaw_world_unwrapped_rad=7.,linear_velocity_world_m_s=[0.,0.,0.],angular_velocity_world_rad_s=[0.,0.,0.]))


def worker_contract():
    return specification(dict.fromkeys(ORDER[:20],[-3.14,3.14]))


class Contracts(unittest.TestCase):
    def setUp(self):
        self.obs=observation();self.contract=worker_contract()
        self.actions=[[0.]*22 for _ in range(2)]

    def test_state_order_and_allowlist(self):
        for index,name in enumerate(ORDER[:20]):self.obs['joints'][name]=index*.01
        self.obs['private_answer']={'target':'must not leak'}
        self.assertEqual(state(self.obs),[i*.01 for i in range(20)]+[1.,2.,7.])
        req=inference_request(self.contract,self.obs)
        self.assertNotIn('private_answer',req['observation'])
        self.assertEqual(req['observation']['context'],'Public synthetic context')

    def test_whole_chunk_fail_closed(self):
        value=actions_response(self.obs,self.actions)
        req,last=action_request(value,self.obs,self.contract,mobile_base.ZERO)
        self.assertEqual(req['actions'],self.actions)
        for field,replacement in [('normalized',True),('step',1),('action_order',list(reversed(ORDER)))]:
            bad=deepcopy(value);bad[field]=replacement
            with self.assertRaises(ValueError):action_request(bad,self.obs,self.contract,mobile_base.ZERO)
        for index,number in [(0,float('nan')),(14,100.),(20,.18)]:
            bad=deepcopy(value);bad['actions'][1][index]=number
            with self.assertRaises(ValueError):action_request(bad,self.obs,self.contract,mobile_base.ZERO)
        bad=deepcopy(value);bad['actions'][1][20:]=[.001,.001]
        with self.assertRaises(ValueError):action_request(bad,self.obs,self.contract,mobile_base.ZERO)

    def test_not_a_generic_robot_schema(self):
        with self.assertRaises(ValueError):validate_control_schema({'version':'aloha','action_dim':14})

    def test_g1_schema_requires_absolute_and_native_gripper(self):
        units=[rp.NATIVE_UNITS[n]for n in ORDER[:20]]+['m/s','rad/s']
        schema=dict(version=rp.MOBILE22,action_dim=22,state_dim=23,control_hz=20,
            action_channels=[dict(name=n,unit=u,representation=r)for n,u,r in zip(ORDER,units,
                ['absolute_joint_position_target']*20+['body_forward_velocity','world_z_yaw_rate'])],
            state_channels=[dict(name=n,unit=u,representation=r)for n,u,r in zip(
                ORDER[:20]+['base_x_world_m','base_y_world_m','base_yaw_world_unwrapped_rad'],
                units[:20]+['m','m','rad'],['measured_joint_position']*20+[
                    'measured_world_position','measured_world_position','measured_world_unwrapped_yaw'])])
        validate_control_schema(schema)
        for index,field,bad_value in [(0,'representation','delta_joint_position'),(18,'unit','normalized_0_1')]:
            bad=deepcopy(schema);bad['action_channels'][index][field]=bad_value
            with self.assertRaises(ValueError):validate_control_schema(bad)

    def test_endpoint_and_timeout_fail_closed(self):
        for url in ['http://user:secret@localhost/infer','http://localhost/infer?token=x','http://localhost/other']:
            with self.assertRaises(ValueError):Client(url,1)
        for timeout in [float('nan'),0,-1,True]:
            with self.assertRaises(ValueError):Client('http://localhost/infer',timeout)

    def test_navigation_preserved_not_consumed(self):
        from robotagentbench.interfaces.observation_extras import MAP_SCHEMA,MAP_SOURCE
        self.obs['navigation']=dict(schema=MAP_SCHEMA,step=0,map_png=png(3,3),origin_m=[0.,0.],
            resolution_m=.04,shape=[1,1],explored_fraction=.1,map_source=MAP_SOURCE,
            premapped=False,task_labels=False,body_footprint_radius_m=.3)
        self.assertEqual(inference_request(self.contract,self.obs)['observation']['navigation'],self.obs['navigation'])
        self.assertIs(actions_response(self.obs,self.actions)['navigation_consumed'],False)

    def test_stub_http_to_existing_jsonl_worker(self):
        calls=[];actions=self.actions
        class StubPolicy:
            def infer(self,request):
                calls.append(request)
                return actions_response(request['observation'],actions)
        server=serve(StubPolicy(),port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_address[1])+'/infer'
            incoming=io.StringIO('\n'.join(json.dumps(row)for row in [
                dict(op='ready',contract=self.contract,observation=self.obs),
                dict(op='result',actions_executed=2,episode_finished=True,observation=observation(2)),
                dict(op='closed',result={'synthetic':True})])+'\n')
            outgoing=io.StringIO()
            result=run(incoming,outgoing,Client(url,2),max_calls=1)
            sent=[json.loads(line)for line in outgoing.getvalue().splitlines()]
            self.assertEqual([row['op']for row in sent],['execute_actions','close'])
            self.assertEqual(sent[0]['expected_step'],0)
            self.assertTrue(result['worker_closed']);self.assertEqual(len(calls),1)
            self.assertNotIn('torch',sys.modules)
        finally:server.shutdown();server.server_close();thread.join()

    def test_invalid_later_row_sends_no_robot_action(self):
        prediction=actions_response(self.obs,self.actions)
        prediction['actions'][1][0]=100.
        class Invalid:
            def infer(self,_):return prediction
        incoming=io.StringIO(json.dumps(dict(op='ready',contract=self.contract,observation=self.obs))+'\n')
        outgoing=io.StringIO()
        with self.assertRaises(ValueError):run(incoming,outgoing,Invalid(),max_calls=1)
        self.assertEqual(outgoing.getvalue(),'')


if __name__=='__main__':unittest.main()
