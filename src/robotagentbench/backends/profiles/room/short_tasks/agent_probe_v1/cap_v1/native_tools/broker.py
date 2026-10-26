"""Trusted robot-only broker; policy code never executes on the Isaac host."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .robot_contract import ORDER, public_robot, public_observation
from .host import InputRejected
from .sandbox import Sandbox

HERE = Path(__file__).resolve().parent
ROBOT = json.loads((HERE/'starter/robot/robot_contract.json').read_text())
GRIPPERS = dict(left='idx41_gripper_l_outer_joint1', right='idx81_gripper_r_outer_joint1')
SCHEMA = 'short_tasks.coding_robot22.v1'


def arm_targets(row, state, previous, limits):
    """Normalized legacy gripper -> native radians, retaining ALL other targets."""
    if not isinstance(row, dict) or set(row)-{'op','side','episode_id','expected_step','targets','gripper'}:
        raise InputRejected('Unexpected step fields')
    side = row.get('side')
    if (row.get('op') != 'step' or side not in ROBOT['arms'] or
            row.get('episode_id') != state['episode_id'] or
            type(row.get('expected_step')) is not int or row['expected_step'] != state['step']):
        raise InputRejected('Stale step or invalid side')
    targets = row.get('targets')
    if not isinstance(targets, dict) or set(targets) != set(ROBOT['arms'][side]):
        raise InputRejected('Require exactly the seven selected arm targets')
    result = dict(previous)
    result.update(targets)
    if 'gripper' in row:
        value = row['gripper']
        if type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise InputRejected('Gripper must be normalized 0..1')
        result[GRIPPERS[side]] = value*math.pi/4
    if set(result) != set(ORDER[:20]):
        raise RuntimeError('Retained command mapping mismatch')
    for name,value in result.items():
        if type(value) not in (int,float) or not math.isfinite(value) or not limits[name][0] <= value <= limits[name][1]:
            raise InputRejected('Native joint target outside limits')
    return result


def tcp_public(value):
    if not isinstance(value,dict) or set(value) != {'left','right'}:
        raise RuntimeError('Missing robot FK')
    for pose in value.values():
        if pose is not None and (not isinstance(pose,list) or len(pose)!=4 or
                any(not isinstance(row,list) or len(row)!=4 or
                    any(type(x) not in (float,int) or not math.isfinite(x) for x in row) for row in pose)):
            raise RuntimeError('Invalid robot FK')
    return deepcopy(value)


class CodingHost:
    def __init__(self, host, held_targets, output, *, sandbox_factory=Sandbox):
        self.host, self.session = host, host.session
        self.out = Path(output)
        self.previous = {n:float(held_targets[n]) for n in ORDER[:20]}
        if any(not host.driver.limits[n][0] <= v <= host.driver.limits[n][1] for n,v in self.previous.items()):
            raise ValueError('Invalid initial held targets')
        self.box = None
        self.closed = False
        self.faulted = False
        self.workspace = self.out/'policy_workspace'
        from .batch_storage import permits
        if self.workspace.exists() or not (self.workspace.resolve().is_relative_to(HERE.parent) or permits(self.workspace)):
            raise ValueError('Fresh task-owned workspace required')
        shutil.copytree(HERE/'starter', self.workspace)
        initial = public_robot(host.observe(robot_only=True))
        self.box = sandbox_factory(dict(observation=initial,held_targets=self.previous,control_hz=20),self.workspace)
        try:
            self.tcp = tcp_public(self.box.ready['tcp'])
        except BaseException:
            self.close()
            raise

    def observe(self):
        return dict(**public_observation(self.host.observe()),tcp=self.tcp,tcp_frame='base_link')

    def ready(self):
        return dict(op='ready', contract=deepcopy(self.host.contract),
                    policy_schema=SCHEMA,observation=self.observe())

    def _sync(self, *, raw=False):
        row = dict(op='sync',observation=public_robot(self.host.observe(robot_only=True)))
        if raw:
            row['held_targets']=dict(self.previous)
        self.box.send(row)
        result=self.box.receive(60)
        if result.get('op')!='result' or result.get('result')!={'synchronized':True}:
            raise RuntimeError('Sandbox synchronization failed')
        self.tcp=tcp_public(result['tcp'])

    def exchange(self, row):
        if row=={'op':'close'}:
            self.close()
            return dict(op='closed', result=self.session.result())
        if self.closed or self.faulted:
            raise RuntimeError('Coding bridge closed/faulted; never retry uncertain actions')
        from .contract import validate_tool
        validate_tool(row)
        try:
            return self._exchange(row)
        except InputRejected:
            raise
        except BaseException:
            self.faulted=True
            raise

    def _exchange(self,row):
        from .exploration import TOOLS, compile_actions, measured_motion
        if row['op'] in TOOLS:
            before = public_robot(self.host.observe(robot_only=True))
            try:
                actions = compile_actions(row, self.previous, self.host.previous_base, self.host.contract)
            except ValueError as error:
                raise InputRejected(str(error)) from error
            command = dict(op='execute_actions',request_id=str(uuid4()),
                           episode_id=before['episode_id'],expected_step=before['step'],actions=actions)
            reply = self._exchange(command)
            reply['motion'] = measured_motion(before,reply['observation'],row,len(actions),reply['actions_executed'])
            (self.out/f'exploration_tool_{uuid4().hex}.json').write_text(json.dumps(
                dict(request=row,command=command,motion=reply['motion']),allow_nan=False))
            return reply
        if row['op']=='execute_actions':
            reply=self.host.exchange(row)
            if reply.get('status')!='rejected' and reply['actions_executed']:
                last=row['actions'][reply['actions_executed']-1]
                self.previous=dict(zip(ORDER[:20],last[:20]))
            self._sync(raw=bool(reply.get('actions_executed')))
            reply['observation']=dict(**reply['observation'],tcp=self.tcp,tcp_frame='base_link')
            return reply
        self._sync()
        if row=={'op':'observe'}:
            return self._result({},0)
        if any(self.host.previous_base.values()):
            raise InputRejected('Stop the base with explicit deceleration rows before arm IK')
        self.box.send(row)
        remaining, actions, rejected, output_bytes = 60.,0,0,0
        while True:
            start=time.monotonic()
            reply=self.box.receive(remaining)
            remaining-=time.monotonic()-start
            output_bytes+=len(json.dumps(reply))
            if output_bytes>1024*1024:
                raise RuntimeError('Per-tool output limit')
            if reply.get('op')=='result':
                self.tcp=tcp_public(reply['tcp'])
                result=self._result(reply.get('result'),actions)
                (self.out/f'coding_tool_{uuid4().hex}.json').write_text(json.dumps(
                    dict(request=row,motion=result['motion'],actions_executed=actions),ensure_ascii=False,allow_nan=False))
                return result
            if reply.get('op')!='step':
                raise RuntimeError('Disallowed sandbox packet')
            if self.session.result()['status']!='running':
                raise RuntimeError('No action after episode termination')
            current=public_robot(self.host.observe(robot_only=True))
            try:
                if actions>=600:
                    raise InputRejected('Per-tool 600-step budget reached; earlier steps remain executed')
                targets=arm_targets(reply,current,self.previous,self.host.driver.limits)
                command=dict(op='execute_actions',request_id=str(uuid4()),episode_id=current['episode_id'],
                             expected_step=current['step'],actions=[[targets[n] for n in ORDER[:20]]+[0.,0.]])
                result=self.host.execute(command,robot_only=True)
            except InputRejected as error:
                rejected+=1
                if rejected>16:
                    raise RuntimeError('Repeated invalid action packet limit')
                self.box.send(dict(op='action_rejected',error=dict(status='rejected',error=str(error),
                    recoverable=True),observation=current))
                continue
            if result['actions_executed']!=1 or result['observation']['step']!=current['step']+1:
                raise RuntimeError('Native acknowledgement mismatch; no retries')
            self.previous=targets
            actions+=1
            with (self.out/'coding_actions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(request=command,step=result['observation']['step']))+'\n')
            observed=public_robot(result['observation'])
            self.box.send(dict(op='episode_finished',observation=observed) if result['episode_finished'] else observed)

    def _result(self,motion,actions):
        return dict(op='result',motion=motion,actions_executed=actions,observation=self.observe(),
                    episode_finished=self.session.result()['status']!='running')

    def close(self):
        if not self.closed:
            self.closed=True
            try:
                if self.box:
                    self.box.close()
            finally:
                self.host.close()
