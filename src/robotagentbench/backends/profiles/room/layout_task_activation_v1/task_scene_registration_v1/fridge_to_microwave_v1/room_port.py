"""One persistent scene-owned box, two original scorers, selected by room host."""
import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from .composite import Composite


def normalize_parameters(p):
    if set(p)!={'payload_variant'} or p['payload_variant'] not in ('beef_box','leftover_box'):
        raise ValueError('Explicit payload_variant beef_box/leftover_box required')
    return p['payload_variant']


class FridgeMicrowavePort:
    asset='refrigerator'
    physics_hz=60

    def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
        from pxr import Usd,UsdPhysics,PhysxSchema
        from .scene import Scene,ROBOT,PLATE_BODY,BODY
        self.scene=Scene(stage,manifest,normalize_parameters(parameters))
        self.episode=episode_id;self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.binding=SimpleNamespace(objects=dict(target=self.scene.target,source=self.scene.source_paths,
                                                  destination=BODY,plate=PLATE_BODY))
        robot=stage.GetPrimAtPath(ROBOT)
        if not robot:raise ValueError('Missing actual room robot')
        if not stage.GetSessionLayer().anonymous:raise ValueError('Anonymous sensor session required')
        paths=[self.scene.target,PLATE_BODY,BODY]+[str(p.GetPath()) for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for path in paths:
            p=stage.GetPrimAtPath(path)
            if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Missing reporter body '+path)
            a=p.GetAttribute('physxContactReport:threshold')
            if a and a.HasAuthoredValueOpinion() and a.Get()!=0:raise ValueError('Conflicting contact sensor '+path)
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for path in paths:PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        self.composite=Composite(episode_id,self.scene.target,self.scene.variant)
        self.index=0;self.score=self.composite.summary();self.source=None;self.enabled=False;self.buffer=[]
        self.trace=(self.output/'selected_facts.jsonl').open('x')

    @property
    def ended(self):return self.composite.success or self.composite.closed
    @property
    def requested_physics_hz(self):return self.composite.requested_physics_hz
    def activation_geometry(self):return [self.scene.activation[self.composite.phase]]

    def initialize(self,clock,view):
        from .native_source import NativeSource
        self.clock=clock;self.source=NativeSource(self.scene,clock,view)

    def _summary(self):
        s=self.composite.summary();s.update(native_reader_connected=self.source is not None,room_registered=True,
            evidence_source='native_physx',full_trajectory_verified=False)
        self.score=s;return s

    def accept_initial(self):
        baseline=self.source.frame(self.episode,0,[])
        self.composite.observe(baseline);self._summary()
        # No right-hand, closed-door, fixed-pose, or out-of-region gates.
        receipt=dict(initial_state_gate_enabled=False,status='not_checked_user_disabled',
                     native_reader_connected=True,measured_baseline=asdict(baseline))
        (self.output/'baseline.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt

    def begin_selected_control(self,enabled):
        if self.buffer:raise RuntimeError('Unconsumed selected evidence')
        self.enabled=bool(enabled)
        if not enabled:self.composite.pause()

    def after_physics(self,helper_report):
        self.source.after_physics(helper_report,self.composite.phase)
        if self.enabled:
            if self.clock.hz!=self.requested_physics_hz:raise RuntimeError('Wrong phase clock')
            self.buffer.append(self.source.latest)

    def observe(self):
        if not self.enabled or len(self.buffer)!=self.requested_physics_hz//20:raise RuntimeError('Missing dense selected interval')
        frame=self.source.frame(self.episode,self.index+1,self.buffer)
        self.composite.observe(frame);self.index+=1;self._summary()
        self.trace.write(json.dumps(dict(facts=asdict(frame),details=self.source.latest['details'],score=self.score))+'\n');self.trace.flush()
        self.discard_control();return self.score

    def discard_control(self):self.buffer.clear();self.enabled=False
    def pause(self):self.composite.pause();self.discard_control()
    def close(self):
        if self.composite.last is not None:
            self.composite.finalize();self._summary()
            (self.output/'final_score.json').write_text(json.dumps(self.score,indent=2)+'\n')
        if self.source:self.source.close()
        self.trace.close()
