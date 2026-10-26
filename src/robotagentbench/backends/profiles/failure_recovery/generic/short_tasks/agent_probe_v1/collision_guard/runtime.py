"""Production wiring; current contact only, lightweight aggregate diagnostics."""
from collections import Counter
import json
import time
import numpy as np
from .reactive_gate import ReactiveGate
from .collision_model import native_pose
from .stage_model import from_stage
from .arm_filter_scope import apply_session_filters

class NativeCollisionGuard:
    def __init__(self,stage,audit_path=None):
        started=time.perf_counter()
        self.gate=ReactiveGate(from_stage(stage))
        # Read original filters first; do not accidentally remove software
        # protection for the pairs whose physical response is disabled.
        self.physics_filters=apply_session_filters(stage,'same_and_cross_arm')
        self.ready=False; self.calls=0; self.overlap_steps=0
        self.max_depth=0.; self.wall_total=0.; self.wall_max=0.
        self.held=Counter(); self.ranges={}; self.raw_ranges={}
        self.audit=None if audit_path is None else audit_path.open('w',buffering=1)
        self.build_s=time.perf_counter()-started
    @staticmethod
    def links(robot):
        return {'/World/Robot/'+name:native_pose(t) for name,t in
                zip(robot.shared_metatype.link_names,robot.get_link_transforms()[0])}
    def bind(self,robot,names):
        links=self.links(robot); q=dict(zip(names,map(float,robot.get_dof_positions()[0])))
        fk=self.gate.model.fk(q,links['/World/Robot/base_link'])
        self.fk_error=max(float(np.max(np.abs(fk[n]-links[n]))) for n in fk)
        if self.fk_error>2e-6: raise RuntimeError('Collision guard live FK mismatch')
        initial=self.gate.scan(links)
        if initial['overlap']: raise RuntimeError('Initial robot has protected self-overlap')
        self.ready=True
    def filter(self,robot,names,desired,sim_time):
        started=time.perf_counter()
        q=dict(zip(names,map(float,robot.get_dof_positions()[0])))
        applied,detail=self.gate.filter_command(q,desired,self.links(robot))
        self.calls+=1; self.held.update(detail['held_joints'])
        self.overlap_steps+=int(detail['state']['overlap'])
        for c in detail['state']['contacts']: self.max_depth=max(self.max_depth,c['penetration_depth_m'])
        for n in desired:
            lo,hi=self.ranges.get(n,(q[n],q[n])); self.ranges[n]=[min(lo,q[n]),max(hi,q[n])]
            lo,hi=self.raw_ranges.get(n,(desired[n],desired[n])); self.raw_ranges[n]=[min(lo,desired[n]),max(hi,desired[n])]
        elapsed=time.perf_counter()-started
        self.wall_total+=elapsed; self.wall_max=max(self.wall_max,elapsed)
        if self.audit is not None:
            self.audit.write(json.dumps(dict(time=sim_time,read_and_filter_wall_s=elapsed,**detail))+'\n')
        return applied
    def summary(self):
        return dict(version='current_overlap_fixed_hold_v2',ready=self.ready,build_s=self.build_s,
            pair_count=len(self.gate.model.pairs),scope_exemptions=self.gate.scope_exemptions,
            approximations=self.gate.model.approximations,physics_filters=self.physics_filters,
            initial_fk_max_error=getattr(self,'fk_error',None),physics_step_checks=self.calls,
            held_substep_counts=dict(self.held),overlap_steps=self.overlap_steps,
            max_geometry_penetration_m=self.max_depth,measured_joint_ranges=self.ranges,
            received_joint_ranges=self.raw_ranges,
            read_and_filter_mean_ms=1000*self.wall_total/self.calls if self.calls else None,
            read_and_filter_max_ms=1000*self.wall_max,
            zero_penetration_guaranteed=False,includes_environment_collision_guard=False)
    def close(self):
        if self.audit is not None: self.audit.close()
