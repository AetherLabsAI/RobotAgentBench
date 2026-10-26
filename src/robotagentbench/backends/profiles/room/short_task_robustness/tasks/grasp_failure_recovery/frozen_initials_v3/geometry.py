"""Read-only FK/geometry gates for exact training arm/body configurations."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ..common import HERE,digest,physics,audit_reset
from .inspect import SOURCES,reset


def compose(variant,out,seed,*,coffee_context=None):
    """Keep the original object sampler; the caller supplies only a frozen arm."""
    if variant=='sofa':raise ValueError('Sofa requires an explicit existing full-room base key')
    config_path=HERE/'procedural_config.json';config=json.loads(config_path.read_text());config['config_path']=str(config_path)
    if variant=='shelf_drink':
        from ..procedural_initial import compose_shelf
        return compose_shelf(out,seed,config)
    # The existing coffee atomic composer has its own pinned source environment.
    # Consume its task-owned scene data here; never mix executable source roots.
    if coffee_context is None:raise ValueError('Pinned original atomic composition context required')
    from pxr import Usd
    import os
    from ..common import owned,verify_sources
    source=owned(coffee_context)/str(seed)/'initial.json';record=json.loads(source.read_text())
    if record['variant']!='coffee_bag' or record['seed']!=seed:raise ValueError('Coffee context identity mismatch')
    verify_sources(record['source_layers']);verify_sources(record['code_sources'])
    if digest(record['scene'])!=record['scene_sha256']:raise ValueError('Original sampled coffee scene changed')
    out=owned(out,fresh=True);out.mkdir(parents=True)
    stage=Usd.Stage.CreateNew(str(out/'initial.usda'));stage.GetRootLayer().subLayerPaths=[os.path.relpath(record['scene'],out)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    return stage,deepcopy(record),physics(stage)


def existing(variant):
    from pxr import Usd
    source=HERE/SOURCES[variant];record=json.loads(source.read_text())
    stage=Usd.Stage.Open(record['scene']);stage.SetEditTarget(stage.GetSessionLayer())
    return stage,record,source


class Gate:
    def __init__(self,stage,record):
        from ..poses import world
        from ..procedural_grasp import collider_points
        from short_tasks.agent_probe_v1.collision_guard.stage_model import from_stage
        from short_tasks.agent_probe_v1.collision_guard.reactive_gate import ReactiveGate
        self.stage,self.record=stage,record;self.variant=record['variant'];self.target=record['target']
        self.points=np.concatenate([r[3] for r in collider_points(stage,self.target)])
        self.base=world(stage,'/World/Robot/base_link');self.original_target=world(stage,self.target).copy()
        self.protected=ReactiveGate(from_stage(stage));self.models={};self.clearances={}
        self.manifest=self.room=None;self.bases=None
        if self.variant=='sofa':
            from ..expansion_v2.operating_station import RoomGate,proposals
            launch=record['original_launch'];self.manifest=json.loads(Path(launch['manifest']).read_text())
            self.room=RoomGate(stage,self.manifest)
            self.bases,self.station_domain=proposals(stage,self.manifest,'sofa_to_coffee_table',launch)
        elif self.variant=='shelf_drink':self.obstacles=record['neighbors']
        else:
            from evaluation.tasks.coffee_machine.specs import MACHINE,CUP,COUNTER
            self.obstacles=[MACHINE,CUP,COUNTER]

    def model(self,hand,q,base):
        from ..procedural_grasp import Kinematics
        if hand not in self.models:self.models[hand]=Kinematics(self.stage,hand,q)
        model=self.models[hand];model.base=base.copy();return model

    def check(self,initial,hand,*,base=None,full=True):
        from ..frozen_initials_v1.cup import hand_geometry
        from ..expansion_v2.empty_grasp import envelope_distance
        from ..expansion_v2.camera_projection_complete import project_target
        from ..shelf_checkpoint import Clearance
        from ..poses import world
        base=self.base if base is None else base;basis=base[:3,:3]
        model=self.model(hand,initial['positions'],base);tips,frames=hand_geometry(self.stage,model,initial['positions'])
        center=(self.points.min(0)+self.points.max(0))/2;target_local=np.linalg.inv(base)@np.r_[center,1.]
        side='l' if target_local[1]>0 else 'r';distance=envelope_distance(self.points,tips,basis)
        gaps={str(s):float((tips@(basis[:,1]*s)).min()-(self.points@(basis[:,1]*s)).max()) for s in (-1,1)}
        signs=[s for s in (-1,1) if .02<=gaps[str(s)]<=.05]
        result=dict(accepted_cpu=False,reason=None,conservative_3d_envelope_distance_m=distance,signed_lateral_gaps_m=gaps,
            supported_miss_signs=signs,active_hand=hand)
        if side!=hand:result['reason']='target_side_hand_mismatch'
        elif not signs or not .02<=distance<=.05:result['reason']='fixed_target_gap_not_2_5cm'
        if result['reason'] is not None:return result
        scan=self.protected.scan(self.protected.model.fk(initial['positions'],base))
        if scan['overlap']:result.update(reason='protected_self_overlap',protected_contacts=[c['bodies'] for c in scan.get('contacts',[])]);return result
        if not full:result['accepted_cpu']=True;return result
        projections=project_target(self.stage,frames,self.target);result['original_camera_projection']=projections
        if not any(x['whole_target_envelope_inside_original_frustum'] for x in projections.values()):
            result['reason']='target_not_wholly_in_original_camera';return result
        if self.variant=='sofa':
            from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
            from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
            from ..frozen_initials_v1.cup import base_matrix
            xyyaw=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))]
            initialize_base_pose(self.stage,xyyaw)
            reason=self.room.check(model,initial['positions'])
            if reason is None:reason=checker(self.stage,self.manifest,xyyaw)({'targets_rad':initial['positions']})
        else:
            if hand not in self.clearances:self.clearances[hand]=Clearance(self.stage,self.target,self.obstacles,hand)
            # Clearance owns an FK base snapshot of this unchanged sampled scene.
            result['clearance']=self.clearances[hand].evaluate(initial['positions'])
            reason='original_collider_clearance_rejected' if result['clearance']['overlaps'] else None
        result.update(reason=reason,accepted_cpu=reason is None)
        if not np.array_equal(world(self.stage,self.target),self.original_target):raise ValueError('Original target moved')
        return result

    def source_initial(self,row):return reset(self.stage,self.record['robot_initialization'],row)

    def sofa_station(self,initial,hand):
        """Finite original station grid only, no IK or guessed new workspace."""
        from ..frozen_initials_v1.cup import base_matrix,hand_geometry
        from ..expansion_v2.empty_grasp import envelope_distance
        model=self.model(hand,initial['positions'],self.base)
        tips,_=hand_geometry(self.stage,model,initial['positions'])
        vertical=max(0.,float(tips[:,2].min()-self.points[:,2].max()),float(self.points[:,2].min()-tips[:,2].max()))
        if vertical>.05:return None,dict(accepted_cpu=False,reason='training_tip_height_outside_confirmed_gap')
        local=(np.c_[tips,np.ones(len(tips))]@np.linalg.inv(self.base).T)[:,:3]
        for row in self.bases:
            base=base_matrix(row['initial_base_xyyaw']);basis=base[:3,:3]
            moved=(np.c_[local,np.ones(len(local))]@base.T)[:,:3]
            center=(self.points.min(0)+self.points.max(0))/2;side=(np.linalg.inv(base)@np.r_[center,1.])[1]
            if ('l' if side>0 else 'r')!=hand:continue
            gaps=[float((moved@(basis[:,1]*s)).min()-(self.points@(basis[:,1]*s)).max()) for s in (-1,1)]
            if not any(.02<=g<=.05 for g in gaps) or not .02<=envelope_distance(self.points,moved,basis)<=.05:continue
            result=self.check(initial,hand,base=base)
            if result['accepted_cpu']:return row,result
        return None,dict(accepted_cpu=False,reason='original_operating_station_grid_exhausted')
