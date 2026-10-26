"""Task-local native facts bridge. All runtime geometry is read-only.

The owner supplies actual floor geometry and one native pad/hand-pose snapshot
at the same substep. There is no command argument or privileged RPC input.
"""
import math
import numpy as np
from approved_rules import Binding,Facts
from native_reader import NativeSample
from folding import fold_measure
from placement import projected_coverage,MaterialReleasedStability
from grasp import PatchRetention
from contact_geometry import material_between_jaws


class ClothFactsReader:
    def __init__(self,*,binding,native_reader,source_solid,destination_solid,
                 table_z,table_polygon_xy,basket_profile,pad_geometry,policy,floor_contact,
                 initial_source_crease=None):
        if not isinstance(binding,Binding)or not policy['contact_evidence']['user_confirmed']:raise ValueError('Approved bound native reader required')
        if policy['coverage_denominator']!='rest_material_area':raise ValueError('Unapproved coverage convention')
        if (source_solid.path,destination_solid.path)!=(binding.source_path,binding.destination_path):raise ValueError('Physical role identity mismatch')
        if native_reader.episode_id!=binding.episode_id or native_reader.body_path!=binding.target_path:raise ValueError('Reader belongs to another physical object')
        if set(pad_geometry)!={f'/World/Robot/gripper_{h}_{s}_link4'for h in('l','r')for s in('inner','outer')}:
            raise ValueError('All original physical finger pads must be bound')
        if not callable(floor_contact):raise ValueError('Actual physical floor binding required')
        self.binding,self.reader,self.mesh=binding,native_reader,native_reader.mesh
        self.source,self.destination=source_solid,destination_solid
        self.table_z,self.table_polygon=table_z,table_polygon_xy
        self.basket,self.pads,self.policy,self.floor_contact=basket_profile,pad_geometry,policy,floor_contact
        self.crease=initial_source_crease;self.origin=self.initial_q=self.last=self.cached=None
        detection=policy['reused_source_detection'];self.shell=detection['support_face_tolerance_m']
        self.retain={h:PatchRetention(self.mesh,window_s=detection['grasp_window_s'],
            position_tolerance_m=detection['relative_position_tolerance_m'],normal_tolerance_deg=detection['relative_angle_tolerance_deg'])for h in('l','r')}
        stability=policy['cloth_stability']
        if not stability['user_confirmed']:raise ValueError('Approved cloth stability definition required')
        self.stability=MaterialReleasedStability(self.mesh,position_span_m=stability['position_span_m'],area_fraction_min=stability['material_area_fraction_min'])

    def start(self,*,sample,body_poses):
        if self.origin is not None:raise RuntimeError('Cannot reset a live cloth reader')
        self.origin=sample.sample_index;self.initial_q=np.array(sample.positions,copy=True)
        return self.observe(sample=sample,body_poses=body_poses)

    def observe(self,*,sample,body_poses):
        if self.origin is None:raise RuntimeError('Owner must accept native initialization before scoring')
        if not isinstance(sample,NativeSample)or sample.episode_id!=self.binding.episode_id or sample.body_path!=self.binding.target_path or sample.topology_id!=self.mesh.identity:
            raise ValueError('Cross-episode or wrong native cloth sample')
        if self.last is not None and sample.sample_index==self.last:return self.cached
        index=sample.sample_index-self.origin
        if index!=(0 if self.last is None else self.last-self.origin+1):raise ValueError('Missing native scoring substep')
        # Native SDK timestamps are float32. Derive exact elapsed time from the
        # measured native step index, not wall time or action request counts.
        time_s=index/60
        q=self.mesh.positions(sample.positions);contacts={}
        required={*self.pads,*(f'/World/Robot/gripper_{h}_base_link'for h in('l','r'))}
        if set(body_poses)!=required:raise ValueError('One complete native pose snapshot required')
        for path,pad in self.pads.items():contacts[path]=pad.measure(q,body_poses[path],surface_shell_m=self.shell)['near_surface']
        finger_contact=bool(any(mask.any()for mask in contacts.values()))
        source_nodes=self.source.node_proximity(q,shell_m=self.shell)
        dest_nodes=self.destination.node_proximity(q,shell_m=self.shell)
        floor=np.asarray(self.floor_contact(q))
        if floor.dtype!=bool or floor.shape!=(len(q),):raise ValueError('Measured native floor-contact mask required')
        held=False;lift=False
        for hand in ('l','r'):
            inner=f'/World/Robot/gripper_{hand}_inner_link4';outer=f'/World/Robot/gripper_{hand}_outer_link4'
            hand_pose=body_poses[f'/World/Robot/gripper_{hand}_base_link']
            mask=contacts[inner]&contacts[outer]
            ids=np.flatnonzero(mask)
            if len(ids):mask[ids]&=material_between_jaws(q[ids],self.pads[inner],self.pads[outer],body_poses[inner],body_poses[outer],hand_pose)
            row=self.retain[hand].observe(time_s=time_s,points=q,
                hand_transform_xyzw=hand_pose,bilateral_node_mask=mask)
            held|=row['held']
            if row['held']:
                ids=np.asarray(row['patch_nodes'],int)
                lift|=bool(not source_nodes[ids].any()and np.all(q[ids,2]>self.initial_q[ids,2]+self.shell)and not floor.any())
        fold_kwargs=dict(overlap_fraction_min=self.policy['fold']['material_overlap_fraction_min'],
            layer_distance_max_m=self.policy['fold']['layer_distance_max_m'],normal_deviation_max_deg=self.policy['fold']['opposing_normal_deviation_max_deg'])
        folded=fold_measure(self.mesh,q,**fold_kwargs)['folded']
        coverage=projected_coverage(self.mesh,q,table_z=self.table_z,table_polygon_xy=self.table_polygon,
            support_tolerance_m=self.shell,denominator_area=self.mesh.area)
        unfolded=bool(not folded and self.policy['reused_source_unfolded_projection_ratio'][0]<=coverage['unfolded_projection_ratio']<=self.policy['reused_source_unfolded_projection_ratio'][1])
        if index==0 and self.binding.task_name=='spread_tablecloth':
            # Source basket initialization is an off-centre, rim-draped fold.
            # It is not awarded collection credit under the new midline rule.
            if self.crease is None:raise ValueError('Exact source folded-initialization crease required')
            initial_fold=fold_measure(self.mesh,q,creases=[(1,self.crease)],**fold_kwargs)['candidates'][0]
            folded=bool(initial_fold['panels_opposed']and min(initial_fold['side_normal_coherence_fractions'])>=self.policy['fold']['material_overlap_fraction_min'])
            if not folded:raise ValueError('Native clean initial state has lost its source fold')
        stable=self.stability.observe(q,not held and not finger_contact)
        self.cached=Facts(binding=self.binding,sample_index=index,time_s=time_s,physics_dt_s=1/60,complete=True,
            finger_contact=finger_contact,held=bool(held),grasp_lift_verified=bool(lift),
            whole_clear_source=self.source.clear_of_entire_surface(self.mesh,q,shell_m=self.shell),
            whole_inside_destination=self.basket.contains(self.mesh,q)if self.binding.task_name=='collect_tablecloth'else False,
            destination_support_verified=bool(dest_nodes.any()),source_supported=bool(source_nodes.any()),
            folded_at_least_once_verified=bool(folded),unfolded_verified=unfolded,coverage=coverage['coverage'],
            stable_verified=bool(stable),on_floor=bool(floor.any()))
        self.last=sample.sample_index;return self.cached

    def close(self):self.reader.close()
