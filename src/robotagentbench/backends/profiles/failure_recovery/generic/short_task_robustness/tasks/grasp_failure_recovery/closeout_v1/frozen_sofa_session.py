"""Bind original sofa readers to the already prepared recovery scene.

The ordinary session builder samples a new target placement. That preparation
belongs to ordinary starts, not a frozen failure state. This task-local binder
uses the same resolved plan, native factory, contacts, interaction/navigation
readers and ShortTaskSession; it performs no scene writes or action assistance.
"""
from copy import deepcopy
import numpy as np


def build(request, stage, manifest, output, *, episode_id, sensor):
    from short_tasks import registry
    from short_tasks.scene import verify_manifest, verify_scene_sources
    from short_tasks.runtime import ShortTaskSession
    from openend.tasks.all_objects_integration.workspace_physics import from_scene
    from openend.tasks.meal_tray.gripper_wake_v1.host import bindings_for, UnifiedGripperRoutes
    from openend.tasks.meal_tray.task_contact_routing import roots_for
    from short_tasks.tasks.sofa_to_drawer.native_geometry import shape_points

    if request['task_name'] != 'sofa_to_coffee_table':
        raise ValueError('Frozen sofa binder is not a generic scene-hook bypass')
    verify_manifest(manifest);verify_scene_sources()
    plan=registry.resolve(request)
    if not plan['available'] or plan['asset'] != 'remote_tv':
        raise ValueError('Original available TV remote task required')
    plan=registry.bind(plan,manifest)
    plan['source_receipt']=registry.verify_sources(plan)
    from .sofa_whole_table import plan_metadata,port_class
    plan_metadata(plan)
    runtime_manifest=deepcopy(manifest)
    rows=[row for row in runtime_manifest['objects'] if row.get('task_id')=='sofa_to_coffee_table']
    if len(rows)!=1 or rows[0]['id']!='remote_tv':
        raise ValueError('Exactly one original TV remote identity required')
    points=np.vstack([v for _,v,collision,visible in shape_points(stage,rows[0]['path']) if collision or visible])
    rows[0]['visible_bounds_world_m']=[points.min(0).tolist(),points.max(0).tolist()]
    runtime_manifest['failure_recovery_initialization_owner']='prepared_frozen_sofa_no_resample.v1'
    selector=from_scene(stage,runtime_manifest,sensor)
    output.mkdir(parents=True,exist_ok=False)
    ports={}
    try:
        for declaration in plan['runtime_components']:
            identity=declaration['instance_id']
            if identity!='sofa_to_coffee_table' or declaration['type']!='remote_transfer':
                raise ValueError('Only the existing sofa remote component is permitted')
            cls=port_class()
            ports[identity]=cls(stage,runtime_manifest,output/identity,episode_id,identity,declaration['parameters'])
            ports[identity].workspace_rates=True
        bindings={d['instance_id']:bindings_for(d['type'],ports[d['instance_id']],{},roots_for)
            for d in plan['runtime_components']}
        routes=UnifiedGripperRoutes(stage,bindings)
        interaction=navigation=None
        if plan.get('interaction_policy'):
            from short_tasks.tasks.interaction_progress_v1.native import NativeProgress
            interaction=NativeProgress(stage,plan,runtime_manifest,ports,episode_id,
                output/'interaction_progress.json',routes.native.fingers)
        if plan.get('navigation_policy'):
            from short_tasks.tasks.navigation_progress_v1.native import NativeNavigation
            navigation=NativeNavigation(stage,plan,ports,selector)
        if plan.get('environment_actor_contract'):
            raise ValueError('No actor allowed for frozen sofa recovery')
        return ShortTaskSession(plan,runtime_manifest,ports,episode_id=episode_id,
            workspace_selector=selector,contact_routes=routes,
            interaction_reader=interaction,navigation_reader=navigation)
    except BaseException:
        for port in ports.values():
            try:port.close()
            except Exception:pass
        raise


def measured_clearance(stage, initial, joints):
    """Use current native USD readback and measured joints, not frozen target XYZ."""
    from ..procedural_grasp import Kinematics,collider_points
    from ..frozen_initials_v1.cup import hand_geometry
    from ..expansion_v2.empty_grasp import envelope_distance
    model=Kinematics(stage,initial['active_hand'],joints)
    tips,_=hand_geometry(stage,model,joints)
    points=np.concatenate([r[3] for r in collider_points(stage,initial['target'])])
    return clearance(points,tips,model.base[:3,:3])


def clearance(points,tips,basis):
    from ..expansion_v2.empty_grasp import envelope_distance
    distance=float(envelope_distance(points,tips,basis))
    gaps={str(s):float((tips@(basis[:,1]*s)).min()-(points@(basis[:,1]*s)).max()) for s in (-1,1)}
    return dict(accepted=.02<=distance<=.05 and any(.02<=v<=.05 for v in gaps.values()),
        conservative_3d_envelope_distance_m=distance,signed_lateral_gaps_m=gaps,
        target_bounds_world_m=[points.min(0).tolist(),points.max(0).tolist()],
        actual_native_readback=True,required_gap_m=[.02,.05])
