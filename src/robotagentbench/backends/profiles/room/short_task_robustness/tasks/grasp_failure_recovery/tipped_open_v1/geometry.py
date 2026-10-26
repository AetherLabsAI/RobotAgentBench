"""Authored initial geometry only; no physics setters, solver or scored action."""
from copy import deepcopy
import math
import numpy as np
from ..common import audit_reset


def opened_initial(c,row):
    from ..frozen_initials_v2.inspect import initialization
    result,closure=initialization(c,row)
    # Existing task ready linkage is the existing open configuration. All arm,
    # body and head coordinates remain the exact selected source row.
    opening=deepcopy(closure['source_coordinates'])
    result['positions'].update(opening)
    for n in result['hold_targets']:result['hold_targets'][n]=result['positions'][n]
    audit_reset(c['stage'],result)
    return result,dict(hand=row['active_hand'],open_coordinates=opening,
        policy='existing_native_ready_open_linkage',arm_body_head_changed=False,
        physical_opening_animation_run=False)


def below_pose(original,cup,visual,delta,table,hand_center_xy):
    """Sideways and supported below the hand; retain existing yaw/support gap."""
    from scipy.spatial.transform import Rotation
    original,cup,visual,delta,table,xy=map(np.asarray,(original,cup,visual,delta,table,hand_center_xy))
    if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Finite active fingertip center required')
    axis=np.asarray(delta[0,:3],float);turn=Rotation.from_rotvec(axis*(math.pi/2)).as_matrix()
    origin=original[:3,3];moved=(cup-origin)@turn.T+origin
    visual_moved=(visual-origin)@turn.T+origin
    shift=np.zeros(3);shift[:2]=xy-(moved.min(0)+moved.max(0))[:2]/2
    # table is canonical; delta follows the existing row-vector table transfer.
    top=float((np.r_[table[1,:3],1.]@delta)[2])
    old_gap=float(cup[:,2].min()-top);shift[2]=top+old_gap-moved[:,2].min()
    proposed=original.copy();proposed[:3,:3]=turn@original[:3,:3];proposed[:3,3]+=shift
    return proposed,moved+shift,visual_moved+shift,dict(
        cup_side_pose_authored=True,physical_drop_replayed=False,original_yaw_preserved_before_fixed_side_rotation=True,
        fixed_side_rotation_rad=math.pi/2,support_gap_preserved_m=old_gap,
        center_below_fingertips_xy=np.allclose(((moved+shift).min(0)+(moved+shift).max(0))[:2]/2,xy,rtol=0,atol=1e-9))


def check(c,initial,hand,xyyaw,gate,protected):
    from ..procedural_grasp import Kinematics
    from ..frozen_initials_v1.cup import hand_geometry
    from ..expansion_v2.arm_first_visible_cup_v4 import inside_region
    from ..expansion_v2.camera_projection_complete import project_target
    from ..poses import place
    from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
    model=Kinematics(c['stage'],hand,initial['positions']);tips,frames=hand_geometry(c['stage'],model,initial['positions'])
    hand_points=np.concatenate([(local@frames[body].T)[:,:3] for _,body,local in gate.robot
        if body.startswith('/World/Robot/gripper_'+hand+'_')])
    # Geometric post-lift cue only, not proof of a physical successful grasp.
    if hand_points[:,2].min()<=c['cup'][:,2].max():return dict(accepted_cpu=False,reason='hand_not_above_original_upright_cup_top')
    scan=protected.scan(protected.model.fk(initial['positions'],model.base))
    if scan['overlap']:return dict(accepted_cpu=False,reason='protected_self_overlap')
    center=(tips.min(0)+tips.max(0))[:2]/2
    proposed,cup,visual,receipt=below_pose(c['original'],c['cup'],c['visual'],c['delta'],c['table'],center)
    if not inside_region(np.concatenate((cup,visual)),c['delta'],c['region']):return dict(accepted_cpu=False,reason='cup_outside_original_placement_region')
    if hand_points[:,2].min()<=max(cup[:,2].max(),visual[:,2].max()):return dict(accepted_cpu=False,reason='hand_not_above_side_cup')
    cup_box=np.array([cup.min(0),cup.max(0)])
    obstacle=next((p for p,b,t in gate.obstacles if not p.startswith('/World/Assets/CoffeeTable/') and
        gate.intersects(cup_box,b) and (t is None or gate.triangles_hit_box(t,cup_box))),None)
    if obstacle:return dict(accepted_cpu=False,reason='cup_other_scene_overlap:'+obstacle)
    place(c['stage'],c['target'],proposed)
    reason=gate.check(model,initial['positions'])
    projections=project_target(c['stage'],frames,c['target'])
    if reason is None and not any(p['whole_target_envelope_inside_original_frustum'] for p in projections.values()):reason='target_not_wholly_in_original_camera'
    if reason is None:reason=checker(c['stage'],c['manifest'],xyyaw)({'targets_rad':initial['positions']})
    result=dict(receipt,accepted_cpu=reason is None,reason=reason,proposed_target_world=proposed.tolist(),
        original_camera_projection=projections,hand_lowest_z_m=float(hand_points[:,2].min()),cup_highest_z_m=float(max(cup[:,2].max(),visual[:,2].max())),
        upright_source_cup_highest_z_m=float(c['cup'][:,2].max()),protected_self_collision_checked=True,
        target_on_table_geometry_checked=True,hand_open_geometry_checked=True,
        physical_settle_verified=False,physical_grasp_proven=False)
    if reason is not None:place(c['stage'],c['target'],c['original'])
    return result
