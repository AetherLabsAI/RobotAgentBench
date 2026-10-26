import json
import numpy as np

from real2sim.production.check_task8_clear import score


def episode(path):
    n=80
    b=np.zeros((n,2,2,3))
    initial=np.array([[[.1,.1,.45],[.16,.16,.55]],[[.25,.1,.45],[.31,.16,.55]]])
    b[:20]=initial
    b[20:40]=initial+[0,0,.1]
    b[40:]=initial+[0,.5,0]
    task='8.2'
    targets=['/World/Assets/Task8Clear/Item0','/World/Assets/Task8Clear/Item1']
    state={'task_id':np.array(task),'frame_index':np.arange(n),'fps':np.array(20.),'target_prims':np.array(targets),'target_bounds':b,'robot_joint_position_rad_or_m':np.zeros((n,34)),'command_arm_rad':np.zeros((n,14)),'command_waist_rad_m':np.zeros((n,2)),'command_head_rad':np.zeros((n,2)),'command_grip_rad':np.full((n,2),.7843981634)}
    phases=np.full(n,'settle',dtype='U32')
    for i in range(2):
        start=40+i*8
        phases[start:start+4]=f'{i}:release'
        phases[start+4:start+8]=f'{i}:retreat'
    state['phase']=phases
    action={'arm':state['command_arm_rad'],'waist':state['command_waist_rad_m'],'head':state['command_head_rad'],'lgrip':np.zeros(n),'rgrip':np.zeros(n)}
    np.savez(path/'full_state.npz',**state);np.savez(path/'action.npz',**action)
    report={'task_id':task,'failure':None,'placed':[{'target':t} for t in targets],'targets':targets,'robot_camera_parameters_unchanged':True,'runtime_target_pose_writes':False,'contact_error':None,'source_revision':{'git_commit':'example-source-commit'},'cloth_bounds':[[0,0,.45],[.5,.5,.45]],'table_bounds':[[-1,-1,0],[1,1,.45]],'initial_bounds':initial.tolist()}
    (path/'result.json').write_text(json.dumps(report));(path/'contacts.json').write_text(json.dumps([{'frame':25,'actor0':t,'actor1':'/World/Robot/gripper_l_outer_link4','impulse_norm_sum_ns':.01,'minimum_separation_m':-.0001} for t in targets]))
    return state,action


def test_both_objects_must_be_lifted_and_stably_cleared(tmp_path):
    state,_=episode(tmp_path)
    assert score('8.2',tmp_path)['ok']
    state['target_bounds'][40:,1,:,1]-=.5
    np.savez(tmp_path/'full_state.npz',**state)
    result=score('8.2',tmp_path)
    assert not result['ok']
    assert '1:left_of_cloth' in result['failed_gates']


def test_final_pose_alone_does_not_prove_pick(tmp_path):
    state,_=episode(tmp_path)
    state['target_bounds'][20:40,:,:,2]-=.1
    np.savez(tmp_path/'full_state.npz',**state)
    assert '0:lifted' in score('8.2',tmp_path)['failed_gates']


def test_knocking_another_target_off_table_cannot_be_recovered_into_success(tmp_path):
    state,_=episode(tmp_path)
    state['target_bounds'][25,1,:,2]-=.5
    np.savez(tmp_path/'full_state.npz',**state)
    assert '1:never_dropped_below_table' in score('8.2',tmp_path)['failed_gates']


def test_mismatched_commands_fail_and_fingertip_contact_is_not_gated(tmp_path):
    _,action=episode(tmp_path)
    action['arm'][25,0]=.2
    np.savez(tmp_path/'action.npz',**action)
    (tmp_path/'contacts.json').write_text('[]')
    result=score('8.2',tmp_path)
    assert 'aligned_action_state' in result['failed_gates']
    assert not any(g.endswith(':left_hand_contact') for g in result['gates'])


def test_task_identity_and_parameters_are_not_inferred_from_filename(tmp_path):
    episode(tmp_path)
    assert 'identity' in score('8.4',tmp_path)['failed_gates']
    report=json.loads((tmp_path/'result.json').read_text());report['robot_camera_parameters_unchanged']=False
    (tmp_path/'result.json').write_text(json.dumps(report))
    assert 'unaltered_robot_camera' in score('8.2',tmp_path)['failed_gates']


def test_severe_interhand_contact_is_not_normal_pad_closure(tmp_path):
    episode(tmp_path)
    path=tmp_path/'contacts.json'
    contacts=json.loads(path.read_text())
    contacts.append({'actor0':'/World/Robot/gripper_l_outer_link4','actor1':'/World/Robot/gripper_r_outer_link4','minimum_separation_m':-.008})
    path.write_text(json.dumps(contacts))
    assert 'no_severe_robot_self_collision' in score('8.2',tmp_path)['failed_gates']


def test_table_contact_does_not_prove_release_from_gripper(tmp_path):
    episode(tmp_path)
    path=tmp_path/'contacts.json';contacts=json.loads(path.read_text())
    contacts.append({'frame':79,'actor0':'/World/Assets/Task8Clear/Item0','actor1':'/World/Robot/gripper_l_outer_link4','impulse_norm_sum_ns':.02})
    path.write_text(json.dumps(contacts))
    assert '0:released_from_robot' in score('8.2',tmp_path)['failed_gates']


def test_delayed_release_then_drop_during_retreat_fails(tmp_path):
    state,_=episode(tmp_path)
    state['target_bounds'][52:56,1,:,2]+=.16
    np.savez(tmp_path/'full_state.npz',**state)
    result=score('8.2',tmp_path)
    assert result['gates']['1:supported']
    assert result['gates']['1:released_from_robot']
    assert '1:released_before_retreat' in result['failed_gates']


def test_release_timing_requires_phase_evidence(tmp_path):
    state,_=episode(tmp_path)
    del state['phase']
    np.savez(tmp_path/'full_state.npz',**state)
    assert 'phase_evidence' in score('8.2',tmp_path)['failed_gates']


def test_book_stack_requires_physical_contact_with_supported_book(tmp_path):
    state,_=episode(tmp_path)
    state['task_id']=np.array('8.4')
    state['target_bounds'][40:,1]=state['target_bounds'][40:,0]+[0,0,.1]
    np.savez(tmp_path/'full_state.npz',**state)
    path=tmp_path/'result.json';report=json.loads(path.read_text());report['task_id']='8.4';path.write_text(json.dumps(report))
    assert '1:supported' in score('8.4',tmp_path)['failed_gates']
    path=tmp_path/'contacts.json';events=json.loads(path.read_text())
    events.append({'frame':45,'actor0':report['targets'][0],'actor1':report['targets'][1],'impulse_norm_sum_ns':.1})
    path.write_text(json.dumps(events))
    assert score('8.4',tmp_path)['ok']


def test_generated_floor_support_does_not_exempt_arm_or_furniture_collisions():
    from real2sim.production.task8_environment import is_base_support_contact
    floor='/World/HouseTopologyCandidate/Shell/Floors/Floor_002'
    hit={'kind':'world','a':'/World/Robot/base_link/collisions/mesh_1/box',
         'b':floor,'depth_m':.0014}
    assert is_base_support_contact(hit,{floor})
    assert not is_base_support_contact({**hit,'a':'/World/Robot/arm_l_link4/collisions/mesh'}, {floor})
    assert not is_base_support_contact({**hit,'b':'/World/Assets/TV'}, {floor})
    assert not is_base_support_contact({**hit,'depth_m':.01}, {floor})
    assert not is_base_support_contact({**hit,'kind':'self'}, {floor})
