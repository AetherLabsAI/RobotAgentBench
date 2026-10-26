"""Explicit diagnostic reset for the migrated L2 motion, not the default grasp.

Current assets/PD and the complete native grasp checkpoint remain unchanged.
Arms use source frame zero. Carry the plate-to-hand and payload-to-plate frames
with the current finger checkpoint, not the incompatible historical plate frame.
Must pass live initial acceptance; never inherit the historical success label.
"""
import copy


def source_snapshot(accepted):
    from .source_reference import load_reference,ARM_NAMES,initial_reference
    manifest,motion=load_reference()
    result=copy.deepcopy(accepted)
    for name,value in zip(ARM_NAMES,motion['arm'][0]):
        result['joint_reset_coordinates'][name]=float(value)
        result['joint_targets'][name]=dict(value=float(value),unit='rad')
    fixed=initial_reference(motion)['stationary_joint_targets']
    for name,record in fixed.items():
        if name=='idx01_body_joint1':continue  # existing .92 m dining lift stays
        result['joint_reset_coordinates'][name]=record['value']
        result['joint_targets'][name]=dict(record)
    # The plate and finger coordinates are one calibrated grasp. Changing only
    # the plate frame mixed two unrelated checkpoints (v1 failed 3/3 resets).
    # Deepcopy above intentionally preserves both relative object transforms.
    result['acceptance']='L2 reference diagnostic; requires live acceptance, not user-approved default or physics pass'
    result['reference_reset']=dict(mode='l2_reference',version='native_grasp_v2',
        source_uuid=manifest['baseline_uuid'],
        source_sha256=manifest['files']['candidate.npz']['sha256'],
        source_destination='CoffeeTable',test_destination='CounterTop_17',
        arm_targets='source frame zero unchanged',plate_frame='accepted native plate-to-hand unchanged',
        finger_state='current accepted checkpoint; not legacy gripper/asset restoration',
        no_ik=True,source_success_inherited=False)
    return result
