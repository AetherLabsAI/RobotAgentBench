"""One-time orientation extraction; not an episode initializer.

Only 3x3 tool rotations and provenance survive. No arm checkpoint, positions,
trajectory arrays, or source-data paths are needed by empty_grasp.build().
The extracted direction is a candidate until native/rendered validation.
"""
import argparse
import json
from pathlib import Path
from ..common import HERE, digest, owned, write_json
from .tipped_initial import load_base


def freeze(base_key, variant, output):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd
    from ..poses import place, world
    from ..procedural_grasp import complete_closed_reset, Kinematics
    task='sofa_to_coffee_table' if variant=='sofa' else 'clear_table_trash'
    _,launch,manifest,bound,ready=load_base(base_key,task)
    stage=Usd.Stage.Open(manifest['scene'])
    # All changes here are anonymous, read-only kinematic calculations.
    stage.SetEditTarget(stage.GetSessionLayer())
    from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
    initialize_base_pose(stage,launch['initial_base_xyyaw'])
    issued=dict(positions=ready['targets_rad'],hold_targets=ready['targets_rad'])
    rotations={}; evidence=[]
    def capture(hand, coordinates, provenance):
        if hand in rotations:return
        initial,_=complete_closed_reset(stage,issued,hand)
        q=dict(initial['positions']);q.update(coordinates)
        model=Kinematics(stage,hand,initial['positions'])
        # Orientation only. Translation and arm coordinates are NOT frozen.
        tool=model.frames(q)[model.tool]
        R=model.base[:3,:3].T@tool[:3,:3]
        if not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-7):
            raise ValueError('Non-rigid source tool orientation')
        rotations[hand]=R.tolist()
        evidence.append(dict(hand=hand,**provenance))
    if variant=='sofa':
        from evaluation.tasks.find_object import specs
        from ..random_find import attempted_commands
        for label in specs.episodes('4.13'):
            path=specs.EPISODES/'4.13'/label/'producer_targets.json'
            for row in attempted_commands(stage,path):
                capture(row['hand'],row['initialization']['positions'],dict(
                    source_api='4.13',episode=label,file_sha256=digest(path),
                    frame=row['frame'],selection='first_legal_source_closure_orientation_per_hand'))
            if len(rotations)==2:break
    elif variant=='table_waste_empty_grasp':
        # Existing 18-4 cup held-reset configuration is already portable. Use
        # its actual right-hand orientation ONLY, not its arm reset or cup pose.
        # This is a named source, not a missing-archive fallback during reset.
        from evaluation.tasks.trash_disposal import accepted_initial
        path=accepted_initial.TASK/'accepted_held_resets_v1.json'
        profiles=json.loads(path.read_text())
        profile=profiles['profiles']['cup']
        capture('r',profile['robot_initialization']['positions'],dict(
            source_disposal_api='18-4',source_subtask='7.10',
            source_configuration='accepted_held_resets_v1.json:profiles.cup',
            acceptance_scope=profiles['acceptance'],file_sha256=digest(path),
            selection='existing_approved_right_cup_held_reset_orientation_only'))
    else:raise ValueError('Explicit new empty-grasp variant required')
    if not rotations:raise ValueError('No original closure direction available; no invented orientation')
    out=owned(output,fresh=True)
    if not out.is_relative_to(HERE/'expansion_v2/configurations'):
        raise ValueError('Task-owned configuration output required')
    out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schema='failure_recovery.fixed_grasp_direction.v1',variant=variant,
        fixed_tool_rotation_base=rotations,orientation_jitter=False,
        provenance=evidence,source_data_needed_during_episode_generation=False,
        arm_positions_frozen=False,source_checkpoint_replayed=False,
        native_grasp_orientation_accepted=False,
        code_sha256=digest(__file__),hand_coverage=sorted(rotations))
    write_json(out,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-key',required=True)
    p.add_argument('--variant',choices=('sofa','table_waste_empty_grasp'),required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.base_key,a.variant,a.output)))


if __name__=='__main__':main()
