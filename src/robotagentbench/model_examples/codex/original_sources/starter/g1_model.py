"""Public robot-only binding. No asset, episode, task, scene or score access."""
import json
from pathlib import Path
import numpy as np
import mujoco
from move_to import MinkIK


class G1ArmIK(MinkIK):
    """Full named state input; explicit body/head/arm-only IK coordinates.

    Finger internal joints do not affect the asset-defined centre-link TCP.
    They stay present and unchanged in observations; no fake values are filled.
    This class does not pack, drop or execute any API command channels.
    """
    def __init__(self, side, *, directory=None, **kwargs):
        directory = Path(directory) if directory is not None else Path(__file__).parent/'robot'
        contract = json.loads((directory/'robot_contract.json').read_text())
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        model = mujoco.MjModel.from_xml_path(str(directory/contract['arm_ik_model']))
        short = {'left':'l', 'right':'r'}[side]
        super().__init__(model, contract['tcp'][short]['site'], contract['arms'][side], **kwargs)
        if set(self.names) != set(contract['arm_ik_coordinates']):
            raise ValueError('Robot model coordinate mapping mismatch')
        self.state_names = {j['name'] for j in contract['state_joints']}
        if (len(self.state_names) != 34 or len(contract['command_order']) != 20 or
                set(self.names) | set(contract['arm_ik_excluded_coordinates']) != self.state_names):
            raise ValueError('Incomplete public robot contract')
        self.contract = contract

    def read_q(self, observation):
        if set(observation.joints) != self.state_names:
            raise ValueError('Require all 34 named measured joints; no truncation or zero filling')
        if not all(np.isfinite(v) for v in observation.joints.values()):
            raise ValueError('Nonfinite robot observation')
        return super().read_q(observation)
