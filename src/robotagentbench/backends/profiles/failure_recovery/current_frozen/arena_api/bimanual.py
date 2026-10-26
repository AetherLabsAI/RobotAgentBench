"""Named robot joints shared by the black-box contract and physical controller."""

LEFT_ARM = tuple(f"idx{20 + i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM = tuple(f"idx{60 + i}_arm_r_joint{i}" for i in range(1, 8))
GRIPPERS = ("idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1")
JOINTS = LEFT_ARM + RIGHT_ARM + GRIPPERS
CONTROL_HZ = 20
CONTROL_DT = 1 / CONTROL_HZ
