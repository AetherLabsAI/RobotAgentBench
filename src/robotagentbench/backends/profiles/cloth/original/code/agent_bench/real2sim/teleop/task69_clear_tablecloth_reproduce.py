"""Bounded robot-relative contact calibration for small Task 69.

The recorded left gripper approaches the photographed pan handle with its
lower fingertip link crossing the raised distal handle before the action-grip
transition.  The bounded correction moves only the left arm drive targets and
is retained through placement so the relocated object keeps the same
robot-relative motion as the accepted trajectory.  The pan, tablecloth, and
canonical CoffeeTable are never pose-written.
"""

TASK69_PAN_TARGET_PATH = "/World/Assets/Task69Items/GreenPan"
TASK69_MUG_TARGET_PATH = "/World/Assets/Task69Items/GinghamMug"
TASK69_BLUE_TARGET_PATH = "/World/Assets/Task69Items/BlueBull"

# The reviewed frame-zero pose moved the pan by (-144, -18) mm while the
# station moved by (-100, +15) mm relative to accepted trace v56.  Preserve
# the accepted early clearance plus the (-44, -33) mm residual translation.
# The small Z term corrects the finite-motion error measured before contact.
TASK69_PAN_CLEARANCE_OFFSET_XYZ_M = (-0.095506, -0.051526, -0.009)

TASK69_PAN_CLEARANCE_PLAN_FRAME = 80
TASK69_PAN_CLEARANCE_FULL_FRAME = 100
TASK69_PAN_CLEARANCE_RELEASE_FRAME = 270
TASK69_PAN_CLEARANCE_CLEAR_FRAME = 290
TASK69_PAN_ALIGN_OFFSET_XYZ_M = (-0.011, -0.011, -0.013)
TASK69_PAN_ALIGN_PLAN_FRAME = 140
TASK69_PAN_ALIGN_FULL_FRAME = 170
TASK69_PAN_ALIGN_RELEASE_FRAME = 270
TASK69_PAN_ALIGN_CLEAR_FRAME = 290
TASK69_PAN_DESCEND_OFFSET_XYZ_M = (0.027, 0.017, -0.018)
TASK69_PAN_DESCEND_PLAN_FRAME = 177
TASK69_PAN_DESCEND_FULL_FRAME = 188
TASK69_PAN_DESCEND_RELEASE_FRAME = 270
TASK69_PAN_DESCEND_CLEAR_FRAME = 290
TASK69_PAN_GRIP_END_FRAME = 267

TASK69_PAN_SURFACE_STOP_FORCE_N = 3.0
TASK69_PAN_SURFACE_STOP_FORCE_HOLD_N = 5.0
TASK69_PAN_SURFACE_STOP_PRELOAD_RAD = 0.005
TASK69_PAN_SURFACE_STOP_LOW_FORCE_STEP_RAD = 0.002
TASK69_PAN_SURFACE_STOP_ONE_SIDED_STEP_RAD = 0.003
TASK69_PAN_SURFACE_STOP_MAX_FOLLOW_RAD = 0.03

# +40 mm in the mug's local Y direction at its reviewed 82 degree yaw.
TASK69_MUG_CLEARANCE_OFFSET_XYZ_M = (
    -0.039611,
    0.005567,
    0.0,
)
TASK69_MUG_CLEARANCE_PLAN_FRAME = 360
TASK69_MUG_CLEARANCE_FULL_FRAME = 385
TASK69_MUG_CLEARANCE_RELEASE_FRAME = 452
TASK69_MUG_CLEARANCE_CLEAR_FRAME = 464
TASK69_MUG_GRIP_START_FRAME = 453
TASK69_MUG_GRIP_END_FRAME = 585
# The reviewed mug moved (-150, -18) mm while the robot station moved
# (-100, +15) mm.  Keep the residual robot-relative translation through the
# complete grasp and release; the original clearance arc above remains
# independent and still clears on frame 464.
TASK69_MUG_RELOCATION_OFFSET_XYZ_M = (-0.085, -0.093, -0.011)
TASK69_MUG_RELOCATION_PLAN_FRAME = 360
TASK69_MUG_RELOCATION_FULL_FRAME = 385
TASK69_MUG_RELOCATION_RELEASE_FRAME = 585
TASK69_MUG_RELOCATION_CLEAR_FRAME = 605

TASK69_MUG_SURFACE_STOP_FORCE_N = 2.0
TASK69_MUG_SURFACE_STOP_FORCE_HOLD_N = 14.0
TASK69_MUG_SURFACE_STOP_PRELOAD_RAD = 0.008
TASK69_MUG_SURFACE_STOP_LOW_FORCE_STEP_RAD = 0.001
TASK69_MUG_SURFACE_STOP_ONE_SIDED_STEP_RAD = 0.003
TASK69_MUG_SURFACE_STOP_MAX_FOLLOW_RAD = 0.025

# Executed action-grip windows for the photographed Blue Bull.  The first
# short right-hand close is a real regrasp attempt; the later overlapping
# left/right windows are the recorded two-hand transfer.  End frames are
# exclusive so an opening command is never overridden.
TASK69_BLUE_GRIP_PHASES_BY_HAND = {
    "l": ((876, 889), (900, 1120)),
    "r": ((742, 775), (793, 953), (975, 1063)),
}

TASK69_BLUE_SURFACE_STOP_FORCE_N = 3.0
TASK69_BLUE_SURFACE_STOP_FORCE_HOLD_N = 5.0
TASK69_BLUE_SURFACE_STOP_PRELOAD_RAD = 0.005
TASK69_BLUE_SURFACE_STOP_LOW_FORCE_STEP_RAD = 0.002
TASK69_BLUE_SURFACE_STOP_MISSING_CONTACT_STEP_RAD = 0.003
TASK69_BLUE_SURFACE_STOP_MAX_FOLLOW_RAD = 0.060

# The first left-hand can contact in the reviewed replay is one-sided because
# the reproduced left pad midpoint is 30 mm west and 13 mm north of the live
# can centre.  Move only the left arm command during the recorded handoff;
# the can and CoffeeTable remain ordinary unwritten rigid bodies.  Contact
# trace v21 also measured the left pad midpoint 95 mm above the right-hand
# stable barrel slice; lower the left arm to that same physical slice.
TASK69_BLUE_HANDOFF_OFFSET_XYZ_M = (0.030, -0.013, -0.095)
TASK69_BLUE_HANDOFF_PLAN_FRAME = 830
TASK69_BLUE_HANDOFF_FULL_FRAME = 860
# Once the upright correction has seated the can, lift the opening hand clear
# immediately; leaving the open wrist at table height lets the recorded
# lateral path sweep the light can.
TASK69_BLUE_HANDOFF_RELEASE_FRAME = 1049
TASK69_BLUE_HANDOFF_CLEAR_FRAME = 1062

# Once the left hand owns the can, use the live rigid-body up vector to apply
# one bounded robot-only orientation correction before set-down.  The can
# remains dynamic and friction-held throughout.
# Apply the accepted bounded wrist correction before the can first settles.
TASK69_BLUE_UPRIGHT_PLAN_FRAME = 970
TASK69_BLUE_UPRIGHT_FULL_FRAME = 995
TASK69_BLUE_UPRIGHT_RELEASE_FRAME = 1062
TASK69_BLUE_UPRIGHT_CLEAR_FRAME = 1075
# One-shot small-angle Cartesian correction bound.
TASK69_BLUE_UPRIGHT_MAX_CORRECTION_RAD = 0.10

# The executed right action stays closed briefly after the left hand has
# established its second bilateral clamp.  In simulation that over-constrains
# the rigid can between diverging arms.  Trace v22 shows both left pads stably
# loaded at frame 935 and an impulse building immediately afterwards.  Begin
# opening four frames earlier, while the left inner pad is already loaded, so
# the right fingers clear as the left bilateral clamp forms rather than
# dragging the can through the start of the left carry.
TASK69_BLUE_RIGHT_RELEASE_START_FRAME = 932
TASK69_BLUE_RIGHT_RELEASE_RAMP_FRAMES = 1

# The accepted left clamp used to jump from its contact-latched 0.248 rad
# target to fully open (0.784 rad) in one recorded frame.  On the current
# clean scene that impulse tipped the supported can before the retreat.  Keep
# the same release time but open over a short physical drive ramp.
TASK69_BLUE_LEFT_RELEASE_START_FRAME = 1085
TASK69_BLUE_LEFT_RELEASE_RAMP_FRAMES = 24

# Once the accepted trajectory has placed the can upright on the table, hold
# the left-arm support pose and slide the friction-held can 165 mm across the
# tabletop.  v7's measured final centre was 21.4 mm inside the cloth's +Y
# edge, so its 26.5 mm footprint required another 47.9 mm merely to touch the
# gate.  The extra 60 mm supplies a bounded 12.1 mm clearance margin.  Release
# only after its full footprint clears the cloth.
TASK69_BLUE_PLACE_OFFSET_XYZ_M = (0.0, 0.165, 0.0)
# Lift the friction-held can clear of the cloth support before beginning the
# horizontal clear motion.  Separating lift and translation prevents the
# cloth's 3 mm support edge from acting as a tip-over fulcrum.
TASK69_BLUE_PLACE_RECENTER_OFFSET_XYZ_M = (0.0, 0.0, 0.015)
TASK69_BLUE_PLACE_RECENTER_PLAN_FRAME = 1038
TASK69_BLUE_PLACE_RECENTER_FULL_FRAME = 1044
TASK69_BLUE_PLACE_PLAN_FRAME = 1049
TASK69_BLUE_PLACE_FULL_FRAME = 1065
# Once the horizontal carry is stationary and clear of the cloth, lower the
# live clamp onto the canonical tabletop before opening.  The measured
# elevated root at frame 1065 was about 9 mm above its supported height.
TASK69_BLUE_PLACE_LOWER_OFFSET_XYZ_M = (0.0, 0.0, 0.0)
TASK69_BLUE_PLACE_LOWER_PLAN_FRAME = 1070
TASK69_BLUE_PLACE_LOWER_FULL_FRAME = 1074
# Correct the live supported can orientation after lowering, rather than
# carrying forward the earlier in-air wrist correction while the can slowly
# rolls in the clamp.
TASK69_BLUE_PLACE_UPRIGHT_PLAN_FRAME = 1075
TASK69_BLUE_PLACE_UPRIGHT_FULL_FRAME = 1080
TASK69_BLUE_PLACE_UPRIGHT_MAX_CORRECTION_RAD = 0.001
TASK69_BLUE_PLACE_RELEASE_FRAME = 1085
TASK69_BLUE_PLACE_CLEAR_FRAME = 1140
TASK69_BLUE_RELEASE_LINEAR_DAMPING = 0.30
TASK69_BLUE_RELEASE_ANGULAR_DAMPING = 1.00

# After opening at the successful supported-slide pose, lift the left hand
# vertically and hold it clear for the remainder of the recorded episode.
TASK69_BLUE_RETREAT_OFFSET_XYZ_M = (0.0, 0.0, 0.120)
TASK69_BLUE_RETREAT_PLAN_FRAME = 1110
TASK69_BLUE_RETREAT_FULL_FRAME = 1130
