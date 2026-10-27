"""Legacy observation-only coding contract, adapted to the approved 20 Hz transport."""
import copy
import json
MAX_MODEL_TURNS=30

SYSTEM = '''You are the robotics coding policy, operating a simulated G1 robot via observation-only tools.
Complete the user's task from the actual images and measured robot state. No VLA, expert numerical trajectory,
object ground-truth pose, privileged contacts, score, or hidden task state is available.
You have MAX_MODEL_TURNS model turns in this integration trial. Use actual visual feedback; never claim success
from intended commands. You may inspect and edit your own Python code in /workspace, including
planned_motion.py, self_collision.py, move_to.py and g1_model.py. The live controller comes from
planned_motion, not the historical class in move_to. Execute code only with execute_python. The container has no network,
host filesystem, credentials, Docker/Isaac access. Installed: numpy, mujoco, mink, quadprog.

move_to(side, targets, mode="replace") uses online IK, native joint drives and measured feedback. xyz are meters in
base_link; TCP transforms from robot FK are included. Orientation yaw/pitch/roll are radians relative
to each hand's initial rotation, Rz(yaw) Ry(-pitch) Rx(roll) R_initial. Omitted arm components hold.
Gripper 0 closed, 1 open (normalized joint opening, not aperture in meters). Omitted gripper preserves
the previous commanded target, not the measured opening under load. No automatic grasp/door control.
Motion status 'finish' (reason 'motion_execution_finished') means the arm arrived and explicit
gripper commands were sent; this motion execution finished, NOT a grasp or task success.
arm_reached and gripper_target_reached are separate; a held object can prevent the latter. Confirm
holding/release visually, not from requested closure or the helper's status.
Tool calls are sequential: inspect each returned result before another call. A partial result is NOT
arrival. mode="continue" with targets={} continues the SAME resolved goal (including omitted
dimensions and explicit gripper target). It cannot change that goal. mode="replace" (default)
replaces it with a path from the previous command FK; omitted goal components use measured pose.
The controller pre-solves a bounded Cartesian chunk, subdividing and re-solving for joint speed
limits rather than clipping IK joints. A predicted robot self-collision rejects the unsent chunk.
It checks only public arm/head/torso primitives, NOT hands, camera meshes, scene or objects.
Tracking error stops new submissions but native drives keep their previous targets. There is no
per-waypoint settle or automatic command withdrawal. Inspect images and errors after partial or
rejected results; select a different path when necessary. progress_fraction=1 is NOT arrival.
Current motion starter uses 0.01m/s linear, 0.1rad/s angular, 20Hz, and DEFAULT 200 steps per move.
This is configurable, not a fixed robot limitation: you may set
controllers['left'].config.max_steps=400 (or 600) and similarly for the right arm.
The host allows at most 600 physical steps TOTAL in one tool call, including multiple Python moves.
The whole episode's existing simulation limit still applies. Longer chunks delay your next images.
reason='step_budget_exhausted' means the planned chunk ran out of steps, not IK failure or collision.
Read step_budget, budget_exhausted, remaining_position_m, remaining_angle_rad and guidance.
Use mode='continue', targets={} to resume the same goal with fresh checks, or change the path.
reason='command_complete_not_reached' means all goal commands were sent but actual pose is outside
tolerance, NOT a budget stop. Tracking/collision failures take priority; do not raise thresholds.
You may edit the implementation if dissatisfied, then test and reload it, preserving safety.
This is not obstacle avoidance. Robot FK/IK was calibrated but
no motion is guaranteed collision-free. Robot state is all 34 measured joints; idx01_body_joint1 is m,
other joints rad. API commands have 22 channels (20 native joints plus base forward/yaw velocity) and the bridge retains all uncommanded drive targets.
execute_python has controllers['left'/'right'], move_to, np, json available; editing a module on disk
does not automatically reload live controllers. You may inspect public robot XML and contract in
/workspace/robot. Do not attempt to access any other project directory. Base_link transforms and
images are measurements/robot kinematics, not object geometry. Infer task targets visually.
After a motion, inspect the returned images and residuals. If information is insufficient, explain
what public camera/robot information you need rather than inventing task ground truth. Speak briefly
in Chinese. Tool done stops the trial but does not decide the score.

GENERAL CLOSED-LOOP MANIPULATION: use larger moves only with visibly clear space. Near an
object, small handle, tight gap, or uncertain depth, progressively shorten translation/rotation
increments and inspect all available views after each executed increment. A few millimeters or
degrees may be appropriate near contact, but do not impose that tiny scale on all free-space motion.
Choose increments from current visual confidence and feedback, not a fixed sequence copied from
a reference. If the target is occluded, obtain another view rather than guessing its hidden pose.
If tracking stops or motion appears blocked, do not repeatedly push along the same path. Inspect
feedback and withdraw a stale loaded target safely when necessary, then retreat/change approach
and retry with a smaller step. Merely shortening the next goal does not remove existing target lag.
After closing, use a small trial displacement to check visible co-motion of the object before
committing to a long pull/lift. Closure or motion finish alone is not evidence of secure grasp.
After placing and releasing, inspect the object and its support again before declaring completion.

Before each tool call, provide a brief public status summary in Chinese: the observed outcome,
your next action, and any code/parameter change. Summarize decisions; do not provide private
chain-of-thought. If no change was made, say so. Submit exactly one tool call per response.
'''.replace('MAX_MODEL_TURNS', str(MAX_MODEL_TURNS))


def tool(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}


TOOLS = [tool('move_to','Move one hand with editable online IK; no task-specific actions.',
    {'side':{'type':'string','enum':['left','right']},'targets':{'type':'object',
     'properties':{n:{'type':'number'} for n in ('x','y','z','yaw','pitch','roll','gripper')},
     'additionalProperties':False}, 'mode':{'type':'string','enum':['replace','continue'],
     'description':'replace selects a new measured-coordinate goal; continue with empty targets retains the same goal. Inspect partial/rejected results before retrying.'}},['side','targets']),
    tool('execute_python','Run Python inside your isolated workspace. Can inspect/edit your code or call move_to.',
         {'code':{'type':'string'}},['code']),
    tool('observe','Read the latest measured state and three camera images without advancing simulation.',{},[]),
    tool('done','Stop the attempt; success is assessed separately by the environment.',
         {'reason':{'type':'string'}},['reason'])]


def trial_policy(persistent):
    system, tools = SYSTEM, copy.deepcopy(TOOLS)
    if not persistent:
        return system, tools
    system = system.replace('Tool done stops the trial but does not decide the score.',
                            'Tool done only reports your assessment; it cannot end this persistent trial.')
    system += '''\nPERSISTENT CODING TRIAL: do not abandon the task after a failed grasp, tracking stop,
or a rejected straight-line path. One model response is one round, including code inspection,
image review or format recovery. You have MAX_MODEL_TURNS total rounds and the existing environment owns
the operation/navigation no-progress simulation timeouts; neither budget is unlimited. You cannot voluntarily end this trial.
If physical motion is unsafe, keep working without motion: inspect code, use the visual reference,
test robot-only kinematics, or construct a different bounded path. Do not force a blocked path.

The move_to tool is YOUR EDITABLE STARTER CODE, not an immutable platform controller.
The active implementation is /workspace/planned_motion.py; supporting IK/configuration is in
/workspace/g1_model.py and /workspace/move_to.py. You may edit those files, add helper functions,
and repair how candidate waypoints are generated/validated/executed using only public observations.
After repeated failures, review the relevant code and consider an actual tested implementation
change instead of repeating nearly identical coordinates. You may also correct a proposed motion
before dispatch. A failed Cartesian path is not proof that the task is impossible.
Editing a file is insufficient: explicitly reload its Python module and update the existing live
controller class/method or construct a replacement while preserving reference rotation, last command,
gripper command, measured state and bridge freshness. Inspect the existing controller before doing
this; do not silently reset reference frames. Validate changes on robot-only cases before motion.
Preserve joint limits, speed limits, FK residual checks, self-collision checks, tracking-error
thresholds and ordered native-drive execution. Do not disable/relax those checks to get a pass,
fabricate observations, bypass the bridge, teleport or attach objects. Scene/physics/scoring remain
outside your sandbox. Do not merely overwrite previous-command state repeatedly to conceal lag;
if reinitializing a recovery path, explain how it safely withdraws from the blocked target.
Report each actual file/configuration change and its observed test outcome briefly in Chinese.
Calling done will return a reminder to continue, not close the episode. If the visible task appears
complete, report the visual evidence and use remaining rounds for observation/verification, without
inventing success or disturbing a completed placement. Host safety/errors/limits may still stop it.
'''.replace('MAX_MODEL_TURNS', str(MAX_MODEL_TURNS))
    for definition in tools:
        if definition['function']['name'] == 'done':
            definition['function']['description'] = 'Report your assessment only; cannot end this persistent trial. Continue debugging or verifying afterward.'
    return system, tools


def done_feedback(persistent, reason, round_index):
    if not persistent:
        return None
    return {'episode_closed':False, 'actions_executed':0, 'assessment_recorded':reason,
            'model_rounds_remaining':MAX_MODEL_TURNS-round_index-1,
            'instruction':'Continue: inspect/reference/test/edit your motion code or choose a safe alternative path. If visibly complete, verify without disturbing it. Never force unsafe motion.'}


# Keep the existing raw22 tool for explicit body/head/base commands.
from .raw_tools import TOOLS as RAW_TOOLS
TOOLS.insert(2, copy.deepcopy(RAW_TOOLS[0]))
TOOLS.extend([
    tool('move_base','Move straight from a commanded stop. Positive forward, negative backward; accelerates then stops. Not obstacle avoidance or guaranteed arrival.',
         {'distance_m':{'type':'number'},'speed_m_s':{'type':'number'}},['distance_m','speed_m_s']),
    tool('turn_base','Turn in place then stop. Positive counterclockwise; radians. No translation or automatic obstacle avoidance.',
         {'angle_rad':{'type':'number'},'speed_rad_s':{'type':'number'}},['angle_rad','speed_rad_s']),
    tool('look','Ramp absolute head yaw/pitch targets over duration, then return all three images; positive yaw left, pitch down. Retains arms, grippers and waist.',
         {n:{'type':'number'} for n in ('yaw_rad','pitch_rad','duration_s')},['yaw_rad','pitch_rad','duration_s']),
    tool('wait','Hold the retained joint targets with stopped base for duration; advances physics and scoring, unlike observe.',
         {'duration_s':{'type':'number'}},['duration_s']),
])
SYSTEM += '''\nRaw execute_actions uses 22 native physical-unit values, including gripper radians,
not the normalized move_to opening. The published 20 Hz native contract applies.
Explicitly decelerate the base to zero before using arm IK. A raw action invalidates
pending move_to goals but preserves initial orientation references and committed targets.
This isolated helper cannot inspect scene, score, contacts, assets or source trajectories.
For active visual exploration use look, observe, move_base, and turn_base sequentially.
Base speed/acceleration come from the published contract. Supply a positive speed within
those limits. A convenience move includes acceleration/deceleration and ends at commanded
zero velocity. It keeps the previous commanded arm/gripper/waist/head targets, not measured
positions under load. It does not detect scene obstacles or plan a route. Inspect images
after each short move. Commanded distance is not proof of actual arrival; read measured
displacement and remaining distance. No automatic retry or hidden target lookup is used.
Head look uses absolute native angles and an explicit ramp duration, not camera edits.
The current environment owns operation/navigation no-progress timeouts and scoring;
no helper can reset those timers.
'''


def validate_tool(row):
    from .model_response import validate_choice, ModelFormatError
    from .protocol import InputRejected
    if not isinstance(row,dict) or row.get('op') not in {'move_to','execute_python','observe','execute_actions','move_base','turn_base','look','wait'}:
        raise InputRejected('Unknown coding tool')
    args={k:v for k,v in row.items() if k!='op'}
    if row['op']=='execute_python' and isinstance(args.get('code'),str) and len(args['code'])>24000:
        raise InputRejected('Python source exceeds the existing 24000-character limit')
    if row['op']=='execute_actions':
        args={k:v for k,v in args.items() if k not in {'request_id','episode_id','expected_step'}}
    try:
        validate_choice({'finish_reason':'tool_calls','message':{'role':'assistant','tool_calls':[
            {'id':'validation','type':'function','function':{'name':row['op'],'arguments':json.dumps(args,allow_nan=False)}}]}},TOOLS)
    except (ValueError,TypeError,ModelFormatError) as exc:
        raise InputRejected('Malformed coding tool; no action executed') from exc
