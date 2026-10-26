"""L2 Task108 in-hand initialization, isolated from the rejected pickup probes.

Source: playback_video3.py, SHA256 e7ee2186d74835737543e51b2927a200072b4c4270c3fa2669329a5e65793148,
lines 167-245 (plane solver), 4247-4340 (pre-seat), 4550-4698 (clamp/box).
This is a PORT, not a byte-identical replay of the old environment. Current
assets, native drives, shared material and dining station stay. The worker uses
the common physics profile; all source hold durations are preserved in seconds.
Legacy Y-up plate coordinates are converted by the initial-scene generator.
The original finite-pad checks and 3-contact command latch are retained. No IK,
table pickup, anchor snap, measured-aperture latch or closure-lead controller.
"""
import json
import math
from arena_api.physics_defaults import PHYSICS_DT

SOURCE_SHA256 = 'e7ee2186d74835737543e51b2927a200072b4c4270c3fa2669329a5e65793148'
MODE = 'l2_inhand_source_v1'
GRIP_WIDTH = .00875
PREOPEN = (GRIP_WIDTH + .04525) / .120


def solve_two_pad_seat(pad_anchors, inward_normals, nominal_center, *, radius):
    """L2 support-plane equations; keep the finite-patch residual check."""
    import numpy as np
    anchors = np.asarray(pad_anchors, dtype=float)
    normals = np.asarray(inward_normals, dtype=float)
    nominal = np.asarray(nominal_center, dtype=float)
    if (anchors.shape != (2, 3) or normals.shape != (2, 3)
            or nominal.shape != (3,) or not np.isfinite(anchors).all()
            or not np.isfinite(normals).all() or not np.isfinite(nominal).all()
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError('pad seat inputs must be finite with positive radius')
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError('pad seat normals must be non-degenerate')
    normals = normals / lengths[:, None]
    rhs = np.einsum('ij,ij->i', normals, anchors) + radius
    center = nominal + np.linalg.lstsq(normals, rhs - normals @ nominal, rcond=None)[0]
    residuals = normals @ center - rhs
    if not np.isfinite(center).all() or np.max(np.abs(residuals)) > 1e-8:
        raise RuntimeError('live pad support planes have no common nonpenetrating seat')
    points = center[None, :] - radius * normals
    delta = points - anchors
    tangent = delta - np.einsum('ij,ij->i', delta, normals)[:, None] * normals
    distances = np.linalg.norm(tangent, axis=1)
    if not np.isfinite(distances).all():
        raise RuntimeError('live pad support solution is non-finite')
    return dict(center=center.tolist(), patch_distances_m=distances.tolist(),
                shift_norm_m=float(np.linalg.norm(center - nominal)),
                plane_residuals_m=residuals.tolist())


class SourceClamp:
    """Original L2 command latch: do NOT substitute measured opening."""
    def __init__(self, angle):
        self.angle = angle
        self.streak = 0

    def observe(self, outer_n, inner_n, fresh):
        if not all(math.isfinite(f) and f >= 0 for f in (outer_n, inner_n)):
            raise ValueError('Invalid target-only force evidence')
        self.streak = self.streak + 1 if fresh and min(outer_n, inner_n) >= .05 else 0
        if not self.streak:
            self.angle = max(0., self.angle - .01)
        return self.streak >= 3


def live_frame(stage):
    import numpy as np
    from pxr import Gf, UsdGeom
    anchors, normals = [], []
    for side, sign in (('outer', 1), ('inner', -1)):
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(
            f'/World/Robot/gripper_r_{side}_link4')).ComputeLocalToWorldTransform(0)
        anchors.append(np.asarray(matrix.Transform(Gf.Vec3d(.0381, sign*.0222, 0))))
        normal = np.asarray(matrix.TransformDir(Gf.Vec3d(0, sign, 0))).copy()
        normals.append(normal / np.linalg.norm(normal))
    anchors, normals = np.asarray(anchors), np.asarray(normals)
    axis = normals[0] - normals[1]
    axis /= np.linalg.norm(axis)
    delta = anchors[1] - anchors[0]
    if delta @ axis < 0: axis = -axis
    return anchors, normals, float(delta @ axis)


def run_setup(*, stage, plan, step, reset, read_plate_matrix, capture, output, **unused):
    import numpy as np
    from pxr import Gf, UsdGeom
    from evaluation.tools.prepare_dining_place_initial import HAND, pose
    if (plan.get('grasp_setup') != MODE or plan.get('grasp_calibration') != 'legacy'
            or plan.get('initial_wrist_ik') is not None or plan.get('table_support')):
        raise ValueError('L2 in-hand mode must not use IK or table pickup')
    receipt = dict(schema='arena.dining_place.l2_inhand_port.v1', status='running',
        source_sha256=SOURCE_SHA256, phase='warmup', pose_writes={'plate':0,'payload':0},
        physics_hz=1/PHYSICS_DT, source_physics_hz=240, original_grip_width_m=GRIP_WIDTH,
        assets_and_drive_parameters_changed=False, controller='L2 three-contact command latch',
        adaptations=['current assets and native drives/material', 'Y-up to Z-up plate coordinates',
                     'dining station/lift height', 'common physics profile; hold durations preserved'],
        preseat_history=[])
    def save():
        (output/'preclamp_report.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    def tick(angle):
        row = step(angle,receipt['phase'])
        if row.get('pad_self_force_n',0) >= .05:
            raise RuntimeError('Actual pad-to-pad contact; not a plate clamp')
        return row
    try:
        angle = PREOPEN
        for _ in range(plan['robot_warmup_steps']): tick(angle)
        receipt['phase']='preseat'
        best = None
        for search in range(121):
            previous, streak = None, 0
            for _ in range(240):
                tick(angle)
                anchors,normals,gap = live_frame(stage)
                streak = streak+1 if previous is not None and abs(gap-previous)<=.0002 else 0
                previous = gap
                if streak >= 5: break
            else:
                raise RuntimeError('L2 pre-seat: live pad gap did not settle')
            candidate = dict(angle_rad=angle,gap_m=gap,normal_dot=float(normals[0]@normals[1]))
            try:
                candidate.update(solve_two_pad_seat(anchors,normals,anchors.mean(axis=0),radius=GRIP_WIDTH/2))
                radius = max(candidate['patch_distances_m'])
                if best is None or radius < max(best['patch_distances_m']): best = candidate
                safe = radius <= .0065+1e-9 and candidate['shift_norm_m'] <= .15 and candidate['normal_dot'] <= -.05
            except (RuntimeError,ValueError) as exc:
                candidate['error']=str(exc);safe=False
            receipt['preseat_history'].append(candidate);receipt['best_preseat']=best;save()
            if safe: break
            next_angle=max(0.,angle-.02)
            if next_angle >= angle-1e-12:
                raise RuntimeError('L2 finite-pad pre-seat failed before inserting plate')
            angle=next_angle
        else:
            raise RuntimeError('L2 finite-pad pre-seat search exhausted')
        receipt['phase']='seat_plate'
        hand=UsdGeom.Xformable(stage.GetPrimAtPath(HAND)).ComputeLocalToWorldTransform(0)
        relative=Gf.Matrix4d(*[v for r in plan['plate_to_hand'] for v in r])
        plate=relative*hand
        receipt['plate_reset']=pose(plate)
        reset('plate',plate);receipt['pose_writes']['plate']=1
        receipt['phase']='clamp'
        controller=SourceClamp(angle)
        for _ in range(160):
            row=tick(controller.angle)
            pads=row['pads']
            done=controller.observe(pads['outer']['force_n'],pads['inner']['force_n'],
                all(p['fresh'] for p in pads.values()))
            receipt['last_clamp_sample']=dict(step=row['step'],pads=pads,plate=row['bodies']['plate'],
                next_command_angle_rad=controller.angle,bilateral_streak=controller.streak)
            if done: break
        else:
            raise RuntimeError('L2 physical clamp: sustained bilateral contact not observed')
        angle=controller.angle
        receipt['clamp']=dict(angle_rad=angle,confirmation_steps=controller.streak)
        # Original L2 waits 150/240 s without adding a velocity gate or changing
        # the command; final independent validation remains responsible for hold.
        receipt['phase']='plate_hold'
        for _ in range(75): row=tick(angle)
        capture('plate_preclamped')
        receipt['phase']='seat_payload'
        box_relative=Gf.Matrix4d(*[v for r in plan['payload_to_plate'] for v in r])
        box=box_relative*read_plate_matrix()
        receipt['payload_reset']=pose(box)
        reset('payload',box);receipt['pose_writes']['payload']=1
        receipt['phase']='payload_settle'
        for _ in range(60): tick(angle)
        capture('payload_seated')
        receipt.update(status='preparation_complete_pending_independent_validation',phase='complete',hold_angle_rad=angle)
        save();return angle
    except Exception as exc:
        receipt.update(status='failed',reason=str(exc));save()
        try: capture('preclamp_failed')
        except Exception as image_error:
            receipt['capture_error']=str(image_error);save()
        raise
