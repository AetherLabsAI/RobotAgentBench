"""Original geometry gates, with a roundoff-stable exact-centerline tie.

The legal operating stations look directly at the target. Matrix inversion
can put an exactly centered target 1e-16 m on either side. A forward-error
bound, not a geometric margin, distinguishes that tie from a physical side.
"""
import numpy as np
from ..frozen_initials_v3.geometry import Gate as OriginalGate


def stable_side(base, center, selected_hand):
    inv = np.linalg.inv(base)
    terms = inv[1] * np.r_[center, 1.]
    value = float(terms.sum())
    # Standard dot-product roundoff bound plus inverse/multiply rounding.
    eps = np.finfo(float).eps
    error = float((8 * eps / (1 - 8 * eps)) * np.abs(terms).sum())
    if abs(value) <= error:
        return selected_hand, dict(kind="floating_point_centerline_tie", local_y_m=value,
            arithmetic_error_bound_m=error, hand=selected_hand, geometric_margin_m=0.)
    return ("l" if value > 0 else "r"), dict(kind="strict_target_side", local_y_m=value)


class Gate(OriginalGate):
    def check(self, initial, hand, *, base=None, full=True):
        from ..frozen_initials_v1.cup import hand_geometry
        from ..expansion_v2.empty_grasp import envelope_distance
        from ..expansion_v2.camera_projection_complete import project_target
        from ..poses import world
        from short_tasks.tasks.robot_pose_alignment_v1.geometry import checker
        from short_tasks.agent_probe_v1.native_owner import initialize_base_pose
        if self.variant != "sofa":
            raise ValueError("Centerline tie applies only to look-at sofa stations")
        base = self.base if base is None else base
        basis = base[:3, :3]
        model = self.model(hand, initial["positions"], base)
        tips, frames = hand_geometry(self.stage, model, initial["positions"])
        center = (self.points.min(0) + self.points.max(0)) / 2
        side, witness = stable_side(base, center, hand)
        distance = envelope_distance(self.points, tips, basis)
        gaps = {str(s): float((tips @ (basis[:, 1] * s)).min() -
            (self.points @ (basis[:, 1] * s)).max()) for s in (-1, 1)}
        signs = [s for s in (-1, 1) if .02 <= gaps[str(s)] <= .05]
        result = dict(accepted_cpu=False, reason=None,
            conservative_3d_envelope_distance_m=distance, signed_lateral_gaps_m=gaps,
            supported_miss_signs=signs, active_hand=hand, side_selection=witness)
        if side != hand:
            result["reason"] = "target_side_hand_mismatch"
        elif not signs or not .02 <= distance <= .05:
            result["reason"] = "fixed_target_gap_not_2_5cm"
        if result["reason"] is not None:
            return result
        scan = self.protected.scan(self.protected.model.fk(initial["positions"], base))
        if scan["overlap"]:
            result.update(reason="protected_self_overlap",
                protected_contacts=[c["bodies"] for c in scan.get("contacts", [])])
            return result
        if not full:
            result["accepted_cpu"] = True
            return result
        projections = project_target(self.stage, frames, self.target)
        result["original_camera_projection"] = projections
        if not any(x["whole_target_envelope_inside_original_frustum"] for x in projections.values()):
            result["reason"] = "target_not_wholly_in_original_camera"
            return result
        xyyaw = [float(base[0, 3]), float(base[1, 3]), float(np.arctan2(base[1, 0], base[0, 0]))]
        initialize_base_pose(self.stage, xyyaw)
        reason = self.room.check(model, initial["positions"])
        if reason is None:
            reason = checker(self.stage, self.manifest, xyyaw)({"targets_rad": initial["positions"]})
        result.update(reason=reason, accepted_cpu=reason is None)
        if not np.array_equal(world(self.stage, self.target), self.original_target):
            raise ValueError("Original target moved")
        return result
