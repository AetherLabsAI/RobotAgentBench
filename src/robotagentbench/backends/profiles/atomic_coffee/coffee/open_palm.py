"""Task-local open pushing pose, not the previously fitted bottle-grip bake.

No source asset edits, unit/scale changes or physics changes. Four fingers
retain the delivered rest pose; the original thumb rig is splayed just enough
to clear the measured palmar plane. The source grip fitter identifies palmar
skin on local +Z; forbid dorsal-face fitting. Skin/colliders share posed bones.
"""
import json
import os
from pathlib import Path
import numpy as np
from short_task_robustness.tasks.stock_wooden_tray_interference.human_asset import HumanAsset, HERE as SOURCE


class OpenPalmHand(HumanAsset):
    palmar_normal_local = np.array([0., 0., 1.])
    pose_identity = 'source_cropped_rig_rest_open_palmar_face'

    def __init__(self):
        super().__init__(SOURCE / 'static_short_hand_011/cropped_rig.json')
        self.rest_points = np.asarray(self.data['frames'][0]['points'], dtype=float)
        self.rest_bones = {key: np.asarray(value, dtype=float) for key, value in self.data['rest_bones'].items()}
        if set(self.rest_bones) != set(self.bones):
            raise ValueError('Original rest skeleton does not match the cropped hand')
        texture = SOURCE / 'assets/model/new_diff.png'
        if not texture.is_file():
            raise ValueError('Original source skin texture missing')
        self.data['texture'] = os.path.relpath(texture, self.path.parent)
        self._pose_flat_palm()

    def _pose_flat_palm(self):
        """Splay the thumb away from the palmar pushing plane, using the rig.

        This is this new actor's open-hand pose, not an asset/physics edit.
        Derive the smallest thumb-root articulation that clears the measured
        palm plane. The four fingers retain their original rest transforms.
        Skin and native convex links use the same posed bones.
        """
        from scipy.spatial.transform import Rotation
        from scipy.optimize import brentq
        groups = self.collision_groups()
        thumb = np.unique(np.concatenate([ids for k, ids in groups.items() if k.startswith('thumb.')]))
        palm = groups['hand.R']
        pivot = self.rest_bones['thumb.01.R'][:3, 3]
        weights = {k: np.array([w.get(k, 0.) for w in self.data['skin_weights']]) for k in self.bones}
        def pose(angle):
            rotation = Rotation.from_rotvec([0., -angle, 0.]).as_matrix()
            move = np.eye(4)
            move[:3, :3] = rotation
            move[:3, 3] = pivot - rotation @ pivot
            bones = {k: (move @ b if k.startswith('thumb.') else b.copy())
                     for k, b in self.rest_bones.items()}
            points = np.zeros_like(self.rest_points)
            for k, b in bones.items():
                tf = b @ np.linalg.inv(self.rest_bones[k])
                points += (self.rest_points @ tf[:3, :3].T + tf[:3, 3]) * weights[k][:, None]
            return points, bones
        def gap(angle):
            points, _ = pose(angle)
            return float(points[thumb, 2].max() - points[palm, 2].max())
        if gap(0.) > 0 and gap(np.pi / 2) >= 0:
            raise ValueError('Source thumb rig cannot present an open palm')
        angle = brentq(gap, 0., np.pi / 2) if gap(0.) > 0 else 0.
        self.thumb_splay_deg = float(np.degrees(angle))
        self.posed_points, self.posed_bones = pose(angle)

    def sample(self, closure):
        if type(closure) not in (int, float) or closure != 0.:
            raise ValueError('Palm pushing uses only the original fully open/rest pose')
        return self.posed_points.copy(), {key: value.copy() for key, value in self.posed_bones.items()}
