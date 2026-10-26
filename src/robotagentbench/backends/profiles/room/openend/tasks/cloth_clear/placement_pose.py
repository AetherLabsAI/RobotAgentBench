"""Scene-layer initial poses; never mutate asset geometry or physics."""
import numpy as np
from pxr import Gf

FACE_UP_ASSETS = frozenset(('cloth_clear_penguin', 'cloth_clear_bear',
                            'cloth_clear_rabbit', 'cloth_clear_cat'))


# Stable orientations measured in cloth coordinates during native hold tests.
# Keep world position out of this policy so every layout still samples position.
SETTLED_QUATERNIONS = {
    'cloth_clear_bear': [0.7086123599185601,-0.6978556853000543,-0.07468652049937106,-0.0727194877779416],
    'cloth_clear_cat': [-0.6883538781164134,0.6940495189304785,0.13697523461167707,0.16031868812792088],
    'cloth_clear_rabbit': [-0.6257298146691074,0.7800395844579815,-0.0005532616804025142,-0.00040844510891905483],
    'cloth_clear_duck': [0.9627891722504289,0.19760931228767922,-0.18218307525589633,0.02823048527029474],
    'cloth_clear_pouch': [0.9977573119723411,0.06680998311258743,-0.004054653589351027,0.0007007014976065444],
    'cloth_clear_tape_measure': [0.9934794118636477,-0.1139742279209031,0.002639404326397866,-0.0012774509038498574],
}


def initial_pose_policy(asset_id):
    if asset_id in SETTLED_QUATERNIONS:
        return 'settled_support_v1'
    return 'penguin_face_up' if asset_id == 'cloth_clear_penguin' else 'source_pose_unchanged'


def initial_pose(stage, asset_id):
    from openend.tasks.cloth_clear.scene import bounds
    rotation = Gf.Matrix4d(1)
    if asset_id not in FACE_UP_ASSETS and asset_id not in SETTLED_QUATERNIONS:
        return rotation, bounds(stage, '/Asset')
    # Rounded supports can tip during settling. Start in a verified resting
    # orientation and include ALL collision geometry in the packing box.
    from openend.tasks.remote_transfer.native_geometry import meshes
    if asset_id in SETTLED_QUATERNIONS:
        q = SETTLED_QUATERNIONS[asset_id]
        rotation.SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    else:
        rotation.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90))
    vertices = np.vstack([v for _, v, _, _ in meshes(stage, '/Asset')])
    transformed = (np.c_[vertices, np.ones(len(vertices))] @ np.asarray(rotation))[:, :3]
    return rotation, np.array([transformed.min(0), transformed.max(0)])


def check_occupancy(boxes, cloth):
    """Same approved 8 mm edge / 15 mm pair gaps, after pose application."""
    for i, box in enumerate(boxes):
        if (np.any(box[0, :2] < cloth[0, :2] + .008 - 1e-7)
                or np.any(box[1, :2] > cloth[1, :2] - .008 + 1e-7)):
            raise ValueError('Posed cloth target escapes placement boundary')
        for other in boxes[:i]:
            if not np.any(np.maximum(box[0, :2]-other[1, :2],
                                     other[0, :2]-box[1, :2]) >= .015 - 1e-7):
                raise ValueError('Posed cloth targets violate original pair gap')
