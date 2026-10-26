"""Initialization-only, instance-layer scale adapter for approved paper sizes."""
import math
from .paper_variants import definition


def apply_variant(stage, asset_path, variant):
    from pxr import UsdGeom, Gf
    spec = definition(variant)
    prim = stage.GetPrimAtPath(asset_path)
    if not prim or not prim.IsActive():
        raise ValueError('Missing paper asset instance')
    if prim.GetCustomDataByKey('roomPaperVariant') is not None:
        raise ValueError('Paper instance already sized; do not scale twice')
    layer = stage.GetEditTarget().GetLayer()
    if layer != stage.GetRootLayer() or '/all_assets/' in layer.identifier:
        raise ValueError('Paper sizing requires a separate scene root layer')
    xf = UsdGeom.Xformable(prim)
    m = xf.GetLocalTransformation()
    scales = [math.sqrt(sum(m[i][j] ** 2 for j in range(3))) for i in range(3)]
    if any(abs(v - 1.5) > 1e-6 for v in scales):
        raise ValueError('Unexpected paper base scale; refusing double scaling')
    # USD uses row-vector transforms. Scale mesh axes BEFORE the retained root
    # rotation, not world XYZ after rotation (the source long axis is vertical).
    relative = Gf.Matrix4d().SetScale(Gf.Vec3d(*spec['relative_mesh_axis_scale']))
    if variant != 'large':
        xf.MakeMatrixXform().Set(relative * m)
    prim.SetCustomDataByKey('roomPaperVariant', variant)
    return spec


def plan_variant(obj):
    """Legacy plans without a variant keep their existing large geometry."""
    if obj['assetId'] != 'paper_ball':
        if 'paperVariant' in obj:
            raise ValueError('Paper variant attached to a non-paper asset')
        return None
    variant = obj.get('paperVariant', 'large')
    expected = definition(variant)['authored_pose_size_m']
    if 'size' in obj and (len(obj['size']) != 3 or
                         any(abs(a - b) > 1e-6 for a, b in zip(obj['size'], expected))):
        raise ValueError('Plan paper dimensions do not match selected variant')
    return variant
