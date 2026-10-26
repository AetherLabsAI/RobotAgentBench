"""Real USD lights in the anonymous session layer, never a white image mask.

Ambient intensity/color and ONE existing RectLight transform are the complete
write allowlist. No new emitter, physics, target or camera properties are edited.
Side-light distance is inherited. Direct flash uses the approved clear segment
between the camera and machine, not the old fixed distance/downward view ray.
"""
import math
from .cycle import LightingCycle

CAMERA = '/World/Robot/head_link2/Head_Camera'


class SceneLighting:
    def __init__(self, stage, seed):
        from pxr import Usd, UsdGeom, UsdLux
        self.stage = stage
        self.layer = stage.GetSessionLayer()
        if not self.layer.anonymous:
            raise ValueError('Anonymous per-episode session layer required')
        self.backup = self.layer.ExportToString()
        self.saved = {}
        self.records = []
        self.cycle = LightingCycle(seed)
        self.closed = False
        self.placement_events = []
        if not stage.GetPrimAtPath(CAMERA):
            raise ValueError('Existing original head camera required')
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI) or not prim.IsActive():
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                continue
            api = UsdLux.LightAPI(prim)
            attrs = [api.GetIntensityAttr(), api.GetColorTemperatureAttr(), api.GetEnableColorTemperatureAttr()]
            if any(a.GetTimeSamples() or a.GetConnections() for a in attrs):
                raise ValueError('Animated/connected scene lights are not supported')
            intensity = api.GetIntensityAttr().Get()
            if intensity is None or not math.isfinite(intensity) or intensity < 0:
                raise ValueError('Invalid existing light intensity')
            if intensity > 0:
                self.records.append(dict(path=str(prim.GetPath()), intensity=float(intensity), type=prim.GetTypeName()))
        self.records.sort(key=lambda r: r['path'])
        rectangles = [r for r in self.records if r['type'] == 'RectLight']
        if not self.records or not rectangles:
            raise ValueError('Positive scene lights and an existing RectLight required')
        self.emitter = max(rectangles, key=lambda r: (r['intensity'], r['path']))
        prim = stage.GetPrimAtPath(self.emitter['path'])
        xform = UsdGeom.Xformable(prim)
        self.original_ops = xform.GetXformOpOrderAttr().Get()
        self.original_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        self.original_distance = (self.original_matrix.ExtractTranslation() -
            UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(CAMERA)).ExtractTranslation()).GetLength()
        if self.original_distance <= 0:
            raise ValueError('Degenerate original emitter/camera distance')

    def _set(self, attr, value):
        path = attr.GetPath()
        if str(path) not in self.saved:
            spec = self.layer.GetAttributeAtPath(path)
            self.saved[str(path)] = (bool(spec), bool(spec and spec.HasInfo('default')),
                                     spec.default if spec and spec.HasInfo('default') else None)
        attr.Set(value)

    def _restore_attr(self, path):
        existed, authored, value = self.saved[path]
        attr = self.stage.GetAttributeAtPath(path)
        if authored:
            attr.Set(value)
        else:
            attr.Clear()
        if not existed:
            attr.GetPrim().RemoveProperty(attr.GetName())

    def arm(self, now):
        self.cycle.arm(now)

    def _emitter_pose(self, side):
        from pxr import Gf, UsdGeom
        cache = UsdGeom.XformCache()
        matrix = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(CAMERA))
        center = matrix.ExtractTranslation()
        if side is None:
            from .clear_position import from_stage
            placement=from_stage(self.stage,CAMERA)
            self.placement_events.append(dict(time_s=self.cycle.last_time,**placement))
            eye=Gf.Vec3d(*placement['position_m'])
            direction=(eye-center).GetNormalized()
        else:
            direction = matrix.TransformDir(Gf.Vec3d(side,0,0)).GetNormalized()
            eye = center + direction * self.original_distance
        up = matrix.TransformDir(Gf.Vec3d(0, 1, 0)).GetNormalized()
        if abs(Gf.Dot(up, direction)) > 0.999999:
            raise ValueError('Degenerate light aiming basis')
        world = Gf.Matrix4d().SetLookAt(eye, center, up).GetInverse()
        prim = self.stage.GetPrimAtPath(self.emitter['path'])
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        return world * parent.GetInverse()

    def tick(self, now):
        from pxr import Sdf, Usd, UsdGeom, UsdLux
        changed = self.cycle.tick(now)
        if not changed:
            return []
        state = self.cycle.state
        with Usd.EditContext(self.stage, self.layer):
            for row in self.records:
                api = UsdLux.LightAPI(self.stage.GetPrimAtPath(row['path']))
                self._set(api.GetIntensityAttr(), row['intensity'] * state['multiplier'])
                self._set(api.GetEnableColorTemperatureAttr(), True)
                self._set(api.GetColorTemperatureAttr(), state['temperature_k'])
            prim = self.stage.GetPrimAtPath(self.emitter['path'])
            xform = UsdGeom.Xformable(prim)
            if state['mode'] in ('side', 'camera_flash'):
                # Separate op, not MakeMatrixXform(): original operations remain untouched.
                op = prim.GetAttribute('xformOp:transform:robustLighting')
                if not op:
                    op = prim.CreateAttribute('xformOp:transform:robustLighting', Sdf.ValueTypeNames.Matrix4d)
                    # This newly created session-only property must disappear on close.
                    self.saved[str(op.GetPath())] = (False, False, None)
                self._set(op, self._emitter_pose(state['side'] if state['mode'] == 'side' else None))
                self._set(xform.GetXformOpOrderAttr(), ['xformOp:transform:robustLighting'])
                if state['flash']:
                    self._set(UsdLux.LightAPI(prim).GetIntensityAttr(),
                              self.emitter['intensity'] * state['flash_multiplier'])
            else:
                path = str(xform.GetXformOpOrderAttr().GetPath())
                if path in self.saved:
                    existed, authored, value = self.saved[path]
                    if authored:
                        xform.GetXformOpOrderAttr().Set(value)
                    else:
                        xform.GetXformOpOrderAttr().Clear()
        return changed

    def close(self):
        from pxr import Usd
        if self.closed:
            return
        with Usd.EditContext(self.stage, self.layer):
            for path in reversed(self.saved):
                self._restore_attr(path)
        self.cycle.close()
        self.closed = True

    def snapshot(self):
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        return dict(state=dict(self.cycle.state), events=list(self.cycle.events),
                    lights=self.records, selected_emitter=self.emitter,
                    original_emitter_camera_distance_m=self.original_distance,
                    direct_flash_placement_events=list(self.placement_events),
                    current_emitter_world_position_m=list(cache.GetLocalToWorldTransform(
                        self.stage.GetPrimAtPath(self.emitter['path'])).ExtractTranslation()),
                    current_camera_world_position_m=list(cache.GetLocalToWorldTransform(
                        self.stage.GetPrimAtPath(CAMERA)).ExtractTranslation()),
                    property_write_allowlist=list(self.saved),
                    image_postprocessing=False, camera_parameter_writes=0,
                    object_pose_writes=0, physical_parameter_writes=0)
