"""Isolated, actor-free visual replay of measured native snapshots.

Importing this module is CPU-only. Only main() starts the render application.
The source layer is never edited/saved; all edits target an anonymous flatten.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import sys

np = None


def _numpy():
    # Runtime may start with -I -S; main installs the supplied SDK site first.
    global np
    if np is None:
        import numpy as numpy_module
        np = numpy_module
    return np


CAMERAS = {
    'head': ('/World/Robot/head_link2/Head_Camera', (1280, 800)),
    'left': ('/World/Robot/gripper_l_base_link/Left_Camera', (848, 480)),
    'right': ('/World/Robot/gripper_r_base_link/Right_Camera', (848, 480)),
}


def snapshot_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[0-9]{4}\.npz', name):
        raise ValueError('Snapshot must be a direct four-digit NPZ filename')
    path = root / name
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError('Missing or non-local snapshot')
    return path


def _strings(value, label, *, scalar=False):
    np = _numpy()
    array = np.asarray(value)
    if array.dtype.kind != 'U' or array.ndim != (0 if scalar else 1):
        raise ValueError('Expected Unicode ' + label)
    result = str(array.item()) if scalar else array.tolist()
    if scalar:
        if not result:
            raise ValueError('Empty ' + label)
    elif len(set(result)) != len(result):
        raise ValueError('Duplicate ' + label)
    return result


def load_snapshot(path):
    np = _numpy()
    with np.load(path, allow_pickle=False) as data:
        required = {'episode_id', 'physics_step', 'time_s', 'rigid_paths',
                    'rigid_pose', 'cloth_paths'}
        if not required <= set(data.files):
            raise ValueError('Incomplete native snapshot')
        episode = _strings(data['episode_id'], 'episode_id', scalar=True)
        step = np.asarray(data['physics_step'])
        time = np.asarray(data['time_s'])
        if step.shape != () or step.dtype.kind not in 'iu' or int(step) < 0:
            raise ValueError('Invalid measured physics_step')
        if time.shape != () or time.dtype.kind not in 'fiu' or not np.isfinite(time) or float(time) < 0:
            raise ValueError('Invalid measured time_s')
        paths = _strings(data['rigid_paths'], 'rigid_paths')
        cloth = _strings(data['cloth_paths'], 'cloth_paths')
        if any(not p.startswith('/World/') or '//' in p or '..' in p.split('/') for p in paths):
            raise ValueError('Invalid rigid prim path')
        poses = np.array(data['rigid_pose'], dtype=float, copy=True)
        if poses.shape != (len(paths), 7) or not np.isfinite(poses).all():
            raise ValueError('Invalid measured rigid poses')
        if np.any(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1.) > 1e-4):
            raise ValueError('Invalid measured rigid quaternion')
        points = {}
        for prim in cloth:
            match = re.fullmatch(r'/World/TableclothPhysics/Cloth_([0-9]+)', prim)
            if not match:
                raise ValueError('Invalid cloth prim path')
            key = 'positions_' + match.group(1)
            if key not in data.files:
                raise ValueError('Missing measured cloth points: ' + key)
            value = np.array(data[key], dtype=float, copy=True)
            if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3 or not np.isfinite(value).all():
                raise ValueError('Invalid measured cloth points')
            points[prim] = value
        if set(data.files) != required | {'positions_' + p.rsplit('_', 1)[1] for p in cloth}:
            raise ValueError('Unexpected native snapshot fields')
    return dict(episode_id=episode, physics_step=int(step), time_s=float(time),
                rigid_paths=paths, rigid_pose=poses, points=points)


def rotation_xyzw(quaternion):
    np = _numpy()
    x, y, z, w = np.asarray(quaternion, float) / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def polar_stretch(world_matrix):
    """Keep scale, shear and any reflection; replace only the proper rotation."""
    np = _numpy()
    linear = np.asarray(world_matrix, float)[:3, :3]
    u, values, vt = np.linalg.svd(linear)
    if min(values) < 1e-12:
        raise ValueError('Singular source world transform')
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    proper_rotation = u @ correction @ vt
    return proper_rotation.T @ linear


def measured_world(pose, stretch):
    np = _numpy()
    world = np.eye(4)
    world[:3, :3] = rotation_xyzw(pose[3:]) @ stretch
    world[:3, 3] = pose[:3]
    return world


def world_points_to_local(points, world):
    np = _numpy()
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ np.linalg.inv(world).T)[:, :3]


def physical_schema(name):
    return name.lower().startswith(('physics', 'physx', 'omniphysics'))


def reference_time(values, previous=None):
    refs = []
    for value in values:
        if (not isinstance(value, dict) or not value.get('referenceTimeDenominator')
                or 'referenceTimeNumerator' not in value):
            raise ValueError('Missing ReferenceTime')
        refs.append(Fraction(int(value['referenceTimeNumerator']),
                             int(value['referenceTimeDenominator'])))
    if len(refs) != 3 or len(set(refs)) != 1 or (previous is not None and refs[0] <= previous):
        raise ValueError('Stale or unsynchronized ReferenceTime')
    return refs[0]


class VisualStage:
    def __init__(self, scene):
        np = _numpy()
        from pxr import Usd, UsdGeom, UsdPhysics, Sdf
        self.Usd, self.UsdGeom = Usd, UsdGeom
        self.scene = scene
        self.source_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
        source = Usd.Stage.Open(str(scene))
        if source is None:
            raise ValueError('Cannot read native source stage')
        self.layer = source.Flatten()
        if not self.layer.anonymous:
            raise ValueError('Visual stage must be anonymous')
        self.stage = Usd.Stage.Open(self.layer)
        self.rigid, self.stretch, self.cloth = {}, {}, {}
        cache = UsdGeom.XformCache()
        self.camera_profile = self._cameras()
        # Capture original world stretch before changing any ancestor transforms.
        prims = list(self.stage.Traverse())
        self.source_scene_paths = {str(p.GetPath()) for p in prims if p.IsA(UsdPhysics.Scene)}
        self.sdk_bookkeeping_scenes = []
        for prim in prims:
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetAttribute('physics:rigidBodyEnabled').Get() is not False:
                self.rigid[path] = prim
                self.stretch[path] = polar_stretch(np.asarray(cache.GetLocalToWorldTransform(prim)).T)
            if re.fullmatch(r'/World/TableclothPhysics/Cloth_[0-9]+', path):
                if not prim.IsA(UsdGeom.Mesh):
                    raise ValueError('Native cloth is not a Mesh')
                self.cloth[path] = UsdGeom.Mesh(prim)
        # This is the isolated render copy only, never a shared native stage.
        for prim in prims:
            kept = [name for name in prim.GetAppliedSchemas() if not physical_schema(name)]
            prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(kept))
            if prim.IsA(UsdPhysics.Scene) or prim.IsA(UsdPhysics.Joint):
                prim.SetActive(False)
        self.assert_visual_only()
        self.assert_source()

    def _cameras(self):
        np = _numpy()
        result = {}
        for name, (path, resolution) in CAMERAS.items():
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(self.UsdGeom.Camera):
                raise ValueError('Missing original camera: ' + path)
            image_size = prim.GetAttribute('omni:lensdistortion:opencvPinhole:imageSize').Get()
            if image_size is not None and tuple(map(int, image_size)) != resolution:
                raise ValueError('Native camera resolution mismatch')
            result[name] = (np.asarray(self.UsdGeom.Xformable(prim).GetLocalTransformation()).copy(),
                            {str(a.GetName()): repr(a.Get()) for a in prim.GetAttributes()})
        return result

    def assert_cameras(self):
        np = _numpy()
        current = self._cameras()
        for name in CAMERAS:
            if (not np.array_equal(current[name][0], self.camera_profile[name][0])
                    or current[name][1] != self.camera_profile[name][1]):
                raise RuntimeError('Native camera mount/intrinsics changed')

    def assert_source(self):
        if hashlib.sha256(self.scene.read_bytes()).hexdigest() != self.source_sha256:
            raise RuntimeError('Native exported source changed')

    def assert_visual_only(self):
        from pxr import UsdPhysics
        if not self.layer.anonymous:
            raise RuntimeError('Visual edits escaped anonymous layer')
        offending, bookkeeping = [], []
        for prim in self.stage.Traverse():
            schemas = list(prim.GetAppliedSchemas())
            # Installed Replicator creates this empty clock/bookkeeping Scene
            # on a later step. It is not a source scene and has no actors. Never
            # allow a revived source scene, rigid/collision/FEM API or joint.
            if (prim.IsA(UsdPhysics.Scene)
                    and re.fullmatch(r'/PhysicsScene_[0-9a-f]{12}', str(prim.GetPath()))
                    and str(prim.GetPath()) not in self.source_scene_paths
                    and set(schemas) <= {'PhysxSceneAPI'}):
                bookkeeping.append(str(prim.GetPath()))
                continue
            if (prim.IsA(UsdPhysics.Scene) or prim.IsA(UsdPhysics.Joint)
                    or any(physical_schema(s) for s in schemas)):
                offending.append(dict(path=str(prim.GetPath()), type=str(prim.GetTypeName()),
                    schemas=schemas))
        if offending:
            raise RuntimeError('Physical schema left active in visual copy: '+json.dumps(offending))
        self.sdk_bookkeeping_scenes = bookkeeping

    def apply(self, snapshot):
        np = _numpy()
        from pxr import Gf, Vt
        if set(snapshot['rigid_paths']) != set(self.rigid):
            raise ValueError('Snapshot must contain every enabled native rigid body')
        if set(snapshot['points']) != set(self.cloth):
            raise ValueError('Snapshot must contain every native cloth')
        poses = dict(zip(snapshot['rigid_paths'], snapshot['rigid_pose']))
        for path in sorted(self.rigid, key=lambda p: (p.count('/'), p)):
            prim = self.rigid[path]
            world = measured_world(poses[path], self.stretch[path])
            # USD uses row vectors. Compute in column convention and transpose once.
            parent_world = np.asarray(self.UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
            local = np.linalg.solve(parent_world, world)
            xf = self.UsdGeom.Xformable(prim)
            xf.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
            xf.SetResetXformStack(False)
        for path, points in snapshot['points'].items():
            mesh = self.cloth[path]
            if len(points) != len(mesh.GetPointsAttr().Get()):
                raise ValueError('Measured cloth topology mismatch')
            world = np.asarray(self.UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())).T
            local = world_points_to_local(points, world)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(local, dtype=np.float32)))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(
                [local.min(0), local.max(0)], dtype=np.float32)))
        self.assert_cameras()
        self.assert_visual_only()
        self.assert_source()

    def assert_snapshot(self, snapshot):
        """Read-only post-render proof; no FK or prescribed-command substitute."""
        np = _numpy()
        cache = self.UsdGeom.XformCache()
        for path, pose in zip(snapshot['rigid_paths'], snapshot['rigid_pose']):
            expected = measured_world(pose, self.stretch[path])
            actual = np.asarray(cache.GetLocalToWorldTransform(self.rigid[path])).T
            bound = 128*np.finfo(float).eps*max(1., float(np.abs(expected).max()))
            if not np.allclose(actual, expected, rtol=0, atol=bound):
                raise RuntimeError('Rendered rigid transform differs from measured snapshot: '+path)
        for path, expected in snapshot['points'].items():
            mesh = self.cloth[path]
            local = np.asarray(mesh.GetPointsAttr().Get())
            world = np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim())).T
            actual = (np.column_stack((local, np.ones(len(local)))) @ world.T)[:, :3]
            bound = 8*np.finfo(np.float32).eps*max(1., float(np.abs(expected).max()))
            if not np.allclose(actual, expected, rtol=0, atol=bound):
                raise RuntimeError('Rendered cloth differs from measured snapshot: '+path)


class Renderer:
    def __init__(self, visual, app, root):
        import omni.usd
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import UsdUtils
        self.visual, self.app, self.root, self.rep = visual, app, root, rep
        self.previous, self.last_snapshot = None, None
        self.seen = set()
        self.frames = root / 'frames'
        if self.frames.is_symlink() or self.frames.resolve() != self.frames:
            raise ValueError('Non-local frame output')
        self.frames.mkdir(exist_ok=True)
        self.cache_id = UsdUtils.StageCache.Get().Insert(visual.stage)
        attached = []
        if not omni.usd.get_context().attach_stage_with_callback(
                self.cache_id.ToLongInt(), lambda ok, error: attached.append((ok, error))):
            raise RuntimeError('Cannot attach anonymous visual stage')
        for _ in range(300):
            app.update()
            if attached:
                break
        if not attached or not attached[0][0]:
            raise RuntimeError('Anonymous visual stage attachment failed: ' + repr(attached))
        omni.timeline.get_timeline_interface().pause()
        self.products, self.rgb, self.references = [], {}, {}
        for name, (path, resolution) in CAMERAS.items():
            product = rep.create.render_product(path, resolution)
            self.products.append(product)
            self.rgb[name] = rep.AnnotatorRegistry.get_annotator('rgb')
            self.references[name] = rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            for annotator in (self.rgb[name], self.references[name]):
                annotator.attach([product])

    def render(self, name):
        np = _numpy()
        from PIL import Image
        snapshot = load_snapshot(snapshot_path(self.root, name))
        if name in self.seen:
            raise ValueError('Snapshot was already rendered')
        if self.last_snapshot is not None:
            last = self.last_snapshot
            if (snapshot['episode_id'] != last['episode_id']
                    or snapshot['physics_step'] < last['physics_step'] or snapshot['time_s'] < last['time_s']):
                raise ValueError('Snapshot episode/physical time regressed')
        self.visual.apply(snapshot)
        current = None
        for _ in range(60):
            # Only this schema-stripped process owns the renderer timeline.
            self.rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
            arrays = {key: np.array(a.get_data(), copy=True) for key, a in self.rgb.items()}
            valid = all(arrays[key].dtype == np.uint8 and arrays[key].shape in (
                (size[1], size[0], 3), (size[1], size[0], 4)) for key, (_, size) in CAMERAS.items())
            if not valid:
                continue
            try:
                current = reference_time([a.get_data() for a in self.references.values()], self.previous)
            except ValueError:
                continue
            break
        if current is None:
            raise RuntimeError('No fresh, synchronized original-resolution RGB batch')
        self.visual.assert_cameras()
        self.visual.assert_visual_only()
        self.visual.assert_source()
        self.visual.assert_snapshot(snapshot)
        files = {}
        for key, array in arrays.items():
            path = self.frames / (name[:-4] + '_' + key + '.png')
            if path.exists() or path.is_symlink():
                raise ValueError('Refusing to overwrite a rendered frame')
            Image.fromarray(array[:, :, :3]).save(path, format='PNG')
            files[key] = dict(path=str(path.relative_to(self.root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.previous, self.last_snapshot = current, snapshot
        self.seen.add(name)
        return dict(op='frame', episode_id=snapshot['episode_id'], physics_step=snapshot['physics_step'],
                    time_s=snapshot['time_s'], files=files, visual_only=True,
                    reference_time=[current.numerator, current.denominator],
                    simulated_body_count=0,
                    native_snapshot_visual_transform_guard_passed=True,
                    sdk_empty_bookkeeping_scenes=self.visual.sdk_bookkeeping_scenes,
                    source_sha256=self.visual.source_sha256)

    def close(self):
        for annotator in [*self.rgb.values(), *self.references.values()]:
            annotator.detach()
        for product in self.products:
            product.destroy()
        self.visual.assert_source()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--sdk-site', required=True)
    args = parser.parse_args()
    # Keep an independent RPC descriptor BEFORE any SDK import/native stdout.
    rpc = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    emit = lambda row: rpc.write(json.dumps(row, allow_nan=False) + '\n')
    app, renderer = None, None
    try:
        root, scene = Path(args.root).absolute(), Path(args.scene).absolute()
        if root.resolve() != root or not root.is_dir() or scene != root/'source.usda' or scene.is_symlink():
            raise ValueError('Expected a local root and its source.usda')
        sys.path.insert(0, args.sdk_site)
        from isaacsim import SimulationApp
        # User-approved resource cap for THIS render-only child. It does not
        # change the original physics owner's initialization or parameters.
        app = SimulationApp(dict(headless=True, width=1280, height=800, limit_cpu_threads=8))
        visual = VisualStage(scene)
        renderer = Renderer(visual, app, root)
        emit(dict(op='ready', required_rigid_paths=sorted(visual.rigid),
                  required_cloth_paths=sorted(visual.cloth), visual_only=True,
                  source_sha256=visual.source_sha256, cpu_thread_budget=8))
        for line in sys.stdin:
            request = json.loads(line)
            if request == {'op': 'close'}:
                renderer.close()
                renderer = None
                emit(dict(op='closing', visual_only=True))
                return 0
            if not isinstance(request, dict) or set(request) != {'op', 'snapshot'} or request['op'] != 'render':
                raise ValueError('Unknown visual-only RPC request')
            emit(renderer.render(request['snapshot']))
        raise EOFError('Visual RPC closed without explicit close')
    except BaseException as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit(dict(op='error', error=type(error).__name__ + ': ' + str(error), visual_only=True))
        return 1
    finally:
        if renderer is not None:
            renderer.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
