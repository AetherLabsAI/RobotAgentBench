"""Read-only explored maps for the public native worker.

The original geometry/FK/reveal/render rules are preserved. Full static grids
and robot geometry stay worker-side; only the explored PNG and its original
metric metadata cross the observation boundary.
"""
import base64
from copy import deepcopy
import io
import json
import math
from pathlib import Path

from robotagentbench.interfaces.observation_extras import MAP_SCHEMA, MAP_SOURCE, public_navigation


class ExploredMapObservation:
    def __init__(self, data, output=None):
        from ._original.navmap import NavMap
        from ._original.explored import ExploredMap
        from ._original.kinematics import RobotModel
        self.nav = NavMap(data)
        self.model = RobotModel(str(Path(data)/'robot_model.json'))
        self.explored = ExploredMap(self.nav)
        self.trajectory = []
        self.output = Path(output) if output is not None else None
        self._episode = self._last_frame = self._last_packet = None
        camera = self.model.cameras['Head_Camera']
        self.hfov = math.degrees(2*math.atan(camera['horizontal_aperture']/(2*camera['focal_length'])))
        self.vfov = math.degrees(2*math.atan(camera['vertical_aperture']/(2*camera['focal_length'])))

    def __call__(self, row):
        # One provider per episode. Never carry explored cells across resets.
        if self._episode is not None and row['episode_id'] != self._episode:
            raise ValueError('Explored map belongs to another episode')
        frame = (row['step'], row['time_s'], dict(row['joints']), deepcopy(row['base_state']))
        if self._last_frame is not None:
            if frame[0] < self._last_frame[0] or frame[1] < self._last_frame[1]:
                raise ValueError('Cannot rewind an explored map')
            if frame[0] == self._last_frame[0]:
                if frame != self._last_frame:
                    raise ValueError('Robot measurements changed within one map frame')
                return deepcopy(self._last_packet)
        base_state = row['base_state']
        base = tuple(base_state['position_world_m'][:2])+(base_state['yaw_world_unwrapped_rad'],)
        camera = self.model.head_camera(row['joints'], base)
        self.explored.reveal(camera, self.hfov, self.vfov, base[:2])
        if not self.trajectory or self.trajectory[-1] != base[:2]:
            self.trajectory.append(base[:2])
        image = self.explored.render(base, self.trajectory)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        raw = buffer.getvalue()
        packet = public_navigation(dict(schema=MAP_SCHEMA, step=row['step'],
            map_png=base64.b64encode(raw).decode(), origin_m=self.nav.origin.tolist(),
            resolution_m=self.nav.res, shape=list(self.nav.shape),
            explored_fraction=self.explored.fraction(), map_source=MAP_SOURCE,
            premapped=False, task_labels=False, body_footprint_radius_m=self.nav.radius), row['step'])
        if self.output is not None:
            (self.output/f'map_step{row["step"]:06d}.png').write_bytes(raw)
        self._episode = row['episode_id']
        self._last_frame, self._last_packet = frame, deepcopy(packet)
        return packet


def from_stage(stage, *, spawn, output):
    """Extract this admitted scene, without loading another scene or stepping it.

    The data directory is private evaluator output, not part of observations.
    Exclusions and raster/voxel parameters match the original map builder.
    No camera, material, collision, physical, or task-scoring value is written.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    from ._original import scene_geometry
    from ._original.extract_robot_model import extract
    from ._original.kinematics import RobotModel
    directory = Path(output)
    directory.mkdir(exist_ok=False)
    data = directory/'private_geometry'
    data.mkdir()
    robot = extract(stage, str(stage.GetRootLayer().identifier))
    (data/'robot_model.json').write_text(json.dumps(robot))
    model = RobotModel(robot)
    inv = np.linalg.inv(model.authored_links['base_link'])
    radius = 0.
    cache = UsdGeom.XformCache()
    for name in ('base_link', 'body_link1'):
        for child in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot/'+name), Usd.TraverseInstanceProxies()):
            if not child.IsA(UsdGeom.Mesh):
                continue
            if UsdGeom.Imageable(child).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            points = UsdGeom.Mesh(child).GetPointsAttr().Get()
            if not points:
                continue
            world = scene_geometry._matrix(cache.GetLocalToWorldTransform(child))
            points = np.asarray(points, dtype=float)@world[:3,:3].T+world[:3,3]
            local = points@inv[:3,:3].T+inv[:3,3]
            local = local[local[:,2] < .9]
            if len(local):
                radius = max(radius, float(np.linalg.norm(local[:,:2], axis=1).max()))
    if radius <= 0:
        raise ValueError('Original measured robot footprint is missing')
    points = scene_geometry.sample_scene(stage, step=.02,
        skip_prefixes=('/World/Robot', '/World/RoomEpisode'))
    if not len(points):
        raise ValueError('No static scene geometry for an explored map')
    maps = scene_geometry.build_maps(points)
    voxel = .05
    origin = np.array(maps['origin']+[0.])
    index = np.floor((points-origin)/voxel).astype(int)
    shape = (int(np.ceil(2.2/voxel)),
        int(np.ceil(maps['shape'][0]*maps['resolution']/voxel))+1,
        int(np.ceil(maps['shape'][1]*maps['resolution']/voxel))+1)
    keep = ((index[:,2]>=0)&(index[:,2]<shape[0])&(index[:,1]>=0)
            &(index[:,1]<shape[1])&(index[:,0]>=0)&(index[:,0]<shape[2]))
    voxels = np.zeros(shape, dtype=bool)
    voxels[index[keep,2],index[keep,1],index[keep,0]] = True
    np.savez_compressed(data/'maps.npz', blocked=maps['blocked'], height=maps['height'],
        floor=maps['floor'], voxels=np.packbits(voxels), voxel_shape=np.array(shape), voxel_size=voxel)
    (data/'scene.json').write_text(json.dumps(dict(spawn=list(spawn), robot_radius_m=radius,
        map=dict(origin=maps['origin'], resolution=maps['resolution'], shape=maps['shape']))))
    return ExploredMapObservation(data, directory)
