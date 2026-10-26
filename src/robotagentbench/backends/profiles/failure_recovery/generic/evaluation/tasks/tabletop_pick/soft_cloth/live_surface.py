"""Read-only native surface snapshots; no USD-points fallback or solver writes.

This is an evidence reader, not a static-to-dynamic activation receipt. A live
surface view alone cannot prove the old static collider has been removed.
The installed PhysX tensor API uses body roots when constructing surface views.
"""
from dataclasses import dataclass
import math

import numpy as np

from arena_api.scene_transitions import PhysicsBoundary
from .clearance_transition import ClothSurface


@dataclass(frozen=True)
class NativeSurfaceSample:
    boundary: PhysicsBoundary
    surface: ClothSurface
    source: str = 'physx_deformable'


class LiveSurfaceReader:
    """One exact body/mesh and one episode, sampled at a completed physics step.

    Own a separate read-only simulation view in world coordinates. Never reset
    the tensor service, recreate the world's robot view, step, or write vertices.
    Runtime re-tensorization support must be tested separately before use after
    activation. An invalid view stops reads; it is not silently reconstructed.
    """
    def __init__(self, *, episode_id, body_root, expected_surface,
                 simulation_view, body_view, read_clock):
        if (not isinstance(episode_id, str) or not episode_id
                or not isinstance(body_root, str)
                or not body_root.startswith('/World/Assets/')
                or any(x in ('', '.', '..') for x in body_root.split('/')[1:])
                or not isinstance(expected_surface, ClothSurface)
                or not expected_surface.mesh.startswith(body_root + '/')
                or not callable(read_clock)):
            raise ValueError('Require an explicit episode, cloth subtree and physical clock')
        self.episode_id, self.body_root = episode_id, body_root
        self.mesh = expected_surface.mesh
        self.node_count = len(expected_surface.points)
        self.triangles = expected_surface.triangles
        self.simulation_view, self.body_view = simulation_view, body_view
        self.read_clock = read_clock
        self.last = None
        self.error = None
        self.closed = False
        self._validate_view()

    @classmethod
    def bind(cls, *, stage, world, episode_id, body_root, expected_surface):
        """Bind only after native creation; this method does not activate a body."""
        from pxr import UsdGeom
        import omni.physics.tensors as tensors

        if (UsdGeom.GetStageUpAxis(stage) != 'Z'
                or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
            raise RuntimeError('Require the existing metre-scale Z-up stage')
        body = stage.GetPrimAtPath(body_root)
        mesh = stage.GetPrimAtPath(expected_surface.mesh)
        if (not body or not body.HasAPI('OmniPhysicsDeformableBodyAPI')
                or not mesh or not mesh.HasAPI('OmniPhysicsSurfaceDeformableSimAPI')):
            raise RuntimeError('Native surface schemas absent; reader will not create them')
        simulation_view = tensors.create_simulation_view('numpy')
        simulation_view.set_subspace_roots('/')  # Reader coordinates, not physics settings.
        body_view = simulation_view.create_surface_deformable_body_view(body_root)
        return cls(episode_id=episode_id, body_root=body_root,
                   expected_surface=expected_surface, simulation_view=simulation_view,
                   body_view=body_view,
                   read_clock=lambda: (int(world.current_time_step_index),
                                       float(world.current_time)))

    def _validate_view(self):
        view = self.body_view
        if not self.simulation_view.is_valid:
            raise RuntimeError('Native tensor view invalid; no USD or stale-state fallback')
        if (view.count != 1 or tuple(view.prim_paths) != (self.body_root,)
                or tuple(view.simulation_mesh_prim_paths) != (self.mesh,)
                or tuple(view.collision_mesh_prim_paths) != (self.mesh,)):
            raise RuntimeError('Native cloth identity or collision mesh differs')
        if (view.num_nodes_per_element != 3
                or view.max_simulation_nodes_per_body != self.node_count
                or view.max_simulation_elements_per_body != len(self.triangles)):
            raise RuntimeError('Native cloth vertex/triangle counts differ')
        indices = np.asarray(view.get_simulation_element_indices())
        if (indices.dtype.kind not in 'iu'
                or indices.shape != (1, len(self.triangles), 3)
                or not np.array_equal(indices[0], np.asarray(self.triangles))):
            raise RuntimeError('Native topology/vertex numbering differs from the bound cloth')

    def read(self, boundary):
        if self.closed or self.error is not None:
            raise RuntimeError('Native cloth reader is closed or failed')
        try:
            if (not isinstance(boundary, PhysicsBoundary)
                    or boundary.episode_id != self.episode_id
                    or type(boundary.physics_step) is not int or boundary.physics_step < 0
                    or type(boundary.time_s) not in (int, float)
                    or not math.isfinite(boundary.time_s) or boundary.time_s < 0
                    or self.read_clock() != (boundary.physics_step, boundary.time_s)):
                raise RuntimeError('Cloth read is not at the requested episode/physics boundary')
            self._validate_view()
            if self.last is not None:
                old = self.last.boundary
                if boundary == old:
                    return self.last  # Same boundary consumers share one immutable sample.
                if (boundary.physics_step <= old.physics_step
                        or boundary.time_s <= old.time_s):
                    raise RuntimeError('Stale native cloth read')
            points = np.asarray(self.body_view.get_simulation_nodal_positions())
            if (points.shape != (1, self.node_count, 3)
                    or points.dtype.kind != 'f' or not np.isfinite(points).all()):
                raise RuntimeError('Missing, malformed or nonfinite native cloth positions')
            # Tuples copy the reusable tensor buffer; later reads cannot mutate evidence.
            surface = ClothSurface(self.mesh,
                tuple(tuple(float(x) for x in p) for p in points[0]), self.triangles)
            if self.read_clock() != (boundary.physics_step, boundary.time_s):
                raise RuntimeError('Physics advanced during cloth read')
            self.last = NativeSurfaceSample(boundary, surface)
            return self.last
        except Exception as exc:
            self.error = str(exc)
            raise

    def close(self):
        self.closed = True
        self.last = None
        # Never call tensors.reset(): other task/robot views belong to the worker.
        self.body_view = self.simulation_view = None
