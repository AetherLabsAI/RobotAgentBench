"""Read existing live rigid bodies without authoring USD or configuring PhysX.

Bind after World.reset, once per episode. Unlike SingleRigidPrim this does not
apply schemas, reset transforms/velocities, set sleep thresholds, register reset
callbacks, prepare sensors, or flush stage changes. Never fall back to USD poses:
missing live physics is an invalid measurement, not a static observation.
"""
import numpy as np


class ReadOnlyRigidBody:
    """Minimal measurement interface; deliberately exposes no physics setters."""

    def __init__(self, prim_path):
        if (not isinstance(prim_path, str) or not prim_path.startswith('/')
                or any(c in prim_path for c in '*?[]()|\\')):
            raise ValueError('Expected one absolute rigid-body prim path')
        self.prim_path = prim_path
        self._view = None
        self._simulation = None

    def initialize(self):
        if self._view is not None:
            raise RuntimeError('Create a new reader for each physics reset/episode')
        from isaacsim.core.simulation_manager import SimulationManager
        simulation = SimulationManager.get_physics_sim_view()
        if simulation is None:
            raise RuntimeError('Live physics must be initialized before binding a reader')
        view = simulation.create_rigid_body_view([self.prim_path])
        if view.count != 1:
            raise RuntimeError(f'Expected one live rigid body at {self.prim_path}, got {view.count}')
        self._simulation = simulation
        self._view = view

    def _values(self, method, width):
        if self._view is None:
            raise RuntimeError('Rigid-body reader is not initialized')
        # Preserve the backend dtype; return owned arrays, never the native buffer.
        values = getattr(self._view, method)()
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        elif hasattr(values, 'numpy'):
            values = values.numpy()
        values = np.asarray(values)
        if values.size != width or not np.isfinite(values).all():
            raise RuntimeError(f'Invalid live {method} measurement for {self.prim_path}')
        return values.reshape(width).copy()

    def get_world_pose(self):
        pose = self._values('get_transforms', 7)
        return pose[:3], pose[[6, 3, 4, 5]]  # PhysX xyzw -> public wxyz

    def get_com(self):
        com = self._values('get_coms', 7)
        return com[:3], com[[6, 3, 4, 5]]  # body-local COM, not world position

    def get_mass(self):
        return float(self._values('get_masses', 1)[0])

    def get_inertia_tensor(self):
        return self._values('get_inertias', 9).reshape(3, 3)

    def get_linear_velocity(self):
        return self._values('get_velocities', 6)[:3]

    def get_angular_velocity(self):
        return self._values('get_velocities', 6)[3:]
