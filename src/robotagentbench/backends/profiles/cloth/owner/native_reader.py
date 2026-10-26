"""Strict read-only solver tensors; no USD-point fallback, setter or stepping."""
from dataclasses import dataclass
import json
import math
import numpy as np


def host_copy(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


@dataclass(frozen=True)
class NativeSample:
    episode_id: str
    body_path: str
    topology_id: str
    sample_index: int
    time_s: float
    positions: np.ndarray
    velocities: np.ndarray


class NativeFEMReader:
    def __init__(self,*,episode_id,body_path,mesh,simulation_view,read_clock,fault_diagnostic=None):
        if not episode_id or not body_path.startswith('/World/')or '..'in body_path.split('/'):
            raise ValueError('Explicit episode and canonical physical body required')
        if type(simulation_view.device_ordinal)is not int or simulation_view.device_ordinal<0:
            raise ValueError('Actual native GPU tensor pipeline required; no CPU/archival substitution')
        self.episode_id,self.body_path,self.mesh=episode_id,body_path,mesh
        self.clock,self.simulation_view=read_clock,simulation_view
        self.view=simulation_view.create_surface_deformable_body_view(body_path)
        v=self.view
        if v.count!=1 or tuple(v.prim_paths)!=(body_path,)or v.max_simulation_nodes_per_body!=len(mesh.uv):
            raise ValueError('Native FEM identity or node count differs')
        triangles=host_copy(v.get_simulation_element_indices())
        if triangles.shape!=(1,len(mesh.faces),3)or not np.array_equal(triangles[0],mesh.faces):
            raise ValueError('Native FEM triangle topology differs; no trimming/remapping')
        self.last_clock=self.cached=None
        self.last_fault=None
        self.fault_buffers=None
        self.fault_diagnostic=fault_diagnostic

    def read(self):
        before=tuple(self.clock());step,time=before
        if type(step)is not int or step<0 or not math.isfinite(time)or time<0:raise ValueError('Invalid measured physics clock')
        if before==self.last_clock:return self.cached
        if self.last_clock is not None and (step!=self.last_clock[0]+1 or not math.isclose(time-self.last_clock[1],1/60,rel_tol=0,abs_tol=1e-8)):
            raise ValueError('Missing/stale native 60Hz sample')
        q=host_copy(self.view.get_simulation_nodal_positions());qd=host_copy(self.view.get_simulation_nodal_velocities())
        if q.shape!=(1,len(self.mesh.uv),3)or qd.shape!=q.shape or not np.isfinite(q).all()or not np.isfinite(qd).all():
            # Diagnostics only: never trim, replace NaNs, reuse stale values or
            # turn an invalid physics frame into a zero-scored valid episode.
            self.last_fault=dict(body_path=self.body_path,step=step,time_s=float(time),
                expected_shape=[1,len(self.mesh.uv),3],positions_shape=list(q.shape),
                velocities_shape=list(qd.shape),positions_nonfinite=int((~np.isfinite(q)).sum()),
                velocities_nonfinite=int((~np.isfinite(qd)).sum()),
                last_valid_step=self.last_clock[0]if self.last_clock else None)
            self.fault_buffers={'positions':q.copy(),'velocities':qd.copy()}
            if self.cached is not None:
                self.last_fault['last_valid_bounds_m']=[self.cached.positions.min(0).tolist(),self.cached.positions.max(0).tolist()]
                self.last_fault['last_valid_max_speed_m_s']=float(np.linalg.norm(self.cached.velocities,axis=1).max())
            if self.fault_diagnostic is not None:
                # A fenced reread is diagnostic ONLY. The first invalid frame
                # remains fatal even if the second copy happens to be finite.
                try:
                    diagnostic=self.fault_diagnostic(self.view)
                    if tuple(self.clock())!=before:raise RuntimeError('Fault diagnostic advanced native time')
                    self.last_fault['same_clock_fenced_reread']={}
                    for name,array in diagnostic.items():
                        value=host_copy(array);self.fault_buffers['diagnostic_'+name]=value
                        self.last_fault['same_clock_fenced_reread'][name]=dict(shape=list(value.shape),nonfinite=int((~np.isfinite(value)).sum()))
                except Exception as exc:self.last_fault['reread_diagnostic_error']=str(exc)
            raise ValueError('Native buffers have wrong shape/nonfinite values: '+json.dumps(self.last_fault,allow_nan=False))
        if tuple(self.clock())!=before:raise RuntimeError('Native readback advanced the owner clock')
        q=q[0].copy();qd=qd[0].copy();q.flags.writeable=qd.flags.writeable=False
        self.cached=NativeSample(self.episode_id,self.body_path,self.mesh.identity,step,float(time),q,qd)
        self.last_clock=before
        return self.cached

    def close(self):
        self.view=self.simulation_view=None;self.cached=self.fault_buffers=self.fault_diagnostic=None
