"""Cloth component for the existing GenieSim app. No input/UI or robot control.
Call author() once before World.reset/play; call readback() after physics steps.
All coefficients are uncalibrated diagnostic inputs, not validated real material.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ClothConfig:
    radius: float = .0015
    density: float = .35
    friction: float = .35
    stretch: float = 5000.
    shear: float = 5000.
    bend: float = 100.
    spring_damping: float = 1.
    velocity_damping: float = .8
    iterations: int = 64

class PhysXClothBackend:
    def __init__(self, stage, initial_positions, uv, faces, cloth_ids, config=ClothConfig()):
        self.stage=stage;self.q0=np.asarray(initial_positions,float);self.uv=np.asarray(uv,float);self.faces=np.asarray(faces,int);self.ids=np.asarray(cloth_ids,int);self.cfg=config;self.paths=[];self.counts=[];self.views=[]
        if self.q0.shape!=(len(self.ids),3) or self.uv.shape!=(len(self.ids),2) or not np.isfinite(self.q0).all():raise ValueError('Invalid initial material mesh')
        if min(config.radius,config.density,config.stretch,config.shear)<=0:raise ValueError('Invalid material parameters')
        self.authored=False
    def author(self):
        import omni.timeline
        from pxr import UsdPhysics,PhysxSchema,UsdShade,Vt
        from omni.physx.scripts import particleUtils
        from tablecloth_native_material import woven_springs
        if self.authored or omni.timeline.get_timeline_interface().is_playing():raise RuntimeError('Cloth can only be authored before episode starts')
        if self.stage.GetPrimAtPath('/World/TableclothPhysics'):raise RuntimeError('Refuse to replace existing cloth state')
        scenes=[p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes)!=1:raise RuntimeError('Need one authoritative physics scene')
        px=PhysxSchema.PhysxSceneAPI(scenes[0])
        if px.GetEnableGPUDynamicsAttr().Get() is not True:raise RuntimeError('Resolve shared GPU physics setting before authoring cloth')
        c=self.cfg;system='/World/TableclothPhysics/Particles';material='/World/TableclothPhysics/Material'
        particleUtils.add_physx_particle_system(self.stage,system,simulation_owner=scenes[0].GetPath(),contact_offset=c.radius*1.5,rest_offset=c.radius,particle_contact_offset=c.radius*1.5,solid_rest_offset=c.radius,solver_position_iterations=c.iterations,global_self_collision_enabled=True,non_particle_collision_enabled=True)
        PhysxSchema.PhysxParticleSystem(self.stage.GetPrimAtPath(system)).CreateEnableCCDAttr(True)
        particleUtils.add_pbd_particle_material(self.stage,material,friction=c.friction,damping=c.velocity_damping)
        UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath(system)).Bind(UsdShade.Material(self.stage.GetPrimAtPath(material)),materialPurpose='physics')
        self.materials=[]
        for cid in np.unique(self.ids):
            sel=np.flatnonzero(self.ids==cid);remap=np.full(len(self.ids),-1);remap[sel]=np.arange(len(sel));faces=remap[self.faces[np.all(self.ids[self.faces]==cid,axis=1)]];uv=self.uv[sel]
            pairs,ke,kd,rest,record=woven_springs(uv,c.stretch,c.shear,c.bend,c.spring_damping)
            flat=np.c_[uv,np.zeros(len(sel))];path=f'/World/TableclothPhysics/Cloth_{cid}'
            particleUtils.add_physx_particle_cloth_with_constraints(self.stage,path,system,Vt.Vec3fArray.FromNumpy(self.q0[sel].astype(np.float32)),[],Vt.Vec3fArray.FromNumpy(flat.astype(np.float32)),Vt.Vec3fArray.FromNumpy(np.zeros_like(flat,dtype=np.float32)),Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)),Vt.Vec2iArray.FromNumpy(pairs),ke.tolist(),kd.tolist(),rest.tolist(),self_collision=True,self_collision_filter=True,particle_group=int(cid))
            UsdPhysics.MassAPI.Apply(self.stage.GetPrimAtPath(path)).CreateMassAttr(float(c.density*np.prod(np.ptp(uv,axis=0))))
            self.paths.append(path);self.counts.append(len(sel));self.materials.append(record)
        self.authored=True
        return dict(paths=self.paths,materials=self.materials,attachment_count=0,calibrated=False,robot_modified=False)
    def bind(self,tensor_sim):
        if not self.authored:raise RuntimeError('Author before reset, bind after reset')
        self.views=[tensor_sim.create_particle_cloth_view(p) for p in self.paths]
        self.masses=[]
        for v,n in zip(self.views,self.counts):
            if v.count!=1 or v.max_particles_per_cloth!=n:raise RuntimeError('Native solver mesh count mismatch')
            mass=v.get_masses().detach().cpu().numpy().reshape(-1).copy()
            if np.any(mass<=0) or not np.isfinite(mass).all():raise RuntimeError('Cloth must have positive finite masses')
            self.masses.append(mass)
    def readback(self):
        if not self.views:raise RuntimeError('Bind native solver views first')
        q=[];qd=[]
        for v,mass in zip(self.views,self.masses):
            if not np.array_equal(v.get_masses().detach().cpu().numpy().reshape(-1),mass):raise RuntimeError('Particle mass changed during episode')
            q.append(v.get_positions().detach().cpu().numpy().reshape(-1,3).copy());qd.append(v.get_velocities().detach().cpu().numpy().reshape(-1,3).copy())
        q=np.concatenate(q);qd=np.concatenate(qd)
        if not np.isfinite(q).all() or not np.isfinite(qd).all():raise RuntimeError('Nonfinite native cloth state')
        return dict(cloth_q=q,cloth_qd=qd,cloth_id=self.ids.copy())
