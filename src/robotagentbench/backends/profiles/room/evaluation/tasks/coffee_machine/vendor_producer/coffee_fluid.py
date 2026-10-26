"""GPU PBD liquid emitter. Create before physics starts, emit after each step.

Uses Isaac 5.1 bundled particleUtils; live append follows FluidBallEmitterDemo.
Existing liquid is never respawned, moved to the cup, or deleted on stop.
"""
import math
import numpy as np
from pxr import Gf, Sdf, Vt, UsdGeom, UsdShade, PhysxSchema
from omni.physx.scripts import particleUtils, physicsUtils


class CoffeeFluid:
    def __init__(self, stage, physics_scene_path, outlet_local, root='/World/CoffeeLiquid', max_particles=16000):
        self.outlet_local = tuple(outlet_local)
        self.stage = stage
        self.root = root
        self.max_particles = max_particles
        self.count = 0
        self.accumulator = 0.
        self.emission_period = 1/120
        self.rest = .0008
        self.spacing = 2*self.rest
        self.particle_volume_ml = self.spacing**3 * 1e6
        self.capacity_reached = False
        self.system_path = root+'/System'
        system = particleUtils.add_physx_particle_system(
            stage, self.system_path, simulation_owner=Sdf.Path(physics_scene_path),
            contact_offset=.00135, rest_offset=.0008,
            particle_contact_offset=.00135, solid_rest_offset=.001,
            fluid_rest_offset=self.rest, solver_position_iterations=8,
            max_neighborhood=96, enable_ccd=True, max_velocity=3.)
        material_path = root+'/CoffeeMaterial'
        particleUtils.add_pbd_particle_material(stage, material_path, density=1000,
            cohesion=.01, viscosity=.0000017, surface_tension=.0074,
            friction=.1, damping=0., vorticity_confinement=0.)
        physicsUtils.add_physics_material_to_prim(stage,system.GetPrim(),material_path)
        mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
        sh = UsdShade.Shader.Define(stage,material_path+'/Surface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.095,.025,.006))
        sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.16)
        sh.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.33)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(mat)
        particleUtils.add_physx_particle_isosurface(stage,self.system_path,enabled=False,
            grid_spacing=.001, surface_distance=.0012, max_vertices=300000,
            max_triangles=600000,max_subgrids=4096,num_mesh_smoothing_passes=2,
            num_mesh_normal_smoothing_passes=2)
        self.points = particleUtils.add_physx_particleset_points(stage,root+'/Particles',
            [],[],[],self.system_path,self_collision=True,fluid=True,
            particle_group=0,particle_mass=0.,density=1000.)
        self.points.GetPrim().CreateAttribute('physxParticle:maxParticles',Sdf.ValueTypeNames.Int).Set(max_particles)
        # With isosurface-only rendering the Points attribute contains physical
        # positions. Do not mix the separate smoothing simulationPoints path
        # with isosurface updates; its readback differs in PhysX 107.3.
        self.sim_points = self.points.GetPointsAttr()
        self.points.CreateDisplayColorAttr().Set([Gf.Vec3f(.095,.025,.006)])
        UsdShade.MaterialBindingAPI.Apply(self.points.GetPrim()).Bind(mat)

    @staticmethod
    def extend(attr, values):
        old = attr.Get()
        attr.Set(list(old) + values if old is not None else values)

    def update(self, dt, dispensing, machine_to_world):
        if not dispensing:
            self.accumulator = 0.
            return 0
        self.accumulator += dt
        new_positions = []
        while self.accumulator + 1e-9 >= self.emission_period:
            self.accumulator -= self.emission_period
            if self.count + len(new_positions) + 7 > self.max_particles:
                self.capacity_reached = True
                break
            for i in range(7):
                dx = 0 if i==0 else self.spacing*math.cos((i-1)*math.pi/3)
                dz = 0 if i==0 else self.spacing*math.sin((i-1)*math.pi/3)
                p = machine_to_world.Transform(Gf.Vec3d(self.outlet_local[0]+dx,self.outlet_local[1]-.007,self.outlet_local[2]+dz))
                new_positions.append(Gf.Vec3f(*p))
        if not new_positions:
            return 0
        direction = machine_to_world.TransformDir(Gf.Vec3d(0,-.192,0))
        n=len(new_positions)
        self.extend(self.sim_points,new_positions)
        self.extend(self.points.GetVelocitiesAttr(),[Gf.Vec3f(*direction)]*n)
        self.extend(self.points.GetWidthsAttr(),[self.spacing]*n)
        self.count += n
        # An empty isosurface can issue a zero-work CUDA launch in PhysX 107.3.
        # Enable reconstruction only after the first real emission exists.
        PhysxSchema.PhysxParticleIsosurfaceAPI(self.stage.GetPrimAtPath(self.system_path)).GetIsosurfaceEnabledAttr().Set(True)
        return n

    def positions(self):
        value = self.sim_points.Get()
        return np.asarray(value,dtype=float).reshape((-1,3))

