"""Post-brew physical liquid fixture. Authored once before physics, no emitter."""
import numpy as np
from pxr import Gf,PhysxSchema
from coffee_fluid import CoffeeFluid

def fill(stage,cup_matrix):
 fluid=CoffeeFluid(stage,'/World/PhysicsScene',(0,0,0))
 positions=[];s=fluid.spacing
 for y in np.arange(.012,.037,s):
  for x in np.arange(-.023,.023+s/2,s):
   for z in np.arange(-.023,.023+s/2,s):
    if x*x+z*z<.023**2:positions.append(Gf.Vec3f(*cup_matrix.Transform(Gf.Vec3d(float(x),float(y),float(z)))))
 fluid.sim_points.Set(positions);fluid.points.GetVelocitiesAttr().Set([Gf.Vec3f(0)]*len(positions));fluid.points.GetWidthsAttr().Set([s]*len(positions));fluid.count=len(positions)
 PhysxSchema.PhysxParticleIsosurfaceAPI(stage.GetPrimAtPath(fluid.system_path)).GetIsosurfaceEnabledAttr().Set(True)
 return fluid

def inspect(fluid,cup_matrix):
 inv=np.array(cup_matrix.GetInverse());p=fluid.positions();local=np.c_[p,np.ones(len(p))]@inv
 inside=(local[:,0]**2+local[:,2]**2<.04**2)&(local[:,1]>.002)&(local[:,1]<.095)
 return dict(count=len(p),inside=int(inside.sum()),fraction=float(inside.mean()) if len(p) else 0.)
