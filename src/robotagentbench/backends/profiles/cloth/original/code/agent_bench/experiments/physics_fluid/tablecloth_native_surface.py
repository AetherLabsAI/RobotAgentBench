"""Diagnostic native FEM surface material; no attachments or runtime state writes."""
from __future__ import annotations
import numpy as np


def create_surface(stage, path, points, uv, faces, total_mass, radius, args):
    from pxr import UsdGeom, UsdShade, PhysxSchema, Vt
    from omni.physx.scripts import deformableUtils
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
    if not deformableUtils.set_physics_surface_deformable_body(stage, path):
        raise RuntimeError('Native surface deformable schema unavailable')
    prim = mesh.GetPrim()
    flat = np.c_[uv, np.zeros(len(uv))]
    prim.GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(flat.astype(np.float32)))
    prim.GetAttribute('omniphysics:mass').Set(float(total_mass))
    prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI')
    parameters = dict(solverPositionIterationCount=args.iterations, selfCollision=True,
                      enableSpeculativeCCD=True, linearDamping=args.velocity_damping,
                      settlingDamping=0., settlingThreshold=0., sleepThreshold=0.,
                      collisionPairUpdateFrequency=args.surface_pair_updates,
                      collisionIterationMultiplier=args.surface_collision_multiplier)
    velocity_bound = getattr(args, 'surface_max_linear_velocity', None)
    if velocity_bound is not None:
        parameters['maxLinearVelocity'] = velocity_bound
    for key, value in parameters.items():
        attr = prim.GetAttribute('physxDeformableBody:'+key)
        if not attr:
            raise RuntimeError('Missing native surface attribute '+key)
        attr.Set(value)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    rest_offset = radius if args.surface_rest_offset is None else args.surface_rest_offset
    contact_offset = radius*1.5 if args.surface_contact_offset is None else args.surface_contact_offset
    if not np.isfinite(rest_offset) or rest_offset <= 0:
        raise ValueError('Surface collision rest offset must be positive')
    if not np.isfinite(contact_offset) or contact_offset <= rest_offset:
        raise ValueError('Surface contact generation distance must exceed rest offset')
    contact.CreateContactOffsetAttr(contact_offset)
    contact.CreateRestOffsetAttr(rest_offset)
    material_path = path+'_Material'
    if not deformableUtils.add_surface_deformable_material(stage, material_path,
            density=total_mass/(np.prod(np.ptp(uv,axis=0))*args.surface_thickness),
            dynamic_friction=args.surface_friction, youngs_modulus=args.youngs_modulus,
            poissons_ratio=.3, surface_thickness=args.surface_thickness,
            surface_bend_stiffness=args.surface_bend_modulus):
        raise RuntimeError('Native surface material schema unavailable')
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    mp = material.GetPrim()
    mp.ApplyAPI('PhysxSurfaceDeformableMaterialAPI')
    mp.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(args.elasticity_damping)
    mp.GetAttribute('physxDeformableMaterial:bendDamping').Set(args.bend_damping)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    # Tensor API exposes positions/velocities but not actual nodal masses. Keep
    # these rest-area estimates explicitly labelled, never call them readback.
    area = .5*np.linalg.norm(np.cross(flat[faces[:,1]]-flat[faces[:,0]],flat[faces[:,2]]-flat[faces[:,0]]),axis=1)
    mass = np.zeros(len(points))
    np.add.at(mass,faces.ravel(),np.repeat(area/3,3))
    mass *= total_mass/mass.sum()
    if (mass<=0).any():raise ValueError('Nonpositive rest-area nodal mass')
    return mass, dict(model='native_FEM_surface', youngs_modulus_Pa=args.youngs_modulus,
                      thickness_m=args.surface_thickness, surface_bend_modulus_Pa=args.surface_bend_modulus,
                      effective_bend_scale_Nm=args.surface_bend_modulus*args.surface_thickness**3,
                      elasticity_damping_per_s=args.elasticity_damping,
                      bend_damping_per_s=args.bend_damping,
                      contact_offset_m=contact_offset, rest_offset_m=rest_offset,
                      native_body_parameters=parameters, calibrated=False,
                      mass_source='positive authored total; nodal rest-area estimates, not GPU mass readback')


class SurfaceReadback:
    def __init__(self, view, expected):
        if view.count != 1 or view.max_simulation_nodes_per_body != expected:
            raise RuntimeError('Native FEM view count/topology mismatch')
        self.view = view
        self.count = view.count
        self.max_particles_per_cloth = expected

    def get_positions(self):
        return self.view.get_simulation_nodal_positions()

    def get_velocities(self):
        return self.view.get_simulation_nodal_velocities()
