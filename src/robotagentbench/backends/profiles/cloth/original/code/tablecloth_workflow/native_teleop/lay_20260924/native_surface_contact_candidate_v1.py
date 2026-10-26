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
    fold_degrees = float(getattr(args, 'clean_rest_fold_deg', 0.)) if str(path).endswith('Cloth_1') else 0.
    bend_modulus=(float(args.clean_surface_bend_modulus) if str(path).endswith('Cloth_1') and getattr(args,'clean_surface_bend_modulus',None) is not None else args.surface_bend_modulus)
    if bend_modulus<=0:raise ValueError('Positive bending modulus required')
    if not 0 <= fold_degrees <= 175: raise ValueError('Rest crease rotation out of range')
    if fold_degrees:
        # An isometric pre-existing crease, set only before simulation.
        # This is a constitutive rest-curvature candidate, never an attachment.
        # Constitutive pre-existing crease, centered on the actual staggered fold.
        # Rest metric remains isometric; finite bend forces only, authored before physics.
        levels, inverse=np.unique(uv[:,1],return_inverse=True)
        mid=levels[np.argmin(abs(levels-((levels.min()+levels.max())/2+.04)))];crease_radius=.002
        signed=levels-mid;distance=np.abs(signed);theta=np.minimum(distance/crease_radius,np.pi/2)
        lateral=crease_radius*np.sin(theta)+np.maximum(distance-np.pi*crease_radius/2,0.)
        height=np.sign(signed)*crease_radius*(1-np.cos(theta))
        folded_curve=np.column_stack((-lateral,height));segments=np.diff(folded_curve,axis=0)
        actual_angles=np.unwrap(np.arctan2(segments[:,1],segments[:,0]))
        center=(actual_angles[0]+actual_angles[-1])/2
        rest_angles=center+(actual_angles-center)*(fold_degrees/180.)
        ds=np.diff(levels);yz=np.vstack((np.zeros(2),np.cumsum(np.column_stack((ds*np.cos(rest_angles),ds*np.sin(rest_angles))),axis=0)))
        flat=np.column_stack((uv[:,0],yz[inverse]))
    prim.GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(flat.astype(np.float32)))
    rest_method=prim.GetAttribute('omniphysics:restBendAnglesDefault')
    if fold_degrees:
        if not rest_method: raise RuntimeError('Rest bend angle schema missing')
        rest_method.Set('restShapeDefault')
        if rest_method.Get()!='restShapeDefault': raise RuntimeError('Rest bend angle authoring failed')
    print('Rest bending mode',path,rest_method.Get(),flush=True)
    prim.GetAttribute('omniphysics:mass').Set(float(total_mass))
    prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI')
    parameters = dict(solverPositionIterationCount=args.iterations, selfCollision=True,
                      enableSpeculativeCCD=True, linearDamping=args.velocity_damping,
                      settlingDamping=0., settlingThreshold=0., sleepThreshold=0.,
                      collisionPairUpdateFrequency=args.surface_pair_updates, collisionIterationMultiplier=args.surface_collision_multiplier)
    if args.surface_self_filter_distance is not None: parameters['selfCollisionFilterDistance']=args.surface_self_filter_distance
    if args.surface_max_linear_velocity is not None: parameters['maxLinearVelocity']=args.surface_max_linear_velocity
    for key, value in parameters.items():
        attr = prim.GetAttribute('physxDeformableBody:'+key)
        if not attr:
            raise RuntimeError('Missing native surface attribute '+key)
        attr.Set(value)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact_offset = radius*1.5 if args.surface_contact_offset is None else args.surface_contact_offset
    if not np.isfinite(contact_offset) or contact_offset <= radius:
        raise ValueError('Surface contact generation distance must exceed rest offset')
    contact.CreateContactOffsetAttr(contact_offset)
    rest_offset = radius if args.surface_rest_offset is None else args.surface_rest_offset
    if not 0 < rest_offset < contact_offset: raise ValueError("Invalid physical surface rest offset")
    contact.CreateRestOffsetAttr(rest_offset)
    material_path = path+'_Material'
    if not deformableUtils.add_surface_deformable_material(stage, material_path,
            density=total_mass/(np.prod(np.ptp(uv,axis=0))*args.surface_thickness),
            dynamic_friction=args.surface_friction, youngs_modulus=args.youngs_modulus,
            poissons_ratio=.3, surface_thickness=args.surface_thickness,
            surface_bend_stiffness=bend_modulus):
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
    return mass, dict(model='native_FEM_surface', rest_crease_source='finite150deg_rest_dihedrals_at_actual_staggered_fold, not fixed constraints' if fold_degrees else 'flat', rest_crease_rotation_deg=fold_degrees, rest_crease_calibrated=False, authored_rest_bend_mode=rest_method.Get(), youngs_modulus_Pa=args.youngs_modulus,
                      thickness_m=args.surface_thickness, surface_bend_modulus_Pa=bend_modulus,
                      effective_bend_scale_Nm=bend_modulus*args.surface_thickness**3,
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
