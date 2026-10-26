"""Read-only admission: external human shares the existing CPU rigid-body world."""


def validate_cpu_scene(stage):
    if stage is None:
        raise ValueError('Native stage required for CPU physics verification')
    prim = stage.GetPrimAtPath('/World/PhysicsScene')
    if not prim:
        raise ValueError('Shared PhysicsScene missing')
    attr = prim.GetAttribute('physxScene:enableGPUDynamics')
    if not attr or attr.Get() is not False:
        raise ValueError('Human interference requires explicit CPU PhysX; no GPU dynamics fallback')
    return dict(physics_backend='cpu_physx', scene_pose_writes=0,
                physics_parameter_writes=0)


def validate_human_rigid_prims(prims):
    if not prims:
        raise ValueError('Human collision bodies are missing')
    for prim in prims:
        if not prim:
            raise ValueError('Missing human collision prim')
        schemas = tuple(map(str, prim.GetAppliedSchemas()))
        if any(any(token in schema.lower() for token in
                   ('deformable', 'particle', 'softbody', 'cloth')) for schema in schemas):
            raise ValueError('Human actor must use rigid bodies, not soft bodies: ' + str(prim.GetPath()))
        if 'PhysicsRigidBodyAPI' not in schemas:
            raise ValueError('Human body lacks rigid-body physics: ' + str(prim.GetPath()))
        enabled = prim.GetAttribute('physics:rigidBodyEnabled')
        if enabled and enabled.Get() is False:
            raise ValueError('Human rigid body is disabled')
    return dict(body_count=len(prims), soft_bodies=False)
