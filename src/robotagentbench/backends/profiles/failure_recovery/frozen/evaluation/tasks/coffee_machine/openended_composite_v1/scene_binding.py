"""Read-only binding of the unified scene's actual coffee assets.

No initialization, USD edits, contact-report configuration, device construction,
or fallback contact facts. Geometry is NOT a live PhysX measurement.
"""
from pathlib import Path
import math
from .composite import Binding


class IncompleteScene(ValueError):
    pass


def one(items, role):
    if len(items) != 1:
        raise IncompleteScene(f'{role}: expected exactly one instance, got {len(items)}')
    return items[0]


def resolve(stage, manifest):
    from pxr import Usd, UsdPhysics

    objects = manifest['objects']
    wrappers = {}
    for role, predicate in (
        ('machine', lambda x: x == 'coffee_machine'),
        ('cup', lambda x: x.startswith('coffee_cup_')),
        ('bag', lambda x: x == 'coffee_bag'),
    ):
        wrappers[role] = one([r['path'] for r in objects if predicate(r['id'])], role)
        p = stage.GetPrimAtPath(wrappers[role])
        if not p or not p.IsActive():
            raise IncompleteScene('Inactive/missing wrapper: '+wrappers[role])

    def rigid_root(role):
        paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(wrappers[role]))
                 if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        path = one(paths, role+' rigid body')
        p = stage.GetPrimAtPath(path)
        if p.GetAttribute('physics:rigidBodyEnabled').Get() is False:
            raise IncompleteScene('Rigid body disabled: '+path)
        if p.GetAttribute('physics:kinematicEnabled').Get() is True:
            raise IncompleteScene('Manipulated object is kinematic: '+path)
        return path

    machine = one([str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(wrappers['machine']))
                   if stage.GetPrimAtPath(str(p.GetPath())+'/TaskAnchors/CupPlacement')
                   and stage.GetPrimAtPath(str(p.GetPath())+'/Joints/TopButtonPress')], 'machine contract root')
    roles = dict(machine=machine, cup=rigid_root('cup'), bag=rigid_root('bag'),
                 button=machine+'/Button', cup_support=machine+'/Body/Collision/CupSupport',
                 counter=manifest['supports']['coffee']['path'])
    for role, path in roles.items():
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsActive():
            raise IncompleteScene(f'{role}: missing active prim {path}')
    if not stage.GetPrimAtPath(roles['button']).HasAPI(UsdPhysics.RigidBodyAPI):
        raise IncompleteScene('Button has no rigid body')
    from openend.tasks.scene_requirements.surface_registry import contact_paths
    counter_paths = contact_paths(stage, manifest, roles['counter'])
    for role, path in [('cup_support', roles['cup_support'])]+[('counter', p) for p in counter_paths]:
        p = stage.GetPrimAtPath(path)
        if not p.HasAPI(UsdPhysics.CollisionAPI) or p.GetAttribute('physics:collisionEnabled').Get() is False:
            raise IncompleteScene('Missing enabled support collider: '+role)

    missing = []
    beans = [r['path'] for r in objects if r['id'] == 'coffee_beans']
    bean_bodies = []
    if not beans:
        missing.append('Unified scene has no coffee_beans binding')
    else:
        root = one(beans, 'coffee_beans group')
        p = stage.GetPrimAtPath(root)
        if not p or not p.IsActive():
            missing.append('Unified scene coffee_beans group missing/inactive')
        else:
            roles['beans'] = root
            bean_bodies = [str(x.GetPath()) for x in Usd.PrimRange(p)
                           if x.HasAPI(UsdPhysics.RigidBodyAPI)
                           and x.GetAttribute('physics:rigidBodyEnabled').Get() is not False
                           and x.GetAttribute('physics:kinematicEnabled').Get() is not True]
            if not bean_bodies:
                missing.append('No enabled dynamic beans in bound group')
    return dict(wrappers=wrappers, roles=roles, bean_bodies=bean_bodies, missing=missing, counter_paths=counter_paths)


class ReadOnlySceneBinding:
    """Attach to a stage already owned by the unified scene; never initialize it."""
    def __init__(self, stage, manifest):
        from pxr import UsdGeom
        if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise IncompleteScene('Expected existing metres/Z-up scene')
        self.stage = stage
        self.resolved = resolve(stage, manifest)
        self.roles = self.resolved['roles']
        self.counter_paths = self.resolved['counter_paths']
        # Reuse the existing read-only geometry algorithms with explicit paths.
        # Do NOT instantiate CoffeeReader: it creates device state and instruments USD.
        from evaluation.tasks.coffee_machine import vendor_producer
        from evaluation.tasks.coffee_machine.isaac_evidence import CupGeometry, BagGeometry
        self.contract = vendor_producer.load('machine_contract').read_contract(stage, self.roles['machine'])
        self.cup = CupGeometry(stage, self.roles['cup'])
        self.bag = BagGeometry(stage, self.roles['bag'])

    def task_binding(self, episode_id, instance_id):
        if self.resolved['missing']:
            raise IncompleteScene('; '.join(self.resolved['missing']))
        return Binding(episode_id, instance_id, tuple(sorted(self.roles.items())))

    def geometry_snapshot(self):
        """Static authored USD pose/geometry, explicitly not scoring evidence."""
        import numpy as np
        from pxr import UsdGeom, Gf
        from evaluation.tasks.coffee_machine.isaac_evidence import BAY_EXIT_LOCAL_Z
        cache = UsdGeom.XformCache()
        matrix = lambda path: cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
        machine = matrix(self.roles['machine'])
        inverse = machine.GetInverse()
        cup = matrix(self.roles['cup'])
        bag = matrix(self.roles['bag'])
        point = lambda m, p: m.Transform(Gf.Vec3d(*map(float, p)))
        origin = lambda m: point(m, (0, 0, 0))
        cp, bp = origin(cup), origin(bag)
        anchor = point(machine, self.contract.cup)
        outlet = point(machine, self.contract.outlet)
        local_outlet = point(cup.GetInverse(), outlet)
        outline = [point(inverse, point(cup, x))[2] for x in self.cup.outline]
        outline += [point(inverse, point(cup, c))[2]-r for c, r in self.cup.spheres]
        tilt = lambda m: math.degrees(math.acos(max(-1., min(1., m.TransformDir(Gf.Vec3d(0, 1, 0)).GetNormalized()[2]))))
        # Both reused geometry constructors reject scaled rigid-body frames.
        bean_positions = [origin(matrix(p)) for p in self.resolved['bean_bodies']]
        bag_local = np.asarray([point(bag.GetInverse(), p) for p in bean_positions]).reshape(-1, 3)
        return dict(source='authored_usd_static_geometry_not_live_physx',
                    cup_position_m=list(cp), bag_position_m=list(bp),
                    cup_tilt_deg=tilt(cup), bag_tilt_deg=tilt(bag),
                    anchor_world_m=list(anchor), outlet_world_m=list(outlet),
                    placement=dict(xy_error_m=math.hypot(cp[0]-anchor[0], cp[1]-anchor[1]),
                                   outlet_over_opening=math.hypot(local_outlet[0], local_outlet[2]) < self.cup.opening_radius,
                                   outside_bay=min(outline) > BAY_EXIT_LOCAL_Z),
                    beans_bound=len(bean_positions),
                    beans_geometrically_in_bag=int(self.bag.inside(bag_local).sum()) if len(bean_positions) else 0,
                    actual_support_contact=None, actual_grasp=None, device_dispensing=None,
                    scoring_evidence_available=False)

    def requirements(self):
        return list(self.resolved['missing']) + [
            'Unified scene must provide persistent coffee device and liquid accounting state',
            'Live PhysX contact/grasp/support collector and 20 Hz evidence bridge not connected',
            'Scene-owned physical settling acceptance not available in static preview',
            'Composite timeout/leave policy and explicit global registration still pending',
        ]
