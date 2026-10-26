"""Read-only USD joint-chain FK used only for pre-episode pose authoring."""


def rigid(translation, quaternion):
    from pxr import Gf
    value = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(quaternion)))
    value.SetTranslateOnly(Gf.Vec3d(*translation))
    return value


def body_transform(stage, body_path, joints, robot_world=None):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath('/World/Robot')
    cache = UsdGeom.XformCache()
    old_root = cache.GetLocalToWorldTransform(root)
    world = old_root if robot_world is None else robot_world
    # A child can have a spherical closure constraint as well as its primary
    # articulation chain. Hand centre has an unambiguous fixed/revolute chain.
    parents = {}
    for p in Usd.PrimRange(root):
        if not (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)
                or p.IsA(UsdPhysics.FixedJoint)):
            continue
        a, b = p.GetRelationship('physics:body0').GetTargets(), p.GetRelationship('physics:body1').GetTargets()
        if len(a) == len(b) == 1:
            if str(b[0]) in parents:
                raise ValueError('Ambiguous FK chain: ' + str(b[0]))
            parents[str(b[0])] = (p, str(a[0]))
    memo = {}
    visiting = set()
    def solve(path):
        if path in memo:
            return memo[path]
        if path in visiting:
            raise ValueError('Cyclic FK chain')
        visiting.add(path)
        if path not in parents:
            if path != '/World/Robot/base_link':
                raise ValueError('Unknown FK anchor ' + path)
            result = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)) * old_root.GetInverse() * world
        else:
            p, parent = parents[path]
            a = rigid(p.GetAttribute('physics:localPos0').Get(), p.GetAttribute('physics:localRot0').Get())
            b = rigid(p.GetAttribute('physics:localPos1').Get(), p.GetAttribute('physics:localRot1').Get())
            motion = Gf.Matrix4d(1)
            if not p.IsA(UsdPhysics.FixedJoint):
                if p.GetName() not in joints:
                    raise ValueError('Missing FK coordinate ' + p.GetName())
                value = joints[p.GetName()]
                axis = {'X': Gf.Vec3d(1,0,0), 'Y': Gf.Vec3d(0,1,0), 'Z': Gf.Vec3d(0,0,1)}[p.GetAttribute('physics:axis').Get()]
                if p.IsA(UsdPhysics.PrismaticJoint):
                    motion.SetTranslate(axis * value)
                else:
                    import math
                    motion.SetRotate(Gf.Rotation(axis, math.degrees(value)))
            result = b.GetInverse() * motion * a * solve(parent)
        visiting.remove(path)
        memo[path] = result
        return result
    return solve(body_path)
