"""Current USD joint frames, numerical FK/IK, and measured gripper geometry.

Joint commands are kinematic targets, not a claim of calibrated motor dynamics.
The exported JSON is numeric asset metadata; it is not a task registry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = position
    return result


def export_robot(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    joints, loops = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        # Spherical joints close the gripper's four-bar loops, rather than
        # supplying another parent in the FK spanning tree.
        if prim.IsA(UsdPhysics.SphericalJoint) and parent and child:
            loops.append(dict(parent=parent[0].name,child=child[0].name,
                anchor0=list(joint.GetLocalPos0Attr().Get()),anchor1=list(joint.GetLocalPos1Attr().Get())))
        if not parent or not child or prim.IsA(UsdPhysics.SphericalJoint):
            continue
        entry = dict(name=prim.GetName(), parent=parent[0].name, child=child[0].name,
                     kind=prim.GetTypeName(), axis=prim.GetAttribute("physics:axis").Get() or "Z")
        for end in (0, 1):
            p = prim.GetAttribute(f"physics:localPos{end}").Get()
            q = prim.GetAttribute(f"physics:localRot{end}").Get()
            entry[f"T{end}"] = transform(list(p), [*q.GetImaginary(), q.GetReal()]).tolist()
        if entry["kind"] != "PhysicsFixedJoint":
            limits = [prim.GetAttribute(f"physics:{key}Limit").Get() for key in ("lower", "upper")]
            scale = np.pi / 180 if prim.IsA(UsdPhysics.RevoluteJoint) else 1
            entry["limits"] = [float(x) * scale for x in limits]
        joints.append(entry)
    from tablecloth_robot_collisions import export_nonhand_collisions
    links = {j['parent'] for j in joints} | {j['child'] for j in joints}
    return dict(schema="real2sim.tablecloth_robot_kinematics.v1", asset=str(Path(path).resolve()),
                joints=joints, loops=loops, nonhand_collision_shapes=export_nonhand_collisions(stage, links))


class Robot:
    def __init__(self, metadata, station):
        self.metadata = metadata
        self.joints = metadata["joints"]
        self.by_name = {j["name"]: j for j in self.joints}
        self.T0 = {j["name"]: np.asarray(j["T0"]) for j in self.joints}
        self.T1_inv = {j["name"]: np.linalg.inv(j["T1"]) for j in self.joints}
        self.base = transform([station[0], station[1], 0], Rotation.from_euler("z", station[2], degrees=True).as_quat())
        self.names = [j["name"] for j in self.joints if j["kind"] != "PhysicsFixedJoint"]

    def fk(self, values, jacobian=False):
        poses, axes = {"base_link": self.base}, {}
        for j in self.joints:
            name = j["name"]
            parent = poses[j["parent"]] @ self.T0[name]
            dof = np.eye(3)["XYZ".index(j["axis"])]
            q = values.get(name, 0.0)
            motion = np.eye(4)
            if j["kind"] == "PhysicsRevoluteJoint":
                motion[:3, :3] = Rotation.from_rotvec(dof * q).as_matrix()
            elif j["kind"] == "PhysicsPrismaticJoint":
                motion[:3, 3] = dof * q
            poses[j["child"]] = parent @ motion @ self.T1_inv[name]
            if jacobian:
                axes[name] = (parent[:3, 3], parent[:3, :3] @ dof)
        return (poses, axes) if jacobian else poses

    @staticmethod
    def arm_names(side):
        prefix = "2" if side == "L" else "6"
        return [f"idx{prefix}{k}_arm_{side.lower()}_joint{k}" for k in range(1, 8)]

    def solve(self, values, side, target, local_point, orientation=None, max_steps=50):
        """Damped Jacobian IK with physical USD limits and previous-pose seed."""
        values = values.copy()
        names = self.arm_names(side)
        link = f"gripper_{side.lower()}_base_link"
        for _ in range(max_steps):
            poses, axes = self.fk(values, jacobian=True)
            hand = poses[link]
            point = hand[:3, :3] @ local_point + hand[:3, 3]
            error = target - point
            jac = np.stack([np.cross(axes[n][1], point - axes[n][0]) for n in names], axis=1)
            if orientation is not None:
                rot_error = Rotation.from_matrix(orientation @ hand[:3, :3].T).as_rotvec()
                error = np.r_[error, 0.12 * rot_error]
                jac = np.vstack((jac, 0.12 * np.stack([axes[n][1] for n in names], axis=1)))
            if np.linalg.norm(error) < 0.0003:
                break
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.008**2 * np.eye(len(error)), error)
            dq *= min(1.0, 0.12 / max(np.max(np.abs(dq)), 1e-12))
            for n, delta in zip(names, dq):
                lo, hi = self.by_name[n]["limits"]
                values[n] = float(np.clip(values.get(n, 0.0) + delta, lo + 1e-5, hi - 1e-5))
        hand = self.fk(values)[link]
        actual = hand[:3, :3] @ local_point + hand[:3, 3]
        return values, float(np.linalg.norm(actual - target))

    def reference_values(self, trajectory, frame):
        values = {}
        for side, offset in (("L", 0), ("R", 7)):
            values.update(zip(self.arm_names(side), trajectory["state_arm"][frame, offset:offset + 7].tolist()))
        values.update(idx01_body_joint1=float(trajectory["waist"][frame, 1]),
                      idx02_body_joint2=float(trajectory["waist"][frame, 0]),
                      idx11_head_joint1=float(trajectory["head"][frame, 0]),
                      idx12_head_joint2=float(trajectory["head"][frame, 1]))
        return values

    def close_loops(self,values):
        """Resolve the passive link2 follower, including the collider-free one."""
        values=values.copy()
        poses=self.fk(values)
        for loop in self.metadata.get("loops",[]):
            joint=next(j for j in self.joints if j["child"]==loop["child"])
            name=joint["name"]
            frame=poses[joint["parent"]]@self.T0[name]
            target=np.linalg.inv(frame)@poses[loop["parent"]]@np.r_[loop["anchor0"],1.]
            local=self.T1_inv[name]@np.r_[loop["anchor1"],1.]
            angle=np.arctan2(target[1],target[0])-np.arctan2(local[1],local[0])
            angle=(angle+np.pi)%(2*np.pi)-np.pi
            values[name]=float(np.clip(angle,*joint["limits"]))
        return values

    def calibrate_loops(self, values):
        """Fit passive joints to the CAD loop anchors without changing motors."""
        values=self.close_loops(values)
        for loop in self.metadata.get("loops",[]):
            joint0=next(j for j in self.joints if j["child"]==loop["child"])
            joint4=next(j for j in self.joints if j["child"]==joint0["parent"])
            joint3=next(j for j in self.joints if j["child"]==joint4["parent"])
            names=[j["name"] for j in (joint3,joint4,joint0)]
            limits=np.array([self.by_name[n]["limits"] for n in names])
            prior=np.array([values.get(n,0.) for n in names])
            def residual(x):
                poses=self.fk(values|dict(zip(names,x)))
                delta=(poses[loop["parent"]]@np.r_[loop["anchor0"],1.]-poses[loop["child"]]@np.r_[loop["anchor1"],1.])[:3]
                return np.r_[delta,(x-prior)*1e-5]
            result=least_squares(residual,np.clip(prior,limits[:,0]+1e-8,limits[:,1]-1e-8),
                bounds=(limits[:,0],limits[:,1]),gtol=1e-12,xtol=1e-12,ftol=1e-12,max_nfev=100)
            if np.linalg.norm(residual(result.x)[:3])>.0001:
                raise ValueError(f"Gripper CAD loop did not close: {loop['child']}")
            values.update(zip(names,result.x.tolist()))
        return values

    def nominal_values(self):
        values={}
        arm=np.array([-1.07415,.61099,.27946,-1.28387,.73038,1.49527,-.18755])
        for side,sign in (("L",1),("R",-1)):
            values.update(zip(self.arm_names(side),(arm*sign).tolist()))
        values.update(idx01_body_joint1=.02,idx02_body_joint2=np.pi/6,idx11_head_joint1=0.,idx12_head_joint2=.43634)
        return values


def export_asset_gripper(metadata):
    """Build runtime calibration directly from the current USD; no real data."""
    from pxr import Usd,UsdGeom,Gf
    from tablecloth_scene_binding import source_record
    stage=Usd.Stage.Open(metadata["asset"])
    root=stage.GetDefaultPrim()
    robot=Robot(metadata,[0,0,0])
    payload={"grip":np.linspace(0,1,41),"seed_json":np.asarray(json.dumps(robot.nominal_values()))}
    links=[]
    for child in root.GetChildren():
        if not child.GetName().startswith("gripper_"):continue
        points,triangles=[],[]
        inverse=UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        for prim in Usd.PrimRange(child):
            if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):continue
            mesh=UsdGeom.Mesh(prim)
            world=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            first=len(points)
            points.extend([list(inverse.Transform(world.Transform(Gf.Vec3d(*p)))) for p in mesh.GetPointsAttr().Get()])
            ids=list(mesh.GetFaceVertexIndicesAttr().Get());cursor=0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face=ids[cursor:cursor+count];cursor+=count
                triangles.extend([[first+face[0],first+face[j],first+face[j+1]] for j in range(1,count-1)])
        if not points:continue
        name=child.GetName();links.append(name)
        payload[f"points_{name}"]=np.asarray(points,dtype=np.float32)
        payload[f"triangles_{name}"]=np.asarray(triangles,dtype=np.int32)
    for index,g in enumerate(payload["grip"]):
        values={j["name"]:0. for j in robot.joints if "gripper_" in j["name"] and j["kind"]!="PhysicsFixedJoint"}
        for name in values:
            if name.endswith("joint1"):values[name]=(-1 if "inner" in name else 1)*.75*(1-g)
        values=robot.calibrate_loops(values)
        for name,value in values.items():
            payload.setdefault(f"joint_{name}",np.zeros(len(payload["grip"])))[index]=value
    payload.update(links=np.asarray(links),schema=np.asarray("real2sim.tablecloth_gripper_calibration.v1"),
        source_record_json=np.asarray(json.dumps(source_record())),calibration_method=np.asarray("current_USD_CAD_closed_loop_no_real_episode"))
    return payload


def export_gripper_profile(metadata, tracks, trajectory):
    """Reduce an Isaac joint sweep/measurement to an asset calibration table.

    It contains no task/episode gate. Runtime policies need neither the sweep's
    arm motion nor a real recording. Passive four-bar angles are measured, not
    guessed from a URDF that omits the closed-loop constraints.
    """
    robot = Robot(metadata, tracks["robot_station_xyyaw"])
    link_keys = {}
    for side in ("L", "R"):
        link_keys[f"gripper_{side.lower()}_base_link"] = f"palm{side}"
        for half in ("inner", "outer"):
            for k in range(1, 5):
                key = f"{half}{k if k < 4 else ''}{side}"
                if key in tracks:
                    link_keys[f"gripper_{side.lower()}_{half}_link{k}"] = key
    payload = {"grip": np.linspace(0, 1, 21)}
    all_values = {}
    for j in robot.joints:
        if j["parent"] not in link_keys or j["child"] not in link_keys:
            continue
        a, b = link_keys[j["parent"]], link_keys[j["child"]]
        qa, qb = tracks[f"quat_{a}"], tracks[f"quat_{b}"]
        ra, rb = Rotation.from_quat(qa).as_matrix(), Rotation.from_quat(qb).as_matrix()
        rotation = np.asarray(j["T0"])[:3, :3].T @ (ra.transpose(0, 2, 1) @ rb) @ np.asarray(j["T1"])[:3, :3]
        values = Rotation.from_matrix(rotation).as_rotvec()[:, "XYZ".index(j["axis"])]
        side = "l" if "_l_" in j["name"] else "r"
        grip = trajectory[f"action_{side}grip"]
        # Median of nearby samples removes dynamics noise from the mechanism
        # calibration. Active joints retain the explicitly commanded angle.
        curve = []
        for g in payload["grip"]:
            nearest = np.argsort(np.abs(grip - g))[:max(3, min(15, len(grip)))]
            curve.append(float(np.median(values[nearest])))
        if j["name"].endswith("joint1"):
            sign = -1 if "inner" in j["name"] else 1
            curve = sign * .75 * (1 - payload["grip"])
        payload[f"joint_{j['name']}"] = np.asarray(curve)
    for link, key in link_keys.items():
        payload[f"points_{link}"] = tracks[f"collisionMeshPoints_{key}"]
        payload[f"triangles_{link}"] = tracks[f"collisionMeshTriangles_{key}"]
    for index,g in enumerate(payload["grip"]):
        values={k[6:]:float(v[index]) for k,v in payload.items() if k.startswith("joint_")}
        corrected=robot.calibrate_loops(values)
        for name,value in corrected.items():
            if f"joint_{name}" not in payload:payload[f"joint_{name}"]=np.zeros(len(payload["grip"]))
            payload[f"joint_{name}"][index]=value
    payload["links"] = np.asarray(list(link_keys))
    payload["schema"] = np.asarray("real2sim.tablecloth_gripper_calibration.v1")
    payload["source_record_json"] = tracks["source_record_json"]
    payload["seed_json"] = np.asarray(json.dumps(robot.reference_values(trajectory, 0)))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracks")
    parser.add_argument("--trajectory")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = export_robot(args.asset)
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.tracks:
        with np.load(args.tracks, allow_pickle=False) as tracks, np.load(args.trajectory, allow_pickle=False) as trajectory:
            np.savez_compressed(output.with_suffix(".npz"), **export_gripper_profile(metadata, tracks, trajectory))
    else:
        np.savez_compressed(output.with_suffix(".npz"), **export_asset_gripper(metadata))


if __name__ == "__main__":
    main()
