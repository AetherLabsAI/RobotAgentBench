"""Read L1 named checkpoints and express poses relative to the microwave.

This is source preparation, NOT a physically accepted reset or an action
adapter. It never writes USD or sets physics parameters. Measured q/dq and
native targets remain separate; source clocks are retained, not resampled.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from .variants import variant_for

SOURCE_ROLES = {
    "microwave": "/World/Assets/DiningTableMicrowave/Asset/body",
    "door": "/World/Assets/DiningTableMicrowave/Asset/door",
    "plate": "/World/Assets/MicrowaveTakePlate/Asset",
    "payload": "/World/Assets/BeefTray/Asset",
}

def numeric(value, shape, label):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fi" or not np.isfinite(a).all():
        raise ValueError("Invalid numeric " + label)
    return a.astype(float, copy=True)

def names(value, label):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "US" or not len(a):
        raise ValueError("Expected named " + label)
    result = a.tolist()
    if len(set(result)) != len(result) or any(not x for x in result):
        raise ValueError("Duplicate or empty " + label)
    return tuple(result)

def pose_matrix(xyz_xyzw):
    v = numeric(xyz_xyzw, (7,), "pose")
    q = v[3:]
    norm = np.linalg.norm(q)
    if not norm > 0:
        raise ValueError("Zero quaternion")
    # Normalize representational roundoff; no pose displacement or scale.
    x, y, z, w = q / norm
    t = np.eye(4)
    t[:3,:3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3,3] = v[:3]
    return t

def rigid_matrix(value):
    t = numeric(value, (4,4), "rigid transform")
    # Numerical representation checks, not task acceptance thresholds.
    if (not np.allclose(t[3], [0,0,0,1], rtol=0, atol=1e-10)
        or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), rtol=0, atol=1e-8)
        or not np.isclose(np.linalg.det(t[:3,:3]), 1, rtol=0, atol=1e-8)):
        raise ValueError("Expected rigid transform without scale/reflection")
    return t

@dataclass
class RecordedCheckpoint:
    variant: str
    episode: str
    positions: dict
    velocities: dict
    hold_targets: dict
    robot_world: np.ndarray
    body_poses: dict
    body_velocities: dict
    metadata: dict
    source_manifest: dict

    @property
    def roles(self):
        variant_for(self.variant)
        return dict(SOURCE_ROLES, payload=("/World/Assets/MwFoodbox/Asset"
                    if self.variant == "leftover_box" else SOURCE_ROLES['payload']))

    def relative_frames(self):
        mw_inv = np.linalg.inv(pose_matrix(self.body_poses[self.roles["microwave"]]))
        result = {"robot": mw_inv @ self.robot_world}
        result.update({role: mw_inv @ pose_matrix(self.body_poses[path])
                       for role, path in self.roles.items()})
        return result

    def placement_plan(self, microwave_world):
        """Pure coordinate calculation, no scene authoring or random sampling.

        Returns only this task's robot/appliance/payload/plate roles.
        Unrelated recorded room objects are deliberately not relocated.
        """
        mw = rigid_matrix(microwave_world)
        return {role: mw @ relative for role, relative in self.relative_frames().items()}

def load_checkpoint(directory, payload_variant):
    selected = variant_for(payload_variant)
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory/"source_manifest.json").read_text())
    if manifest.get("payload_variant") != selected.name:
        raise ValueError("Checkpoint belongs to a different payload variant")
    allowed_schema = ("arena.microwave_put.source_candidate.v1" if selected.name == "leftover_box"
                      else "arena.microwave_put.source_checkpoint.v1")
    if manifest.get("schema") != allowed_schema:
        raise ValueError("Unknown source manifest schema")
    entries = manifest["files"]
    required = {"initial_state.npz", "control_state_60hz.npz", "sim_info.json",
                "robot_and_camera_parameters.json", "initial_grip_detail.json"}
    listed = [item["local_name"] for item in entries]
    if len(set(listed)) != len(listed) or not required.issubset(listed):
        raise ValueError("Incomplete or duplicate source manifest")
    if manifest["payload_variant"] != selected.name:
        raise ValueError("Checkpoint belongs to a different payload variant")
    if manifest["api_id"] != selected.api_id or manifest["benchmark_id"] != selected.benchmark_id:
        raise ValueError("Manifest task ID does not match the selected variant")
    for item in manifest["files"]:
        path = (directory/item["local_name"]).resolve(strict=True)
        if not path.is_relative_to(directory):
            raise ValueError("Source escaped checkpoint directory")
        if sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source hash mismatch: " + item["local_name"])
    meta = json.loads((directory/"robot_and_camera_parameters.json").read_text())
    info = json.loads((directory/"sim_info.json").read_text())
    detail = json.loads((directory/"initial_grip_detail.json").read_text())
    payload = ("/World/Assets/MwFoodbox/Asset" if selected.name == "leftover_box"
               else SOURCE_ROLES['payload'])
    object_kind = "foodbox" if selected.name == "leftover_box" else "beef"
    if (detail["config"]["object_kind"] != object_kind
        or detail["objpath"] != payload
        or info["episode"] != manifest["episode"]):
        raise ValueError("Source task/object identity mismatch")
    if meta["planning_frame"] != ("microwave-relative canonical coordinates in diagnostic "
        "samples only; physics_60hz body poses and velocities are world coordinates"):
        raise ValueError("Unknown source coordinate convention")
    for key in ("physics_hz","command_hz","training_hz"):
        rate = meta[key]
        if isinstance(rate,bool) or not isinstance(rate,(float,int)) or not np.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid recorded rate")
    with np.load(directory/"initial_state.npz", allow_pickle=False) as z:
        joints = names(z["dof_names"], "DOFs")
        paths = names(z["rigid_body_paths"], "bodies")
        if set(joints) != set(meta["dof_names"]) or set(paths) != set(meta["rigid_body_paths"]):
            raise ValueError("NPZ/metadata named fields disagree")
        arrays = {k:numeric(z[k],(len(joints),),k) for k in ("q","dq","native_target")}
        poses = numeric(z["body_pose_xyzw"], (len(paths),7), "body poses")
        velocities = numeric(z["body_velocity"], (len(paths),6), "body velocities")
    for path in dict(SOURCE_ROLES, payload=payload).values():
        if path not in paths:
            raise ValueError("Missing task body: " + path)
    for pose in poses:
        pose_matrix(pose)
    bp, bq_wxyz = meta["robot_world_pose"]
    bp = numeric(bp,(3,),"robot position")
    bq = numeric(bq_wxyz,(4,),"robot quaternion wxyz")
    robot = pose_matrix(np.r_[bp,bq[1:],bq[0]])
    return RecordedCheckpoint(selected.name, info["episode"],
        dict(zip(joints,arrays["q"].tolist())), dict(zip(joints,arrays["dq"].tolist())),
        dict(zip(joints,arrays["native_target"].tolist())), robot,
        dict(zip(paths,poses.tolist())), dict(zip(paths,velocities.tolist())),
        meta, manifest)

def inspect_recording(directory, checkpoint):
    """Check full native recording integrity; do not decimate 60 Hz to 30 Hz."""
    with np.load(Path(directory)/"control_state_60hz.npz",allow_pickle=False) as z:
        joints = names(z["dof_names"],"recording DOFs")
        if set(joints) != set(checkpoint.positions):
            raise ValueError("Checkpoint/recording joint mismatch")
        n = len(z["control_time_s"])
        if not n:
            raise ValueError("Empty control recording")
        for key in ("q","dq","native_target"):
            numeric(z[key],(n,len(joints)),key)
        actions = names(z["action_names"],"action channels")
        numeric(z["target"],(n,len(actions)),"target")
        clock = numeric(z["control_time_s"],(n,),"clock")
        expected = (np.arange(n)+1)/checkpoint.metadata["command_hz"]
        if not np.allclose(clock,expected,rtol=0,atol=1e-9):
            raise ValueError("Control clock is not the recorded regular clock")
        return {"control_frames":n,"duration_s":float(clock[-1]),
                "native_channels":len(joints),"command_channels":len(actions),
                "resampled":False}
