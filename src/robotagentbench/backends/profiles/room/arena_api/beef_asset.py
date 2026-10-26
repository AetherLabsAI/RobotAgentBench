"""One shared, metre-authored beef asset. No runtime geometric scaling."""
from pathlib import Path
import hashlib
import math

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "resources/shared_assets/beef_box_standard243_v1"
ASSET = DIRECTORY / "beefbox.usda"
SCENE_LAYER = DIRECTORY / "scene_bindings.usda"
REVISION = "beef_box_standard243_v1"
SIZE_M = (0.243, 0.1215, 0.0522)
MASS_KG = 0.20
MASS_REVISION = "beef_box_mass020_v1"
VISUAL_SHA256 = "8d0b53928067ff0622c3c1961a5a20d310f3d79ea9d20c3fc4888a67b2cdb314"
CONTACT_SHA256 = "a3b33b8b4d8ea9f9966fe4d7549c638cc4fa6a7d3471a8d94330908908b7f1a6"
FILES = {
    "beefbox.usda": "d1f612cca6bde9f6bb91fea98cc685e79475ff1d88564b4c247b5afba86271f3",
    "scene_bindings.usda": "c1059823793cac54c5b5eb26905f6ef2f4e2e58114062d9ecba973ff7db0a7f1",
    "beefbox_standard243_v1.usda": VISUAL_SHA256,
    "beefbox_contact_standard243_v1.usda": CONTACT_SHA256,
    "textures/top.jpg": "8acde73356b919715648f312567fe410c1f9cf10fe4d53c8a91bc8beaa1aa883",
    "textures/bottom.jpg": "a3a965a164be186dc9b3876203d161d0e9890f4a6a755494fa57139e065e9ad0",
    "textures/side_a.jpg": "dcfa39081e75eb510a087962958a48725b1ed3b56c3ee371aea81952b7ecfeec",
    "textures/side_b.jpg": "dfe0bee0a59acc84d40b5ea40493fcf0dfc3494b40884b636bddfde5b0d1dddb",
}
INSTANCES = ("/World/Assets/MicrowaveBeefbox", "/World/Assets/Task108Beefbox")


def verify_files():
    for name, digest in FILES.items():
        if hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Shared beef asset changed: " + name)


def verify_instance_scale(stage):
    """Fail on double scaling instead of silently changing reset/world poses."""
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    for path in INSTANCES:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        # Unit orthonormal basis also rejects inherited nonuniform scale/shear.
        for i in range(3):
            for j in range(3):
                dot = sum(matrix[i][k] * matrix[j][k] for k in range(3))
                if not math.isclose(dot, float(i == j), abs_tol=1e-6):
                    raise ValueError(f"Shared beef box must load at scale=1: {path}")


def verify_instance_mass(stage):
    """Reject stale stronger mass overrides; do not change runtime mass."""
    for path in INSTANCES:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            continue
        mass = prim.GetAttribute("physics:mass").Get()
        if mass is None or not math.isclose(mass, MASS_KG, rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"Shared beef box must use mass={MASS_KG} kg: {path}")


def apply(stage):
    """Bind before simulation, including scenes bearing an older L1 marker."""
    verify_files()
    session = stage.GetSessionLayer()
    path = str(SCENE_LAYER)
    session.subLayerPaths = [path] + [p for p in session.subLayerPaths if p != path]
    verify_instance_scale(stage)
    verify_instance_mass(stage)
    return {"revision": REVISION, "asset": str(ASSET), "instance_scale": 1,
            "size_mm": [243, 121.5, 52.2], "mass_kg": MASS_KG,
            "mass_revision": MASS_REVISION, "physics_revalidated": False}
