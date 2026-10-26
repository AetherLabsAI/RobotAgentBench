"""Unscored initial snapshot through the unchanged public Isaac backend.

Uses native settle/physics/render/policies/cleanup; user-approved startup wait is 900 s.
Does not register 9.1, monkey-patch the worker, or send a robot action.
"""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

TASK_DIR = Path(__file__).resolve().parent
PROJECT = TASK_DIR.parents[2]
SCENE = TASK_DIR / "prepared/existing_box_standard243_initial.usda"


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def smoke_options(gpu, output, isaac_python=None, *, record_initial_evidence=False):
    from arena_api.isaac_backend import IsaacOptions
    # User approved extending only this diagnostic startup wait.
    # Public defaults, task limits and physics/render parameters are untouched.
    return IsaacOptions(python=Path(isaac_python) if isaac_python else PROJECT.parent / ".venv/bin/python",
                        scene=SCENE, output=output, gpu=gpu,
                        initialization_timeout_s=900.0,
                        diagnostic_id='refrigerator_initial.v1' if record_initial_evidence else None)


def joint_errors(fields, expected):
    measured = dict(fields["joint_positions_rad"])
    measured.update(fields["joint_positions_m"])
    missing = sorted(set(expected) - set(measured))
    if missing:
        raise ValueError("Missing measured initial joints: " + ", ".join(missing))
    return {name: float(measured[name] - value) for name, value in expected.items()}


def preflight():
    from pxr import Usd, UsdGeom
    from arena_api import l1_scene
    stage = Usd.Stage.Open(str(SCENE))
    if not stage:
        raise ValueError("Cannot open initial scene")
    paths = ["/World/Robot", "/World/Assets/MicrowaveBeefbox", l1_scene.FRIDGE]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError("Inactive or missing: " + path)
    def poses():
        cache = UsdGeom.XformCache()
        return {p: str(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))) for p in paths}
    before = poses()
    l1_scene.verify(stage)
    stage.SetEditTarget(stage.GetSessionLayer())
    policy = l1_scene.apply(stage)
    if poses() != before:
        raise ValueError("Public L1 preparation changed task initial transforms")
    sources = {}
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            path = Path(layer.realPath)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(scene=str(SCENE), scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),
                source_sha256=sources, l1_policy=policy,
                initial_transforms_preserved=True,
                physics_accepted=False)


def smoke(gpu, output, preflight_path, isaac_python=None, *, record_initial_evidence=False):
    from arena_api.isaac_backend import IsaacBackend, CAMERAS
    from PIL import Image
    receipt = json.loads(preflight_path.read_text())
    if receipt["scene"] != str(SCENE) or receipt["scene_sha256"] != hashlib.sha256(SCENE.read_bytes()).hexdigest():
        raise ValueError("Preflight scene mismatch")
    for path, digest in receipt["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Dependency changed after preflight: " + path)
    output = output.resolve()
    if not output.is_relative_to(TASK_DIR) or output == TASK_DIR:
        raise ValueError("Output must remain below this task directory")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    episode_id = uuid4()
    report = dict(benchmark_id="9.1", sheet_api_id="26-1",
                  runtime_registered=False, diagnostic="native_settle_three_camera_snapshot",
                  run_id=str(episode_id), source=receipt,
                  task_success=None, scoring_implemented=False, initial_state_accepted=False,
                  box_support_verified=False, policy_actions_sent=0,
                  startup_wait_override_user_approved=True,
                  initialization_timeout_s=900.0,
                  isaac_python=str(smoke_options(gpu, output, isaac_python).python),
                  physics_control_render_parameters_overridden=False)
    write_json(output / "request.json", report)
    backend = None
    try:
        backend = IsaacBackend(
            smoke_options(gpu, output, isaac_python, record_initial_evidence=record_initial_evidence),
            episode_id=episode_id, seed=1,
            instruction="9.1 initial-state diagnostic only; no task action or scoring.")
        fields = dict(backend.fields)
        expected = json.loads((TASK_DIR / "sources/initial_frame0.json").read_text())["hold_targets"]
        report["measured_initial_joint_error_native_units"] = joint_errors(fields, expected)
        report["observation"] = fields
        report["metadata"] = backend.metadata
        rgb_digests = {name: hashlib.sha256(backend.frame_rgb(0, name)).hexdigest() for name in CAMERAS}
        backend.close()
        report["worker_exit_code"] = backend.process.returncode
        if record_initial_evidence:
            diagnostic = json.loads((backend.output / "diagnostic_recording.json").read_text())
            if diagnostic.get("completed") is not True or diagnostic.get("samples", 0) < 2:
                raise ValueError("Initial diagnostic recording incomplete")
            report["initial_diagnostic"] = diagnostic
        report["images"] = {}
        for name in CAMERAS:
            path = backend.output / ("step0000_" + name + ".png")
            with Image.open(path) as im:
                digest = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
                size = list(im.size)
            if digest != rgb_digests[name]:
                raise ValueError("Recorded PNG differs from returned RGB: " + name)
            report["images"][name] = dict(path=str(path), size=size, rgb_sha256=digest)
        for path, digest in receipt["source_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError("Source changed during snapshot: " + path)
        report["source_files_unchanged"] = True
        report["snapshot_completed"] = True
        # The generic backend exposes robot state, not box support/pose evidence.
        # Successful startup and images must not be promoted to fridge acceptance.
        write_json(output / "snapshot.json", report)
        print(json.dumps(dict(output=str(output), snapshot_completed=True,
                              initial_state_accepted=False)), flush=True)
    except BaseException as exc:
        report.update(snapshot_completed=False, error=repr(exc))
        if backend is not None:
            report["worker_exit_code"] = backend.process.returncode
        write_json(output / "failed.json", report)
        raise
    finally:
        if backend is not None and not backend.closed:
            backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--record-initial-evidence", action="store_true")
    p.add_argument("--gpu", type=int, choices=(0, 1))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path)
    p.add_argument("--isaac-python", type=Path, help="Existing interpreter/launcher; container uses /isaac-sim/python.sh")
    a = p.parse_args()
    output = a.output.resolve()
    if not output.is_relative_to(TASK_DIR) or output == TASK_DIR:
        p.error("Output must remain below this task directory")
    if a.preflight:
        write_json(output, preflight())
        print(str(output))
    else:
        if a.gpu is None or a.preflight_receipt is None:
            p.error("Smoke requires --gpu and --preflight-receipt")
        smoke(a.gpu, output, a.preflight_receipt.resolve(), a.isaac_python,
              record_initial_evidence=a.record_initial_evidence)


if __name__ == "__main__":
    main()
