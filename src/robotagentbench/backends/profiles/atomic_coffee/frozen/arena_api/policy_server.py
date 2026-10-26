"""Provider-local adapter served over black-box WebSocket/MessagePack; no generic RPC."""
import argparse
import base64
from collections import OrderedDict
from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from .app import RequestGuard
from .policy_contract import (BLACKBOX_PROFILE, digest, blackbox_specification,
                              pack_measured, validate_chunk)
from .schemas import WireModel, Number, StepIndex
from . import robot_profiles as rp, bimanual as b


class Reset(WireModel):
    instruction: Annotated[str, Field(min_length=1, max_length=1000)]
    profile_id: Literal[BLACKBOX_PROFILE, "arena.g1.joint20.blackbox.v1"]
    profile_sha256: str
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32-1)] = 7
    max_inferences: Annotated[int, Field(strict=True, ge=1, le=6000)] = 10


class RGB(WireModel):
    png_base64: Annotated[str, Field(min_length=1, max_length=2_000_000)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SensorObservation(WireModel):
    episode_id: UUID
    step_index: StepIndex
    time_s: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    source: Literal["isaac_physics"]
    joint_positions_rad: dict[str, Number]
    joint_positions_m: dict[str, Number] = Field(default_factory=dict)
    images: dict[str, RGB]


class Act(WireModel):
    request_id: UUID
    expected_inference_index: StepIndex
    observation: SensorObservation


def decode_images(images):
    from PIL import Image
    import numpy as np
    if set(images) != {"head", "left", "right"}:
        raise ValueError("All three real camera frames are required")
    result = {}
    for name, encoded in images.items():
        data = base64.b64decode(encoded.png_base64, validate=True)
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(data).hexdigest() != encoded.sha256:
            raise ValueError("Image signature/checksum mismatch")
        with Image.open(BytesIO(data)) as im:
            if im.size != (640, 480) or im.mode != "RGB":
                raise ValueError("Expected 640x480 RGB PNG, no implicit image conversion")
            result[name] = np.array(im)
    return result


def create_policy_app(model, output: Path, token=None, *,
                      service_name="unnamed-service", service_version="unspecified", robot_profile=rp.JOINT16):
    """model is a trusted local adapter; CPU tests must provide an explicit fake."""
    if token is not None and len(token) < 32:
        raise ValueError("Policy token must have at least 32 characters")
    output.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    sessions = OrderedDict()
    active = None
    instance = str(uuid4())
    profile = blackbox_specification(robot_profile)
    public_model = {"kind": "feedback_test" if getattr(model, "metadata", {}).get("kind") == "feedback_test" else "remote_service",
                    "declared_name": service_name, "declared_version": service_version}
    metadata = {"schema_version": "arena.policy.v1", "instance_id": instance,
                "profile": profile, "profile_sha256": digest(profile), "model": public_model,
                "evaluation_valid": False, "max_active_sessions": 1,
                "methods": ["reset", "act", "close"], "transport": "WebSocket + MessagePack / PNG bytes",
                "server_idle_session_timeout_s": 300, "evaluation_mode": "remote_blackbox",
                "model_identity_verified": False, "action_origin_verified": False,
                "weights_required": False, "wire_protocol": "arena.blackbox.ws.v1",
                "max_session_inferences": 6000}

    @asynccontextmanager
    async def lifespan(app):
        yield
        model.close()

    app = FastAPI(title="Arena model policy API — closed-loop framework", lifespan=lifespan)
    app.add_middleware(RequestGuard, token=token, max_body_bytes=8 * 1024**2)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({"error": "invalid_policy_request"}, status_code=422)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "instance_id": instance, "model_loaded": True,
                "evaluation_valid": False}

    @app.get("/v1/metadata")
    def get_metadata():
        return metadata

    def find(session_id):
        if session_id not in sessions:
            raise HTTPException(404, "Unknown or expired policy session")
        return sessions[session_id]

    def reset(session_id: UUID, request: Reset):
        nonlocal active
        with lock:
            if request.profile_id != profile["profile_id"] or request.profile_sha256 != metadata["profile_sha256"]:
                raise HTTPException(409, "Profile hash mismatch")
            if session_id in sessions:
                session = find(session_id)
                if session["request"] != request or session["status"] != "running":
                    raise HTTPException(409, "Session UUID reused or closed")
                return session["ack"]
            if active is not None:
                old = find(active)
                if time.monotonic() - old["touched"] <= 300:
                    raise HTTPException(409, "Policy session busy")
                old["status"] = "expired"
                active = None
            directory = output / str(session_id)
            if directory.exists():
                raise HTTPException(409, "Persisted session UUID cannot be reused after a restart")
            directory.mkdir()
            model.reset(request.seed)
            ack = {"session_id": str(session_id), "instance_id": instance,
                   "profile_sha256": metadata["profile_sha256"], "status": "running"}
            session = {"request": request, "ack": ack, "status": "running", "index": 0,
                       "last_step": -1, "last_time": -1., "cache": OrderedDict(),
                       "directory": directory, "touched": time.monotonic()}
            while len(sessions) >= 8:
                sessions.popitem(last=False)
            sessions[session_id] = session
            active = session_id
            (directory / "reset.json").write_text(json.dumps({"metadata": metadata, "request": request.model_dump(mode="json")}, indent=2) + "\n")
            return ack

    def act(session_id: UUID, request: Act):
        nonlocal active
        with lock:
            session = find(session_id)
            signature = digest(request.model_dump(mode="json", exclude_unset=True))
            cached = session["cache"].get(request.request_id)
            if cached:
                if cached[0] != signature:
                    raise HTTPException(409, "Request UUID reused with changed input")
                return cached[1]
            obs = request.observation
            if session["status"] != "running" or active != session_id:
                raise HTTPException(409, "Policy session not running")
            if time.monotonic() - session["touched"] > 300:
                session["status"], active = "expired", None
                raise HTTPException(409, "Policy session idle timeout")
            if request.expected_inference_index != session["index"] or obs.episode_id != session_id:
                raise HTTPException(409, "Stale inference or wrong episode")
            if obs.step_index <= session["last_step"] or obs.time_s <= session["last_time"]:
                raise HTTPException(409, "Fresh measured observation required after execution")
            if session["index"] >= session["request"].max_inferences:
                raise HTTPException(409, "Inference budget exhausted")
            try:
                expected = rp.get(robot_profile).units
                if (set(obs.joint_positions_rad) != {n for n,u in expected.items() if u == "rad"} or
                        set(obs.joint_positions_m) != {n for n,u in expected.items() if u == "m"}):
                    raise ValueError("Measured joint names/units mismatch selected robot profile")
                state = pack_measured(rp.measured(obs, robot_profile), robot_profile)
                images = decode_images(obs.images)
            except (ValueError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc
            started = time.perf_counter()
            index = session["index"]
            audit_obs = obs.model_dump(mode="json", exclude={"images"})
            hashes = {n: image.sha256 for n, image in obs.images.items()}
            audit = {"request_id": str(request.request_id), "request_sha256": signature,
                     "observation": audit_obs, "image_sha256": hashes, "packed_measured_state": state,
                     "instruction": session["request"].instruction, "profile_sha256": metadata["profile_sha256"]}
            (session["directory"] / f"infer{index:04d}_input.json").write_text(json.dumps(audit, indent=2) + "\n")
            try:
                chunk = model.infer(session["request"].instruction, state, images)
                validate_chunk(chunk, robot_profile)
            except Exception as exc:
                session["status"], active = "error", None
                (session["directory"] / "error.json").write_text(json.dumps({"kind": type(exc).__name__, "message": str(exc)}) + "\n")
                raise HTTPException(503, "Model failed; session poisoned, no automatic inference retry") from exc
            response = {"schema_version": "arena.policy.v1", "session_id": str(session_id),
                        "instance_id": instance, "request_id": str(request.request_id),
                        "observation_step_index": obs.step_index, "observation_time_s": obs.time_s,
                        "inference_index": index, "profile_sha256": metadata["profile_sha256"],
                        "input_sha256": signature, "packed_measured_state": state,
                        "image_sha256": hashes, "action_chunk": chunk, "dt_s": b.CONTROL_DT,
                        "inference_s": time.perf_counter() - started, "evaluation_valid": False}
            try:
                (session["directory"] / f"infer{index:04d}_output.json").write_text(json.dumps(response, indent=2, allow_nan=False) + "\n")
            except OSError as exc:
                session["status"], active = "error", None
                raise HTTPException(503, "Inference audit write failed; session poisoned") from exc
            session.update(index=index + 1, last_step=obs.step_index, last_time=obs.time_s, touched=time.monotonic())
            session["cache"][request.request_id] = (signature, response)
            while len(session["cache"]) > 32:
                session["cache"].popitem(last=False)
            return response

    def close(session_id: UUID):
        nonlocal active
        with lock:
            session = find(session_id)
            if active == session_id:
                active = None
            session["status"] = "closed"
            response = {"session_id": str(session_id), "status": "closed", "inferences": session["index"]}
            (session["directory"] / "closed.json").write_text(json.dumps(response) + "\n")
            return response

    from .policy_ws import install_websocket
    install_websocket(app, metadata=metadata, reset=reset, act=act, close=close,
                      reset_type=Reset, act_type=Act, token=token)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Provider-local module:factory; configure weights/GPU in the factory")
    parser.add_argument("--service-name", default="unnamed-service")
    parser.add_argument("--service-version", default="unspecified")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-profile", choices=(rp.JOINT20, rp.JOINT16), default=rp.JOINT16,
                        help="Provider IO contract; joint16 is explicitly restricted to arms/grippers")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    import importlib
    import uvicorn
    module, separator, factory = args.adapter.partition(":")
    if not separator or not module or not factory.isidentifier() or factory.startswith("_"):
        parser.error("--adapter must be module:public_factory")
    model = getattr(importlib.import_module(module), factory)()
    if not all(callable(getattr(model, name, None)) for name in ("reset", "infer", "close")):
        parser.error("Adapter must implement reset, infer, close")
    # Never expose an unprotected inference service on all interfaces.
    app = create_policy_app(model, args.output, os.environ.get("ARENA_POLICY_TOKEN"),
                            service_name=args.service_name, service_version=args.service_version, robot_profile=args.robot_profile)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False,
                ws_max_size=6 * 1024**2, ws_max_queue=2, ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
