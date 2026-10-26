"""HTTP facade. All mutable environment calls are serialized by the manager."""
from __future__ import annotations

from contextlib import asynccontextmanager
import secrets
from uuid import UUID, uuid4

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .manager import ArenaError, EpisodeManager
from .isaac_backend import IsaacOptions
from .schemas import EpisodeConfig, EpisodeResult, EpisodeState, ErrorResponse, StepRequest
from . import policy_execution as pe, robot_profiles as rp

MAX_BODY_BYTES = 64 * 1024


class LifecycleRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_step_index: StrictInt = Field(ge=0)


class RequestGuard:
    """Bound streamed bodies before parsing and optionally authenticate all but health."""

    def __init__(self, app, token: str | None, max_body_bytes: int = MAX_BODY_BYTES):
        self.app, self.token, self.max_body_bytes = app, token, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if self.token and scope["path"] != "/healthz":
            headers = dict(scope["headers"])
            provided = headers.get(b"authorization", b"")
            expected = b"Bearer " + self.token.encode("utf-8")
            if not secrets.compare_digest(provided, expected):
                response = JSONResponse(
                    {"error": {"code": "unauthorized", "message": "A valid bearer token is required."}},
                    status_code=401, headers={"WWW-Authenticate": "Bearer"},
                )
                return await response(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                response = JSONResponse(
                    {"error": {"code": "body_too_large", "message": f"Request body limit is {self.max_body_bytes} bytes."}},
                    status_code=413,
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(*, token: str | None = None,
               isaac_options: IsaacOptions | None = None, registry=None) -> FastAPI:
    if token is not None and len(token) < 32:
        raise ValueError("ARENA_API_TOKEN must have at least 32 characters, or be unset for loopback use")
    manager = EpisodeManager(isaac_options=isaac_options, registry=registry)
    instance_id = str(uuid4())

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.shutdown()

    app = FastAPI(
        title="Robot Agent Arena", version=__version__, lifespan=lifespan,
        description="Physical execution and registered task evaluators. Inspect /v1/tasks for per-task capabilities.",
        responses={code: {"model": ErrorResponse} for code in (401, 404, 409, 413, 422, 500, 503)},
    )
    app.state.manager = manager
    app.add_middleware(RequestGuard, token=token)

    @app.exception_handler(ArenaError)
    async def arena_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo bodies, non-finite values or credentials into JSON/logs.
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": "Request does not match the v1 schema; see /openapi.json."}},
            status_code=422,
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "api_version": "v1", "version": __version__,
                "instance_id": instance_id, "physics_ready": manager.physics_ready,
                "isaac_configured": isaac_options is not None}

    @app.get("/v1/capabilities")
    def capabilities():
        return {
            "api_version": "v1", "instance_id": instance_id,
            "backend": "isaac_physics" if isaac_options else "none",
            "physics_ready": manager.physics_ready, "go1_ready": False, "evaluation_valid": False,
            "physics_configuration_validated": False,
            "max_active_episodes": 1, "max_episode_steps": 1000, "retained_episodes": manager.retention,
            "task_max_episode_steps": {t.task_id: t.max_steps for t in manager.registry.all() if t.ready},
            "task_max_inferences": {t.task_id: t.max_inferences for t in manager.registry.all() if t.ready},
            "persistence": "in_memory_only", "max_request_bytes": MAX_BODY_BYTES,
            "physics_rate_switch": {"schema": "arena.physics_rate.v1", "available": bool(isaac_options),
                                    "request_field": "physics_hz", "supported_hz": [60, 120, 240],
                                    "default_hz": 240, "control_hz": 20,
                                    "scope": "entire_physics_scene", "boundary": "before_next_control_action",
                                    "eligibility": "unscored rigid whole-scene episode without force-feedback controller",
                                    "mixed_rate_trajectory_export": False},
            "live_pause": {"schema": "arena.live_pause.v1", "supported": bool(isaac_options),
                           "advances_physics": False, "survives_worker_restart": False,
                           "extends_simulation_budget": False,
                           "requires_keepalive": True, "worker_idle_timeout_s": 300},
            "default_robot_profile": rp.DEFAULT,
            "robot_profiles": {p: rp.specification(p) for p in (rp.JOINT20, rp.JOINT16)},
            "action_specs": {p: pe.spec(p) for p in (rp.JOINT20, rp.JOINT16)} if isaac_options else {},
            "action_spec": pe.spec(rp.DEFAULT) if isaac_options else None,
            "task_action_specs": {t.task_id: pe.spec(rp.DEFAULT) for t in manager.registry.all() if t.ready} if isaac_options else {},
            "task_randomization_supported": {t.task_id: list(t.randomization_flags) for t in manager.registry.all() if t.ready},
            "task_evaluators": {t.task_id: t.evaluator_id for t in manager.registry.all() if t.ready},
            "observation_spec": ({"cameras": ["head", "left", "right"], "source": "isaac_physics",
                                  "resolution": [640, 480], "encoding": "png", "images": "authenticated_relative_urls"}
                                 if isaac_options else None),
            "randomization_supported": [],
            "limitations": (["Select a robot profile at reset; joint16 explicitly holds torso/head targets",
                              "Both profiles expand paired gripper targets; other native targets are held",
                              "See per-episode initialization/evaluation and worker.log for physical validity",
                              "Task reset/scoring/randomization are registration-specific; inspect /v1/tasks",
                              "Controller limits do not guarantee collision safety",
                              "Worker idle timeout 300 s; initialize a fresh episode after worker exit"]
                             if isaac_options else ["No Isaac execution", "No GO-1 mapping", "No task scoring or rendered images",
                                                    "One process/worker only", "No scene randomization in API resets"]),
        }

    @app.get("/v1/tasks")
    def tasks():
        return {"tasks": [dict(task_id=t.task_id, ready=t.ready and isaac_options is not None,
            evaluation_valid=False, evaluator_id=t.evaluator_id, scoring_available=t.evaluator_id is not None,
            reason=t.description, instruction=t.instruction, max_steps=t.max_steps,
            default_steps=t.default_steps, randomization_supported=list(t.randomization_flags),
            task_parameter_schema=t.parameter_schema)
            for t in manager.registry.all()]}

    @app.put("/v1/episodes/{episode_id}", response_model=EpisodeState)
    def reset(episode_id: UUID, config: EpisodeConfig):
        """Create/reset with a NEW client UUID. Same UUID+config retries return the initial response."""
        return manager.create(episode_id, config)

    @app.get("/v1/episodes/{episode_id}/observation", response_model=EpisodeState)
    def observe(episode_id: UUID):
        """Read current state without advancing time."""
        return manager.observe(episode_id)

    @app.post("/v1/episodes/{episode_id}/step", response_model=EpisodeState)
    def step(episode_id: UUID, request: StepRequest):
        """Apply one action. Retry with the exact same request UUID and body after a timeout."""
        return manager.step(episode_id, request)

    @app.get("/v1/episodes/{episode_id}/result", response_model=EpisodeResult)
    def result(episode_id: UUID):
        return manager.result(episode_id)

    @app.post('/v1/episodes/{episode_id}/pause', response_model=EpisodeState)
    def pause(episode_id: UUID, request: LifecycleRequest):
        return manager.lifecycle(episode_id, 'pause', request.expected_step_index)

    @app.post('/v1/episodes/{episode_id}/resume', response_model=EpisodeState)
    def resume(episode_id: UUID, request: LifecycleRequest):
        return manager.lifecycle(episode_id, 'resume', request.expected_step_index)

    @app.post('/v1/episodes/{episode_id}/keepalive', response_model=EpisodeState)
    def keepalive(episode_id: UUID, request: LifecycleRequest):
        return manager.lifecycle(episode_id, 'keepalive', request.expected_step_index)

    @app.get("/v1/episodes/{episode_id}/frames/{step_index}/{camera}.png", response_class=FileResponse)
    def frame(episode_id: UUID, step_index: int, camera: str):
        """Serve a recorded RGB only. No arbitrary paths or additional simulation steps."""
        content = manager.frame_content(episode_id, step_index, camera)
        if isinstance(content, bytes):
            from .image_recording import encode_png
            # Encode requested pixels in memory, outside the simulation lock.
            return Response(encode_png(content), media_type='image/png')
        return FileResponse(content, media_type="image/png")

    @app.delete("/v1/episodes/{episode_id}", response_model=EpisodeState)
    def close(episode_id: UUID):
        """Release environment resources; retain the bounded diagnostic record."""
        return manager.close(episode_id)

    return app
