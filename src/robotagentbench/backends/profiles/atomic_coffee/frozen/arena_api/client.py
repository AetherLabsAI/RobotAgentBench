"""Small synchronous Python client. Retry identity is explicit, never guessed."""
from __future__ import annotations

from uuid import UUID, uuid4
import math

import httpx

from . import robot_profiles as rp
from .schemas import EpisodeConfig, EpisodeResult, EpisodeState, JointPositionAction, StepRequest


class ArenaClient:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 30.0,
                 reset_timeout: float = 360.0):
        if isinstance(reset_timeout, bool) or not math.isfinite(reset_timeout) or reset_timeout <= 0:
            raise ValueError('reset_timeout must be finite and positive')
        self.reset_timeout = reset_timeout
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # Do not leak localhost traffic/tokens to an inherited proxy.
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, trust_env=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()

    def _request(self, method: str, path: str, payload=None, *, timeout=None):
        response = self.http.request(method, path, json=payload,
                                     **({} if timeout is None else {'timeout': timeout}))
        response.raise_for_status()
        return response.json()

    def capabilities(self) -> dict:
        return self._request("GET", "/v1/capabilities")

    def reset(self, episode_id: UUID, config: EpisodeConfig) -> EpisodeState:
        return EpisodeState.model_validate(self._request(
            "PUT", f"/v1/episodes/{episode_id}", config.model_dump(mode="json"),
            timeout=self.reset_timeout))

    @staticmethod
    def prepare_step(state: EpisodeState, targets: dict[str, float], *, physics_hz=None) -> StepRequest:
        """Keep this object until the response arrives; retry it unchanged on timeout."""
        return StepRequest(request_id=uuid4(), expected_step_index=state.observation.step_index,
                           physics_hz=physics_hz,
                           action=JointPositionAction(targets=targets, unit=rp.get(state.robot_profile).wire_unit))

    def step(self, episode_id: UUID, request: StepRequest) -> EpisodeState:
        return EpisodeState.model_validate(self._request(
            "POST", f"/v1/episodes/{episode_id}/step", request.model_dump(mode="json")))

    def observe(self, episode_id: UUID) -> EpisodeState:
        return EpisodeState.model_validate(self._request("GET", f"/v1/episodes/{episode_id}/observation"))

    def result(self, episode_id: UUID) -> EpisodeResult:
        return EpisodeResult.model_validate(self._request("GET", f"/v1/episodes/{episode_id}/result"))

    def close_episode(self, episode_id: UUID) -> EpisodeState:
        return EpisodeState.model_validate(self._request("DELETE", f"/v1/episodes/{episode_id}"))
