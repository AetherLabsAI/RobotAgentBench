"""Arena black-box wire v1. Binary PNG, bounded MessagePack, no generic RPC.

This is an Arena protocol, not XPolicyLab wire compatibility. The HTTP-shaped
client methods are a local facade for the existing closed-loop controller.
"""
import asyncio
import base64
import copy
import secrets
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import msgpack
from anyio import CancelScope
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .schemas import WireModel

WIRE = "arena.blackbox.ws.v1"
MAX_FRAME = 6 * 1024**2
MAX_PNG = 1_500_000


class Message(WireModel):
    protocol: Literal[WIRE]
    request_id: UUID
    op: Literal["metadata", "reset", "act", "close"]
    session_id: UUID | None = None
    payload: dict = Field(default_factory=dict)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("Duplicate or non-string map key")
        result[key] = value
    return result


def _no_ext(code, data):
    raise ValueError("MessagePack extensions are not supported")


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_FRAME:
        raise ValueError("Expected bounded binary MessagePack frame")
    value = msgpack.unpackb(data, raw=False, strict_map_key=True,
                           object_pairs_hook=_pairs, ext_hook=_no_ext,
                           max_str_len=64_000, max_bin_len=MAX_PNG,
                           max_array_len=2048, max_map_len=256, max_ext_len=0)
    if not isinstance(value, dict):
        raise ValueError("Frame must be a map")
    return value


def pack(value):
    data = msgpack.packb(value, use_bin_type=True)
    if len(data) > MAX_FRAME:
        raise ValueError("Frame limit exceeded")
    return data


def binary_images(payload, *, to_wire):
    """Canonical audit hash uses the existing base64 JSON representation only
    in local memory. Network frames carry PNG bytes, never base64 strings.
    """
    value = copy.deepcopy(payload)
    images = value["observation"]["images"]
    if not isinstance(images, dict) or set(images) != {"head", "left", "right"}:
        raise ValueError("Exactly three cameras required")
    for entry in images.values():
        source, target = ("png_base64", "png_bytes") if to_wire else ("png_bytes", "png_base64")
        if not isinstance(entry, dict) or set(entry) != {source, "sha256"}:
            raise ValueError("Unexpected image fields")
        data = base64.b64decode(entry.pop(source), validate=True) if to_wire else entry.pop(source)
        if not isinstance(data, bytes) or len(data) > MAX_PNG:
            raise ValueError("Invalid PNG binary payload")
        entry[target] = data if to_wire else base64.b64encode(data).decode("ascii")
    return value


def install_websocket(app, *, metadata, reset, act, close, reset_type, act_type, token):
    @app.websocket("/v1/blackbox/ws")
    async def endpoint(ws: WebSocket):
        # RequestGuard handles HTTP only. Authenticate WS separately, before accept.
        # SDK is server-to-server; reject browser-origin connections.
        if ws.headers.get("origin") or (token and not secrets.compare_digest(
                ws.headers.get("authorization", "").encode(), ("Bearer " + token).encode())):
            await ws.close(code=1008)
            return
        await ws.accept()
        owned = set()
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_bytes(), timeout=300)
                    request = Message.model_validate(unpack(data))
                except (ValueError, TypeError, KeyError, ValidationError, msgpack.UnpackException):
                    await ws.close(code=1008, reason="Invalid binary protocol frame")
                    return
                response = {"protocol": WIRE, "request_id": str(request.request_id),
                            "instance_id": metadata["instance_id"], "status": 200}
                try:
                    if request.op == "metadata":
                        if request.session_id is not None or request.payload:
                            raise ValueError("Metadata accepts no payload/session")
                        result = metadata
                    elif request.op == "reset":
                        if request.session_id is None or (owned and request.session_id not in owned):
                            raise ValueError("One owned session per connection")
                        result = await run_in_threadpool(reset, request.session_id,
                                                         reset_type.model_validate(request.payload))
                        owned.add(request.session_id)
                    else:
                        if request.session_id not in owned:
                            raise HTTPException(409, "Connection does not own session")
                        if request.op == "act":
                            body = act_type.model_validate(binary_images(request.payload, to_wire=False))
                            result = await run_in_threadpool(act, request.session_id, body)
                        else:
                            if request.payload:
                                raise ValueError("Close accepts no payload")
                            result = await run_in_threadpool(close, request.session_id)
                            owned.remove(request.session_id)
                    response["result"] = result
                except HTTPException as exc:
                    response.update(status=exc.status_code, result={"error": "policy_request_rejected"})
                except (ValueError, TypeError, KeyError, ValidationError):
                    response.update(status=422, result={"error": "invalid_policy_request"})
                except Exception:
                    # Do not send adapter exception text, local paths or credentials.
                    response.update(status=503, result={"error": "provider_failure"})
                    await ws.send_bytes(pack(response))
                    return
                await ws.send_bytes(pack(response))
        except (WebSocketDisconnect, asyncio.TimeoutError, OSError, RuntimeError):
            pass
        finally:
            # ASGI disconnect/shutdown can cancel the connection's scope while
            # we are awaiting the threadpool. Finish releasing an owned session.
            with CancelScope(shield=True):
                for session in owned:
                    try:
                        await run_in_threadpool(close, session)
                    except Exception:
                        pass


def validate_endpoint(url, token):
    parsed = urlsplit(url)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a ws/wss endpoint without credentials, query or fragment")
    if parsed.path != "/v1/blackbox/ws":
        raise ValueError("Expected /v1/blackbox/ws endpoint")
    loopback = parsed.hostname in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme == "ws" and not loopback:
        raise ValueError("Remote plaintext WS rejected; use WSS or a loopback SSH tunnel")
    if token is not None and len(token) < 32:
        raise ValueError("Token must contain at least 32 characters")
    if not loopback and not token:
        raise ValueError("Remote WSS requires an authentication token")


class BlackboxClient:
    """Synchronous, single-controller client. No reconnect or inference retry.

    An uncertain reply poisons the connection. The controller must stop the
    episode, rather than resend, resample or play the rest of an old chunk.
    """
    blackbox_transport = True

    def __init__(self, url, *, token=None, timeout=120, connection=None):
        validate_endpoint(url, token)
        self.timeout, self.instance_id, self.failed = timeout, None, False
        if connection is not None:
            self.ws = connection  # Explicit in-process transport for CPU tests.
        else:
            from websockets.sync.client import connect
            self.ws = connect(url, additional_headers={"Authorization": "Bearer " + token} if token else {},
                              proxy=None, open_timeout=15, close_timeout=3,
                              compression=None, max_size=MAX_FRAME, max_queue=2,
                              ping_interval=20, ping_timeout=20)

    def request(self, method, path, json=None):
        if self.failed:
            raise RuntimeError("Black-box connection failed; no automatic retry")
        session = None
        payload = {} if json is None else json
        if method == "GET" and path == "/v1/metadata":
            op = "metadata"
        else:
            parts = path.strip("/").split("/")
            if len(parts) not in (3, 4) or parts[:2] != ["v1", "sessions"]:
                raise ValueError("Not an allowed policy operation")
            session = str(UUID(parts[2]))
            if method == "POST" and len(parts) == 4 and parts[3] == "act":
                op, payload = "act", binary_images(payload, to_wire=True)
            elif method in ("PUT", "DELETE") and len(parts) == 3:
                op = "reset" if method == "PUT" else "close"
            else:
                raise ValueError("Not an allowed policy operation")
        message = {"protocol": WIRE, "request_id": str(uuid4()), "op": op,
                   "session_id": session, "payload": payload}
        try:
            self.ws.send(pack(message))
            response = unpack(self.ws.recv(timeout=self.timeout))
            if (set(response) != {"protocol", "request_id", "instance_id", "status", "result"} or
                    response["protocol"] != WIRE or response["request_id"] != message["request_id"] or
                    type(response["status"]) is not int or not 200 <= response["status"] <= 599):
                raise ValueError("Invalid or mismatched service reply")
            instance = str(UUID(response["instance_id"]))
            if self.instance_id is not None and instance != self.instance_id:
                raise ValueError("Provider instance changed")
            self.instance_id = instance
            if op == "metadata" and response["status"] == 200 and response["result"].get("instance_id") != instance:
                raise ValueError("Metadata instance mismatch")
            return httpx.Response(response["status"], json=response["result"],
                                  request=httpx.Request(method, "http://blackbox.invalid" + path))
        except Exception as exc:
            self.failed = True
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError("Black-box reply uncertain or invalid; episode must stop") from exc

    def get(self, path):
        return self.request("GET", path)

    def delete(self, path):
        return self.request("DELETE", path)

    def close(self):
        self.ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
