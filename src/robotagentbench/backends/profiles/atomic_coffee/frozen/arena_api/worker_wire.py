"""Private inherited socket IPC; no network listener, pickle, or code execution."""
import json
import struct
import math

from .image_recording import CAMERAS, RGB_BYTES, check_rgb

MAX_MESSAGE = 1024 * 1024


def send(sock, value, *, images=None):
    if images is not None:
        check_rgb(images)
        observation = value['observation']
        value = dict(_arena_rgb_v1=dict(frame_index=observation['frame_index'], time_s=observation['time_s'],
                                      cameras=list(CAMERAS), bytes_per_camera=RGB_BYTES), payload=value)
    body = json.dumps(value, allow_nan=False).encode()
    if len(body) > MAX_MESSAGE:
        raise ValueError("Worker message too large")
    sock.sendall(struct.pack("!I", len(body)) + body)
    if images is not None:
        for name in CAMERAS:
            sock.sendall(images[name])


def receive(sock, *, image_sink=None):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = sock.recv(size - len(result))
            if not chunk:
                raise EOFError("Worker connection closed")
            result.extend(chunk)
        return bytes(result)
    size, = struct.unpack("!I", exact(4))
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid worker message length")
    value = json.loads(exact(size))
    if isinstance(value, dict) and '_arena_rgb_v1' in value:
        if set(value) != {'_arena_rgb_v1', 'payload'}:
            raise ValueError('Invalid image envelope')
        spec, payload = value['_arena_rgb_v1'], value['payload']
        observation = payload.get('observation', {})
        if (spec.get('cameras') != list(CAMERAS) or spec.get('bytes_per_camera') != RGB_BYTES
                or type(spec.get('frame_index')) is not int or spec['frame_index'] < 0
                or spec['frame_index'] != observation.get('frame_index')
                or type(spec.get('time_s')) not in (int, float) or not math.isfinite(spec['time_s'])
                or spec.get('time_s') != observation.get('time_s')):
            raise ValueError('Image/state frame mismatch or invalid bounded image size')
        images = {name: exact(RGB_BYTES) for name in CAMERAS}
        if image_sink is not None:
            image_sink(spec['frame_index'], spec['time_s'], images)
        return payload
    return value
