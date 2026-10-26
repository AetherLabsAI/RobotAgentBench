"""Single-process API; optional isolated Isaac main-thread worker per episode."""
import argparse
import ipaddress
import math
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .isaac_backend import IsaacOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--backend", choices=("none", "isaac"), default="none")
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--isaac-init-timeout", type=float, default=300.,
                        help="Wall-clock seconds for cold startup/reset, not task simulation time")
    args = parser.parse_args()
    token = os.environ.get("ARENA_API_TOKEN")
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and (token is None or len(token) < 32):
        parser.error("Non-loopback binding requires ARENA_API_TOKEN (32+ characters); prefer an SSH tunnel.")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if not math.isfinite(args.isaac_init_timeout) or args.isaac_init_timeout <= 0:
        parser.error("isaac-init-timeout must be finite and positive")
    options = None
    if args.backend == "isaac":
        if not args.isaac_python or not args.scene or not args.output or args.gpu is None or args.gpu < 0:
            parser.error("Isaac requires --isaac-python --scene --output --gpu")
        # Keep the venv executable's symlink path: resolving it can lose the venv.
        if not args.isaac_python.is_file() or not os.access(args.isaac_python, os.X_OK) or not args.scene.is_file():
            parser.error("Isaac interpreter or scene not found")
        options = IsaacOptions(args.isaac_python.absolute(), args.scene.resolve(), args.output.resolve(), args.gpu,
                               initialization_timeout_s=args.isaac_init_timeout)
    elif any(value is not None for value in (args.isaac_python, args.scene, args.output, args.gpu)):
        parser.error("Isaac worker arguments require --backend isaac")
    try:
        app = create_app(token=token, isaac_options=options)
    except ValueError as exc:
        parser.error(str(exc))
    # Explicitly ignore WEB_CONCURRENCY: multiple managers could drive one robot.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
