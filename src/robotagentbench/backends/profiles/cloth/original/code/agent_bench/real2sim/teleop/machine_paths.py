# -*- coding: utf-8 -*-
"""Where this machine keeps the things the reproduce pipeline needs.

One module owns every machine-specific location, and every entry can be
overridden by an environment variable. Before this, 55 files under teleop
carried 95 copies of ``/home/aether/...``, which meant the pipeline could only
run on the workstation those literals described -- syncing the code to l1 gave
you a repository that could not resolve a single scene or recording. Pointing
the pipeline at another machine is now an export, not an edit:

    export REAL2SIM_DEMO_ROOT=/mnt/data1/workspace/genie_sim_demo/genie_sim_demo/demo_sim
    export REAL2SIM_RECORD_ROOT=/mnt/data1/workspace/data_collected/record

The repository baseline is the source. ``REAL2SIM_DEMO_ROOT`` selects only a
writable runtime mirror; a session always restores that mirror from the source
baseline. Stdlib only: session_scene imports this inside mesh311 and the
initializers inside base.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# One read-only House authoring source.  A release may override this to the
# root of its installed House asset package, but task code never selects a
# second scene or a task-specific root.
HOUSE_SOURCE_ROOT = Path(os.environ.get(
    "REAL2SIM_HOUSE_SOURCE_ROOT",
    str(REPO_ROOT / "arena_scene" / "v1_2" / "demo_sim"),
))

# The Isaac scene root: scene_room.usda, the read-only baseline, cam_preview,
# usdz assets. Everything the render chain reads or restores lives under it.
DEMO_ROOT = Path(os.environ.get(
    "REAL2SIM_DEMO_ROOT",
    "/home/aether/workspace/genie_sim_demo/genie_sim_demo/demo_sim",
))

# Real-robot recordings, one directory per episode UUID. This is the GENERAL
# default (the on-disk working copy); the reproduce queue historically pins the
# external-disk copy instead and keeps its own default for that reason.
RECORD_ROOT = Path(os.environ.get(
    "REAL2SIM_RECORD_ROOT",
    "/home/aether/workspace/data_collected/record",
))

# The queue's authoritative recording source on this workstation. Respecting
# REAL2SIM_RECORD_ROOT here (instead of overwriting it, as the queue used to)
# is what lets the same queue run against l1's copy.
QUEUE_RECORD_ROOT = Path(os.environ.get(
    "REAL2SIM_RECORD_ROOT",
    "/media/aether/Backup Plus/AetherAI/livingroom_data/record",
))

# The robot USD asset inside the genie_sim checkout. Derived from DEMO_ROOT
# because the two live in one repository whose internal layout is fixed
# (demo_sim sits two levels below the checkout root); an env override exists
# for machines that split them anyway.
ROBOT_ASSET_ROOT = Path(os.environ.get(
    "REAL2SIM_ROBOT_ASSET_ROOT",
    str(DEMO_ROOT.parents[1] / "genie_sim" / "source" / "geniesim"
        / "assets" / "robot" / "G1_omnipicker"),
))

# The docker container that owns Isaac rendering.
CONTAINER_NAME = os.environ.get("REAL2SIM_CONTAINER", "genie_sim_benchmark")

# Optional absolute Isaac interpreter path INSIDE the selected container.
# Unset preserves existing containers that expose the legacy omni_python alias.
# An explicit setting launches directly, without relying on an interactive shell.
ISAAC_PYTHON = os.environ.get("REAL2SIM_ISAAC_PYTHON")


def isaac_python_command(script: str, *arguments: str) -> list[str]:
    """Return only the container-side argv; callers retain env and timeouts."""
    if ISAAC_PYTHON is None:
        return ["bash", "-ic", shlex.join(["omni_python", script, *arguments])]
    if (
        not ISAAC_PYTHON
        or not Path(ISAAC_PYTHON).is_absolute()
        or any(character in ISAAC_PYTHON for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(
            "REAL2SIM_ISAAC_PYTHON must be an absolute container executable path")
    return [ISAAC_PYTHON, script, *arguments]


# The CUDA device index the render/align chain pins INSIDE that container.
# Container-relative: a container created with NVIDIA_VISIBLE_DEVICES=2,3
# sees those as 0 and 1.  Lanes that own two cards use this to keep a render
# off the card their studio/second job occupies.
RENDER_GPU = os.environ.get("REAL2SIM_RENDER_GPU", "0")

# Conda interpreters for the helper environments. The pipeline shells out to
# them by absolute path because `conda run` from inside a locked render is not
# always available; each machine keeps its own conda prefix.
MESH311_PYTHON = Path(os.environ.get(
    "REAL2SIM_MESH311_PYTHON",
    "/home/aether/miniconda3/envs/mesh311/bin/python"))
PGSR_PYTHON = Path(os.environ.get(
    "REAL2SIM_PGSR_PYTHON",
    "/home/aether/miniconda3/envs/pgsr/bin/python"))
BASE_PYTHON = Path(os.environ.get(
    "REAL2SIM_BASE_PYTHON", "/home/aether/miniconda3/bin/python"))


def describe() -> str:
    """One line per resolved path, for logs and hand-over sanity checks."""
    return "\n".join((
        f"REAL2SIM_HOUSE_SOURCE_ROOT -> {HOUSE_SOURCE_ROOT}",
        f"REAL2SIM_DEMO_ROOT   -> {DEMO_ROOT}",
        f"REAL2SIM_RECORD_ROOT -> {RECORD_ROOT} (queue: {QUEUE_RECORD_ROOT})",
        f"REAL2SIM_CONTAINER   -> {CONTAINER_NAME}",
        f"REAL2SIM_ISAAC_PYTHON -> {ISAAC_PYTHON!r} (None: legacy omni_python)",
        f"ROBOT_ASSET_ROOT     -> {ROBOT_ASSET_ROOT}",
        f"MESH311/PGSR/BASE    -> {MESH311_PYTHON} | {PGSR_PYTHON} | {BASE_PYTHON}",
    ))


if __name__ == "__main__":
    print(describe())
