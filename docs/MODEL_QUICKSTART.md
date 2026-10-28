# Shared setup for the model examples

Use this setup, then choose the [InternVLA](../src/robotagentbench/model_examples/internvla/README.md)
or [Codex](../src/robotagentbench/model_examples/codex/README.md) walkthrough.
Both run the **native worker in Docker on the same Linux machine as the model
client**. You do not need to install Isaac Sim or locate its Python libraries
on the host. The walkthroughs select `prepare_coffee`, episode 1, from the saved
configuration bank; they do not resample it.

To keep an existing Codex login on another computer, use the
[remote-client instructions](../src/robotagentbench/model_examples/codex/NATIVE_SETUP.md#remote-linux-worker--container-deployment)
after preparing the Linux worker. The CLI and credentials stay on that client.

## Requirements

- Linux x86_64, Python 3.11+ with `venv` support, Git, Docker Engine and NVIDIA Container Toolkit.
- Docker access as a **non-root user**. These examples use
  `unix:///var/run/docker.sock`; do not run the model client with `sudo`.
- An Isaac Sim-compatible NVIDIA GPU. InternVLA additionally needs a separate
  GPU with **more than 20,000 MiB free**, CUDA 12.8 build tools for its attention
  dependencies, and Conda for the separate model environment.
- Free disk for the 7.04 GB asset download, extracted assets, native Docker
  image and per-run caches. InternVLA adds approximately 10 GB of model files
  and its Python environment. Download sizes are not total installed sizes.

Run commands on that Linux machine, in one Bash session. Set GPU indices only
to available devices. All jobs on the host must use the **same GPU-lock
directory**; the native worker and InternVLA model retain their occupancy checks.

On Debian/Ubuntu, install the matching Python venv package first (for example,
`python3-venv` for the distribution's default Python). If `python3 -m venv`
reports that `ensurepip` is missing, stop and install that prerequisite or use
a separately installed Python with venv support; do not continue with a partial
environment.

## 1. Install the evaluator

```bash
git clone --branch main https://github.com/AetherLabsAI/RobotAgentBench.git
cd RobotAgentBench
export RAB_ROOT="$(pwd -P)"
python3 -m venv .venv
export RAB_PY="$RAB_ROOT/.venv/bin/python"
"$RAB_PY" -m pip install .
"$RAB_PY" -m robotagentbench --version
export DOCKER_HOST=unix:///var/run/docker.sock
docker info >/dev/null
nvidia-smi
```

If already installed, update your checkout and reinstall the package, then
rebuild the native image below from that **same checkout**. Do not combine a
new host client with an old image. Choose new Runtime/output directories when
following this guide again.

## 2. Install shared assets and the saved configurations

```bash
"$RAB_PY" -m robotagentbench assets fetch resources/assets-0.1.7.json --cache downloads
"$RAB_PY" -m robotagentbench assets install resources/assets-0.1.7.json \
  --cache downloads --destination Assets
"$RAB_PY" -m robotagentbench assets configure resources/assets-0.1.7.json \
  --destination Assets --runtime Runtime --profile room

"$RAB_PY" -m robotagentbench.simulation.saved_instance_release.download \
  --manifest resources/saved-instances-0.2.15.json \
  --cache downloads/configurations --destination "$RAB_ROOT/SavedConfigurations"
mkdir -p "$RAB_ROOT/Restored" "$RAB_ROOT/model-configs"

# Use the existing shared directory instead if other jobs are already running.
export RAB_LOCKS="$RAB_ROOT/gpu-locks"
mkdir -p "$RAB_LOCKS"
```

The bank contains **30 configurations per group, 1,470 total**. These are
release configurations, not the exact instances used for the paper's scores.
See [saved configurations](INSTANCES.md) for the task identifiers and profiles.

## 3. Build the native image

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t robotagentbench:native .
```

This installs the pinned native SDK **inside the image**. Subsequent `restore`
and `configure` commands start CPU-only containers to restore the saved scene
and prepare fresh private Kit/MDL caches. They do not start physics or call a
model. The workflow resolves the image tag to its immutable local ID.

Review the [NVIDIA Omniverse agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
**Only if you accept it**, set this before running a native episode:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## 4. Choose a client

- [InternVLA: install the model environment, download weights, select an episode and run](../src/robotagentbench/model_examples/internvla/README.md)
- [Codex: build its tool sandbox, sign in, select an episode and run](../src/robotagentbench/model_examples/codex/README.md)

Each guide supplies a complete `restore → configure → run` sequence. The helper
reads the selected launch and top-level `public_input_binding` automatically;
you do not need to edit JSON, supply host SDK paths, or manually translate
container paths. It invokes the existing model loop and original task worker.

The walkthroughs cover the standard room/short-task worker, not a universal
replacement for the [cloth](CLOTH.md) and [specialized](SPECIALIZED.md) owners.
Use those dedicated interfaces for their task-specific execution profiles.

## Outputs and reruns

`configure` prints the absolute `config` and `output` paths. Configuration files
live under `model-configs/`; execution outputs live in the Runtime's task-owned
source directory so the original coding sandbox's host mounts remain valid.
`run CONFIG` inspects without executing. **Only `run CONFIG --execute` starts
the model and simulator.**

Use a new `--name` and `--destination` for another model run; it may reuse the
same saved `selection.json`. Configuration prepares one private native cache
set per run. A dispatched run is never retried automatically, including after
a lost acknowledgment. Do not delete its intent or reuse its identity.

With fresh native caches, the first rendered observation can take several
minutes while graphics shaders compile. Keep the original run active rather
than starting a second copy; model inference begins only after the worker
returns its admitted observation.

For InternVLA, inspect `report.json` and `native/native_report.json`. For Codex,
inspect `operator_result.json`, `observations/codex_report.json` and
`native/native_report.json`. Connection, clean shutdown and task success are
different outcomes; scores and task completion come from the original owner.
