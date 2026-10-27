# Installation

## Core package

Use Python 3.11 or newer in a separate environment. The evaluation code is on
the `main` branch. For example, on a POSIX shell:

```sh
git clone --branch main https://github.com/AetherLabsAI/RobotAgentBench.git
cd RobotAgentBench
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m robotagentbench --version
python -m robotagentbench catalog standalone --format json
```

The core wheel has no third-party Python runtime requirements. Building from
source requires setuptools 77 or newer; the build requirement is declared in
`pyproject.toml`. Installing the package does not install Isaac Sim, native
collision libraries, model SDKs or assets.

The CLI also accepts an operator-reviewed dependency configuration:

```sh
robotagentbench check-dependencies /path/to/dependencies.json
```

It returns zero only when all configured resources match their pinned hashes.
Task startup also checks the selected source profile and required resources.

## Native environment

Keep the simulator installation separate from the core package environment.
The reviewed native source profile uses the following baseline:

| Component | Profile |
| --- | --- |
| Interpreter | CPython 3.11, `cpython-311` ABI |
| Native platform | Linux x86_64 |
| Isaac Sim distributions | 5.1.0.0 |
| NumPy | 1.26.0 |
| SciPy | 1.15.3 |
| Pillow | 11.3.0 |
| python-fcl | 0.7.0.11 |
| Shapely | 2.1.2 |
| Node.js scene planner | 22.23.3 |

Native collision artifacts, shared libraries, USD plugins, material resolvers
and GPU access must match this profile. Python 3.12 cannot load CPython 3.11
extension artifacts.

Obtain simulator software and native dependencies from their upstream
distributions under their own licenses. Accept required agreements yourself;
the package does not accept licenses, install system packages or repair an
existing environment on your behalf. Do not use a live evaluation's writable
caches or output directories for setup checks.

## Install assets and configure a workspace

Task source is included with the package. Fetch the matching asset release,
install it, then assemble a local workspace:

```sh
robotagentbench backends --profile room --verify
robotagentbench assets fetch resources/assets-0.1.0.json --cache downloads
robotagentbench assets install resources/assets-0.1.0.json --cache downloads --destination Assets
robotagentbench assets configure resources/assets-0.1.0.json --destination Assets --runtime Runtime --profile room
```

Use `atomic_coffee`, `failure_recovery`, or `cloth` for the corresponding
specialized environment. Keep these source profiles separate. The commands
create fresh local directories and preserve the code and asset versions in
their manifests. See [assets](ASSETS.md) for offline installation.

Configure the separate Isaac Sim interpreter and native libraries using
[dependency configuration](DEPENDENCIES.md). Run task workers as dedicated
processes with fresh output directories and a shared cooperative GPU-lock
directory. The [coffee example](EXAMPLE.md) walks through the container setup,
scene preparation and interactive worker. See [the runtime guide](RUNTIME.md)
for the worker interface.

## Optional model clients

For a complete same-host Docker workflow, begin with
[shared model setup](MODEL_QUICKSTART.md). Its `docker_workflow` helper restores
a selected initial state and generates the native/agent configurations using
the image-owned SDK, without requiring a host Isaac installation.

Choose the [InternVLA guide](../src/robotagentbench/model_examples/internvla/README.md)
for the fine-tuned G1 policy, or the
[Codex guide](../src/robotagentbench/model_examples/codex/README.md) for a
code-based agent. Both guides include saved-instance restoration, configuration,
an inspect-only command, execution and result locations.

The Codex guide includes execution code, a Dockerfile and locked Python
dependencies. Build the code-execution sandbox on the Linux Docker host used
by the worker, then select your local image when staging the Runtime. Its
resolved ID is recorded for that deployment; no server-specific ID is required.
This sandbox is **not** the Isaac Sim image or a model, and receives no model
credentials, network access, Docker socket or GPU. A lightweight sandbox smoke
check is provided; it does not call a model or start the simulator.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Dependency verification fails | Canonical paths, permissions, original hashes, symlinks and unresolved Git LFS pointers |
| Native metadata probe fails | Interpreter target, ABI, platform and exact installed distribution metadata |
| Asset loading fails after file checks pass | Nested USD dependencies, payloads, textures, plugins and material paths |
| A specialized owner is unavailable | Its dedicated interpreter handoff and complete source/resource profile |
| GPU admission is rejected | Occupancy and the shared lock; do not terminate unrelated processes |

Setup and runtime errors are reported separately from scored task outcomes.
