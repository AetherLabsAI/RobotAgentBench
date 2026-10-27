# Codex Robo Agent example

Run a code-based agent using Codex App Server and the project's existing robot
tools. The client receives public observations, invokes robot tools through the
native worker, and closes the episode normally. It does not replace the task
scorer, controller, physics, or safety checks.

## Prerequisites

- Install the evaluator, assets and a complete independent native SDK/Python
  environment as described in [Installation](../../../../docs/INSTALL.md).
- Install Codex CLI on the agent computer and sign in using your own account.
  The original recipe uses `gpt-6-astra`, `medium`, and a 200-response budget.
  Model usage can incur charges or consume account quota. There is no automatic
  model, provider, transport or API-key fallback.
- Prepare a saved room/short-task launch and a shared GPU-lock directory.
  Dedicated cloth and specialized owners are not replaced by this room example.
- Build the code-execution sandbox from the Dockerfile below.
  This is separate from the native simulator environment and model credentials.

The native worker runs on Linux. The agent client can run on that machine or
connect from a separate computer using a trusted SSH JSONL command. Credentials
remain on the agent computer, not in the policy sandbox or asset release.

## Build the code-execution sandbox once

The sandbox contains Python and numerical/kinematics dependencies, **not Codex
weights or Isaac Sim**. Its [Dockerfile](sandbox/Dockerfile) and
[dependency lock](sandbox/requirements.lock) are included in both the source
repository and Python package. On the native worker's **Linux amd64** Docker
host, as a non-root user with Docker access:

```sh
# Export to a new directory (or use the sandbox/ directory in this repository).
python -m robotagentbench.model_examples.codex.sandbox_build --export sandbox-build
docker --host unix:///var/run/docker.sock build --platform linux/amd64 \
  -t robotagentbench-policy:local sandbox-build

# Inspect this local build; no container or model starts.
python -m robotagentbench.model_examples.codex.sandbox_build \
  --image robotagentbench-policy:local --docker-host unix:///var/run/docker.sock
```

Build time needs access to the pinned Python base and Python package index.
Execution has no network. Use the same Docker socket for staging and running.
You may choose your own tag. Staging resolves it to an immutable **local** image
ID, so moving the tag later cannot silently change an existing evaluation.
The image is checked for Linux amd64, the build/lock labels, and no declared
volumes or build hooks. These checks validate a trusted operator's build, not
the safety of an arbitrary untrusted image. The published dependency versions
and sandbox restrictions are preserved; no shared image or running container
is modified.

For a short connection check, choose a fresh absolute output directory under
an existing parent:

```sh
python -m robotagentbench.model_examples.codex.sandbox_smoke \
  --image robotagentbench-policy:local --docker-host unix:///var/run/docker.sock \
  --output /absolute/path/to/sandbox-check-001
```

This starts the real isolated Python-tool transport, checks the installed
dependency versions, initializes both robot controllers, executes a small
Python command, reads FK, and removes its own container. It uses explicitly
synthetic robot state: **no Codex call, Isaac startup, physical action or score**.
The execution code is included in the packaged backend (`native_tools/sandbox.py`
and `sandbox_boot.py`); the unchanged robot helpers are supplied by
[server_profile.py](server_profile.py). They are mounted into the restricted
workspace by the worker rather than baked into a machine-specific image.

## 1. Stage the coding source profile

Create a new independent Runtime with the included server coding tools:

```sh
python -m robotagentbench.model_examples.codex.stage_runtime \
  --runtime /absolute/path/to/Runtime \
  --destination /absolute/path/to/codex-workspace \
  --policy-image robotagentbench-policy:local \
  --docker-host unix:///var/run/docker.sock --stage
```

Use `codex-workspace/Runtime` below. Staging binds the exact included robot-tool
resources and your resolved local image ID without changing the provider Runtime
or task scoring/physics. Only the launcher image binding changes; its non-root,
network-off, read-only and resource-limit arguments remain unchanged. The
destination must be new. It does not generate a scene, start Isaac, or call a
model. See [saved configurations](../../../../docs/INSTANCES.md) to select an
already generated initial state.

## 2. Configure the worker and agent

In the prepared native environment:

```sh
python -m robotagentbench.model_examples.codex.configure \
  --runtime /absolute/path/to/codex-workspace/Runtime \
  --sdk-site /absolute/path/to/native/site-packages \
  --launch /absolute/path/to/prepared/launch.json \
  --input-root /absolute/path/to/prepared \
  --native-output /absolute/path/to/codex-workspace/Runtime/short_tasks/agent_probe_v1/deployment_release/source/short_tasks/agent_probe_v1/cap_v1/codex_runs/episode_01 \
  --agent-output /absolute/path/to/results/episode_01 \
  --lock-root /absolute/path/to/shared/gpu-locks --gpu 0 \
  --docker-host unix:///var/run/docker.sock \
  --native-python /absolute/path/to/native/bin/python3.11 \
  --codex-executable /absolute/path/to/codex \
  --cli-version 'codex-cli YOUR_INSTALLED_VERSION' \
  --destination /absolute/path/to/config/episode_01
```

Replace paths with existing canonical paths. Use the exact string printed by
your `codex --version`; the client rejects a different binary version. For a
saved room instance with separately supplied public context, also pass its
`--public-input-binding` and all of its `--input-root` directories. The worker
checks that this public text belongs to the unchanged selected launch.

Configuration checks the staged image on that Docker endpoint and writes
`worker.json` and `operator.json` but starts no container or model. Execution
uses the recorded immutable ID without pulling or falling back to another image.
The native output must be a fresh directory under the Runtime's `cap_v1`
owner; the agent output must also be new. Native outputs stay in that location
because the original isolated sandbox uses host-daemon file binds there.

## 3. Run

```sh
# Inspect the configured entry without starting a model or simulator.
python -m robotagentbench.model_examples.codex /absolute/path/to/config/episode_01/operator.json

# Start the native worker and real Codex agent.
python -m robotagentbench.model_examples.codex /absolute/path/to/config/episode_01/operator.json --execute
```

The command starts the saved worker, reads its admitted observation, runs the
tool loop, and requests the original close handshake. There is no manual
file-descriptor handoff and no automatic action retry. Starting a new run
requires a new output/intent; never reuse a finished or uncertain episode.

## Remote Linux worker / container deployment

If simulation is remote, keep the client and its Codex login local. Configure
the local client with an explicit trusted worker command:

```sh
python -m robotagentbench.model_examples.codex.client_config \
  --codex-executable /absolute/local/path/to/codex \
  --cli-version 'codex-cli YOUR_INSTALLED_VERSION' \
  --worker-argv-json '["/your/strict-ssh-wrapper","sim-host","/your/native-worker-command"]' \
  --output /absolute/local/path/to/results/episode_01 \
  --destination /absolute/local/path/to/config/operator.json

python -m robotagentbench.model_examples.codex /absolute/local/path/to/config/operator.json --execute
```

The SSH wrapper must preserve stdin/stdout, strict host verification and no
TTY. It must select a prepared native environment, not a shell supplied by the
model. Do not copy account files to the simulator.

For Docker, `python -m robotagentbench.model_examples.codex.namespace CONFIG`
describes the trusted namespace and `CONFIG --execute` dispatches it once.
The JSON schema is `robotagentbench.codex_namespace.v1`; explicitly provide
`runtime`, `image`, `name`, `gpu`, `lock_root`, `sdk_site`, `package_root`,
`entry_argv`, `readonly_roots`, `docker_cli`, `docker_host`, and `intent`.
Saved-instance `readonly_aliases` map original logical paths to downloaded
local backing directories. Keep the Runtime mounted at its actual host path so
the original policy sandbox's host-daemon binds resolve correctly.

For a read-only SDK image, prepare the original task-private Kit/MDL caches
and NVIDIA ICD without importing Isaac:

```sh
python -m robotagentbench.model_examples.native_graphics prepare /absolute/path/to/fresh/graphics
python -m robotagentbench.model_examples.native_cache \
  --sdk-site /absolute/path/to/native/site-packages \
  --destination /absolute/path/to/fresh/cache
```

Pass the returned `graphics_binding` and `native_cache_binding` objects in the
trusted namespace JSON. These permit only the original ICD and two fixed
writable cache targets. Never share a run's caches or change SDK permissions.
The native owner needs the operator's Docker CLI/socket for the isolated policy
sandbox; **the policy sandbox itself receives neither socket nor GPU**. The
generic coffee `--prepare-native` command does not automatically configure this
model deployment.

## Observations, tools and results

Codex receives the current public instruction/context, head and wrist images,
measured joints/base/TCP, and the explored navigation map. Targets, object
poses, contacts, answer bindings, scores and evaluator state are not model
input. The map is an observation, not an automatic navigation action.

Tools are `move_to`, `execute_python`, `execute_actions`, `observe`, `move_base`,
`turn_base`, `look`, `wait`, and `done`. Python/control-function code executes
only in the original isolated robot sandbox. Host shell, filesystem browsing,
network access and external tools are not agent capabilities.

`operator_result.json` records termination and cleanup. Original owner scores
are forwarded separately when available. A model's `done` or a successful
process exit is not a task-success claim. Unacknowledged actions are never
resent. Raw model reasoning and App Server events are not persisted in agent
result files; do not publish private operator/native logs.

A release connection check used the real CLI/model and successfully executed a
robot Python tool that read the two controller configurations, returned public
observations, and closed the native worker normally. That short check executed
no physical movement and is not a complete task rollout or benchmark score.

The integration retains the project's original experimental App Server wire
protocol. Match your CLI version and consult the [official App Server
documentation](https://learn.chatgpt.com/docs/app-server) for its versioned
protocol. Source identities are recorded in [provenance.py](provenance.py).
