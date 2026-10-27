# Run the Codex Robo Agent

This walkthrough runs Codex on **Prepare coffee, saved episode 1**. Complete
[shared setup](../../../../docs/MODEL_QUICKSTART.md) first and retain its shell
variables. The native worker runs in the prepared Docker image; Codex and its
login stay on the Linux host. A second, restricted Docker container executes
robot Python tools. It receives neither account credentials nor network/GPU
access. Scoring, physics and robot tools remain unchanged.

## 1. Install Codex and sign in

Follow the [official CLI installation](https://learn.chatgpt.com/docs/cli)
for Linux, then use the [ChatGPT login flow](https://learn.chatgpt.com/docs/auth):

```bash
codex login
codex login status
codex --version
export RAB_CODEX="$(command -v codex)"
```

Use your own account with access to `gpt-6-astra`. This recipe uses that model,
`medium` effort and a 200-response budget; it does not switch providers/models
or use an API-key fallback. Running it consumes account usage and may incur
charges. Configuration records the exact CLI version and rejects a subsequent
version change. The integration was checked with `codex-cli 0.164.0-alpha.11.1`;
it uses experimental App Server dynamic tools, so a different CLI must support
that protocol rather than merely report a version.

## 2. Build the robot-tool sandbox

Run as the non-root Docker user from shared setup:

```bash
cd "$RAB_ROOT"
"$RAB_PY" -m robotagentbench.model_examples.codex.sandbox_build --export sandbox-build
docker build --platform linux/amd64 -t robotagentbench-policy:local sandbox-build
"$RAB_PY" -m robotagentbench.model_examples.codex.sandbox_build \
  --image robotagentbench-policy:local --docker-host "$DOCKER_HOST"
```

This builds the included Dockerfile and locked numerical/kinematics
dependencies. It is **not the Isaac image**. The following optional check runs
only the isolated Python transport and FK on synthetic state, without Codex,
Isaac, physical actions or scores:

```bash
"$RAB_PY" -m robotagentbench.model_examples.codex.sandbox_smoke \
  --image robotagentbench-policy:local --docker-host "$DOCKER_HOST" \
  --output "$RAB_ROOT/model-configs/sandbox-check01"
```

## 3. Stage a coding workspace, then restore the episode

Stage **before** restoring: saved absolute paths must belong to the coding
Runtime, not its provider. The provider Runtime is left untouched.

```bash
"$RAB_PY" -m robotagentbench.model_examples.codex.stage_runtime \
  --runtime "$RAB_ROOT/Runtime" --destination "$RAB_ROOT/codex-workspace" \
  --policy-image robotagentbench-policy:local --docker-host "$DOCKER_HOST" --stage

"$RAB_PY" -m robotagentbench.model_examples.docker_workflow restore \
  --runtime "$RAB_ROOT/codex-workspace/Runtime" --image robotagentbench:native \
  --release "$RAB_ROOT/SavedConfigurations/release.json" \
  --task prepare_coffee --episode 1 --output "$RAB_ROOT/Restored/codex-coffee01"
```

Restoration uses the CPU-only native image. It does not call Codex, start
physics or resample the scene. The selected episode matches the InternVLA
example's task/episode. The staged source contains the original coding tools
and the immutable ID of your local policy-sandbox build.

## 4. Configure both sides

Choose an available native GPU (GPU 0 below). No model GPU is needed for the
hosted Codex service.

```bash
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow configure \
  --agent codex --image robotagentbench:native \
  --selection "$RAB_ROOT/Restored/codex-coffee01/selection.json" \
  --name codex_coffee01 --gpu 0 --lock-root "$RAB_LOCKS" \
  --codex-executable "$RAB_CODEX" \
  --destination "$RAB_ROOT/model-configs/codex-coffee01"
```

The helper prepares fresh private native caches and writes the worker/operator
configurations. It automatically uses the selected launch, read-only inputs
and top-level `public_input_binding`. There are no host Isaac paths or JSON
placeholders to fill. This step does not call a model or start physics.

The trusted native owner receives the operator's Docker CLI/socket to create
the restricted tool container. The **model's tool container does not receive
that socket**, the simulator, the SDK or account credentials.

## 5. Inspect, then run

```bash
# Inspect only: no model or simulator starts.
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow run \
  "$RAB_ROOT/model-configs/codex-coffee01/run.json"

# After accepting the NVIDIA terms in shared setup; uses your Codex account.
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow run \
  "$RAB_ROOT/model-configs/codex-coffee01/run.json" --execute
```

The printed output directory contains `operator_result.json`, public tool
observations, `observations/codex_report.json`, and `native/native_report.json`.
Read the native outcome for score and success; a model saying it is done does
not establish task completion. Use a new name/configuration for another run;
uncertain physical actions are never retried automatically.

Codex receives the public instruction/context, camera images, measured robot
state and explored navigation map. It does not receive answer bindings, hidden
evaluator state or scores as policy input. The release short check verified a
real acknowledged tool call, not full physical task completion.

## Other deployments

The guided example uses the standard room/short-task worker. Dedicated cloth
and specialized owners retain their own interfaces. For an already managed
native SDK, remote Linux worker, or explicit custom container configuration,
see [advanced deployment](NATIVE_SETUP.md). Do not copy account credentials
into the simulator or policy image.
