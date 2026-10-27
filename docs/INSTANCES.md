# Saved initial configurations

Install the shared assets once, then choose one of 30 saved initial
configurations for a task or scenario. Different agents can use the same saved
configuration. Running an evaluation does not select another initial state.

## Evaluation sets

| Family | Groups | Configurations per group | Total |
| --- | ---: | ---: | ---: |
| Manipulation: short tasks | 18 | 30 | 540 |
| Manipulation: long tasks | 8 | 30 | 240 |
| Task Discovery | 4 scenarios | 30 | 120 |
| Context Reasoning | 4 tasks | 30 | 120 |
| Failure Recovery | 5 tasks | 30 | 150 |
| Robustness | 6 variants | 30 | 180 |
| OpenEnd | 4 scenarios | 30 | 120 |
| **Total** | **49** | **30** | **1,470** |

These release configurations are for reproducible evaluation. They are **not
the exact instances used for the paper's reported results**. Report the code,
asset and configuration-bank versions, model checkpoint, task identifiers and
episode numbers with new results.

Episode numbers 1–30 identify saved configurations. Each also records the
generator's `actual_seed`; accepted seeds need not be consecutive.

## What is stored

The configuration bank contains small state records: the selected shared
layout and asset identifiers, task parameters, object transforms, robot base
pose and joint initialization, and any task-owned initial-state attachments.
Cloth tasks retain their required material-state arrays. Public scenario text
is stored separately from evaluator bindings.

It does **not** distribute a complete room/mesh/texture copy per configuration.
Room geometry, furniture, lights, robot and task objects come from the shared
[asset release](ASSETS.md). The native environment assembles a local scene
from those resources and the saved state when preparing a run. These generated
runtime files are not additional downloads.

## 1. Install code and shared assets

Follow [Installation](INSTALL.md), then configure a fresh task workspace:

```sh
robotagentbench assets fetch resources/assets-0.1.0.json --cache downloads
robotagentbench assets install resources/assets-0.1.0.json --cache downloads --destination Assets
robotagentbench assets configure resources/assets-0.1.0.json --destination Assets --runtime Runtime --profile room
```

Use the task's matching profile for dedicated cloth or specialized owners;
do not replace those owners with the ordinary room worker.

## 2. Download the configuration bank

The bank is approximately **232 MB** across six archives; the shared simulation
assets are installed separately and are not duplicated in this download.

```sh
python -m robotagentbench.simulation.saved_instance_release.download \
  --manifest resources/saved-instances-0.1.0.json \
  --cache downloads/configurations --destination /absolute/path/to/SavedConfigurations
```

The manifest pins archive sizes, member hashes and an immutable Hugging Face
revision. Downloads reuse verified archive parts. Installation requires a new
destination and rejects unexpected members or unsafe paths. For offline
installation, supply the archives at their manifest-relative cache paths and
add `--offline`.

## 3. Restore a selected initial state

For the model walkthroughs, use the [shared Docker setup](MODEL_QUICKSTART.md)
and let the image supply the native SDK:

```bash
"$RAB_PY" -m robotagentbench.model_examples.docker_workflow restore \
  --runtime "$RAB_ROOT/Runtime" --image robotagentbench:native \
  --release "$RAB_ROOT/SavedConfigurations/release.json" \
  --task prepare_coffee --episode 1 --output "$RAB_ROOT/Restored/coffee01"
```

For Codex, stage the coding Runtime first and use that Runtime here, as shown
in its walkthrough. Restoration is CPU-only and does not start physics.

Advanced operators with an existing matching native Python/SDK environment
can instead invoke the original restorer directly:

```sh
python -m robotagentbench.simulation.saved_instance_release.restore \
  --release /absolute/path/to/SavedConfigurations/release.json \
  --runtime /absolute/path/to/Runtime \
  --task prepare_coffee --episode 1 \
  --output /absolute/path/to/Restored/coffee01 \
  --sdk-site /absolute/path/to/native/site-packages --execute
```

Omit `--execute` to describe the selection without writing files or importing
USD. Execution prepares the scene from the saved recipe and restores the saved
initial state; it does not run a model, start physics, or resample the episode.
Use the native CPython 3.11 environment specified in [Installation](INSTALL.md).
For containers, run this command inside the configured source/asset namespace.

For cloth and specialized tasks, also pass `--owner-config OPERATOR` with the
matching dedicated environment configuration from [Cloth](CLOTH.md) or
[specialized owners](SPECIALIZED.md). Restoration reuses that owner's native
settings and verification; it does not substitute the ordinary room owner.

The output contains `selection.json` and a restoration receipt. Generated task
files are placed automatically in the original task owner's fresh Runtime
directory. Existing output directories are never overwritten. Keep the
receipt with the evaluation result.

## 4. Run an agent

Use the Runtime and prepared launch from `selection.json` with the task's
original worker. Where provided, forward the selection's read-only input roots
and hash-bound public instruction/context. Operator metadata, task manifests,
answer bindings and private evaluator state are not policy observations.

For the ordinary restored room/short-task format, `runtime`, `source_root`,
`launch` and `public_input_binding` are **top-level fields** in `selection.json`.
There is no `operator.public_input_binding` nesting in this format. The model
walkthrough helper reads those fields and supplies the trusted source root and
public-text binding automatically; do not substitute a target/answer file.

- [InternVLA-A1.5 example](../src/robotagentbench/model_examples/internvla/README.md)
- [Codex Robo Agent example](../src/robotagentbench/model_examples/codex/README.md)
- [Cloth owner](CLOTH.md) and [specialized owners](SPECIALIZED.md)

Use a fresh run output, the host's shared exclusive-GPU lock directory, and
normal worker shutdown. Restoring an initial state is not a task result;
process scores and completion come from verified physical execution by the
original task owner.
