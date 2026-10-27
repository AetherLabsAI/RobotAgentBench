# Run the coffee example

This example uses the included `prepare_coffee` task, its original request and
ready pose, and the matching asset release. Native execution requires Linux,
Docker, NVIDIA Container Toolkit and an Isaac Sim-compatible NVIDIA GPU.

## 1. Install code and assets

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
robotagentbench assets fetch resources/assets-0.1.1.json --cache downloads
robotagentbench assets install resources/assets-0.1.1.json --cache downloads --destination Assets
robotagentbench assets configure resources/assets-0.1.1.json --destination Assets --runtime Runtime --profile room
```

For offline setup, place all release archives in `downloads` and skip `fetch`.

## 2. Build the native environment

```sh
docker build -f docker/Dockerfile -t robotagentbench:native .
```

The image installs the pinned Python, Node.js and native dependencies. It does
not contain task assets or accept NVIDIA license agreements on your behalf.
See NVIDIA's [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
for the simulator's requirements and license terms.

## 3. Inspect and prepare an episode

```sh
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native --plan
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native --prepare-native --name coffee_01
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native --prepare --name coffee_01
```

`plan` prints the task instruction, score maximum and selected layout.
`prepare-native` copies the image's Kit into a fresh episode-owned directory,
creates an empty MDL cache and the original NVIDIA ICD configuration, and seals
them to the image's immutable ID. The host does not need an SDK installation.
`prepare`
creates the scene, initial-state records and launch file in the local workspace.
These commands use a CPU-only container. Choose a new episode name for a new
preparation; existing episodes are not overwritten.

Native execution mounts only that private Kit/MDL pair writable inside the
otherwise read-only image. Its original SDK core hashes are checked before
startup. Use the same image and episode name for both preparations and the run;
do not share an episode's caches between concurrent runs.

The namespace launcher maps the locally installed resources to the task owner's
logical paths inside the container. No original server checkout is needed, and
no legacy directories are created on the host.

## 4. Start the task

After reviewing and accepting NVIDIA's license terms, set the acceptance
variable as described in its installation guide. Create a dedicated shared
GPU-lock directory, then run:

```sh
mkdir -p gpu-locks
python -m robotagentbench.namespace --runtime Runtime --image robotagentbench:native \
  --run --name coffee_01 --gpu 0 --lock-root gpu-locks
```

The worker emits a JSONL `ready` message and accepts the common robot action
protocol over stdin. To request an observation and close the episode:

```json
{"op":"observe"}
{"op":"close"}
```

Keep stdin open while an agent is interacting with the worker. The runtime
returns synchronized camera observations, measured robot state, supplied public
context, and a progressively explored static map. Coffee preparation has no
additional household memory, so its `context` is empty. The worker evaluates
actions at 20 Hz, and writes scores and terminal results under the episode's
`evaluation` directory. Use the [runtime guide](RUNTIME.md) to connect an agent
or recorded action source.

`ready.contract.observation_fields` declares the returned fields. The map image
is `observation.navigation.map_png` (base64 PNG); its `origin_m`, `resolution_m`
and `shape` describe the metric grid. Map PNGs are also recorded under
`evaluation/navigation/`. Internal geometry in `private_geometry/` is not model
input. Neither observations nor map reads execute a navigation action.

All jobs on a host must use the same GPU-lock directory. The worker checks
occupancy and refuses an occupied device; the launcher does not stop other jobs.
Use `--print-command` to inspect the container command without executing it.

## 5. Connect an agent

The worker alone does not load a model. To evaluate a policy, use one of the
provided clients with an existing saved launch:

- [InternVLA-A1.5](../src/robotagentbench/model_examples/internvla/README.md):
  download the fine-tuned checkpoint from
  [AetherLabs-AI/InternVLA-A1.5-RAB](https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB),
  configure the native worker and matching model environment, then run
  `python -m robotagentbench.model_examples internvla run --config CONFIG --task TASK --output OUTPUT`.
- [Codex Robo Agent](../src/robotagentbench/model_examples/codex/README.md):
  stage the included coding source profile, configure your native worker and
  Codex login, then run
  `python -m robotagentbench.model_examples.codex OPERATOR_JSON --execute`.

See [saved configurations](INSTANCES.md) to choose one of 30 existing initial
states instead of generating a new one. These saved release instances are
separate from the paper's evaluation set. The model guides describe their
native SDK/container prerequisites; the coffee cache preparation above does
not automatically set up a model environment.
