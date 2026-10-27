# InternVLA-A1.5 example

Run our fine-tuned G1 policy on an existing RobotAgentBench instance using
`internvla configure` followed by `internvla run`. The client retains the
original model preprocessing and action projection; scoring and physics
are unchanged.

## Prerequisites

- Install RobotAgentBench, shared assets, and an independent native environment
  following [Installation](../../../../docs/INSTALL.md).
- Select a [saved configuration](../../../../docs/INSTANCES.md). The policy
  reuses this initial state; it does not generate a new one.
- Prepare a separate model Python environment using the
  [upstream installation guide](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/tutorials/installation.md).
  Install RobotAgentBench in the operator, native-worker and model environments.
- Reserve separate GPUs for the native worker and model, using the same
  host-wide GPU-lock directory as other RobotAgentBench jobs.

The automatic entry below supports the standard room/short-task raw22 worker.
Cloth and specialized Recovery/Robustness workers require their dedicated
bindings. A Docker-only installation also needs an explicit container
namespace: preparing Kit caches does not install an SDK on the host.

## 1. Download the model

Download **[AetherLabs-AI/InternVLA-A1.5-RAB](https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB)**:

```sh
python -m pip install huggingface_hub
hf download AetherLabs-AI/InternVLA-A1.5-RAB --local-dir weights/InternVLA-A1.5-RAB
```

If access is required, obtain it and sign in with `hf auth login` first. Use
the model card to locate the checkpoint, matching G1 normalization statistics,
control schema and VLM files. These must belong to the same model release;
generic base or RoboTwin weights are not interchangeable. The runner loads
local files and does not silently download or substitute weights.

## 2. Configure one saved instance

Replace the placeholders below with **absolute local paths**. The launch and
output must be inside the selected Runtime's source root. The output must not
exist, and its parent directory must already exist.

```sh
python -m robotagentbench.model_examples internvla configure \
  --runtime /absolute/path/to/Runtime --sdk-site /absolute/path/to/native/site-packages \
  --launch /absolute/path/to/Runtime/source/saved/launch.json \
  --native-python /absolute/path/to/native-env/bin/python3.11 \
  --model-python /absolute/path/to/model-env/bin/python \
  --checkpoint /absolute/path/to/G1/pretrained_model \
  --stats /absolute/path/to/G1/stats.json --stats-key G1_omnipicker \
  --control-schema /absolute/path/to/G1/control_schema.json --vlm /absolute/path/to/local/VLM \
  --lock-root /absolute/path/to/shared/gpu-locks --worker-gpu 0 --model-gpu 1 \
  --task short01 --output /absolute/path/to/Runtime/source/run001 --config internvla-run.json
```

For a materialized saved instance, retain its trusted read-only execution
namespace and pass each prepared input directory with `--input-root`. When
the selection includes `operator.public_input_binding`, also pass
`--public-input-binding /absolute/path/to/operator/public_input_binding.json`.
This supplies the selected public instruction and context, not evaluator answers.

Configuration checks the files and records their hashes. It does not start
Isaac, load the model or resample the task. `--task` labels the selected launch;
changing this label does not change its objects or targets.

## 3. Run

```sh
python -m robotagentbench.model_examples internvla run \
  --config internvla-run.json --task short01 \
  --output /absolute/path/to/Runtime/source/run001
```

The client starts the worker and model service, reads observations, predicts
and projects action chunks, sends each chunk once, and records acknowledgments.
It closes its owned processes when the run ends. Use a fresh output directory
for every run; uncertain actions are not retried.

## Observations, actions and results

The policy uses the current public instruction, measured 23-dimensional robot
state and all three RGB cameras. It predicts 50-action chunks, decoded into
22 native command channels and executed through the original 20 Hz projection.
The worker also supplies map/context fields, but this model adapter does not
encode those separate fields as additional model inputs.

Outputs contain predictions, applied actions, projection records,
acknowledgments and cleanup results. Read scores and termination from
`native/native_report.json`; a clean shutdown is not task completion.
See the [model interface](SERVER_REUSE.md) for shapes, units and advanced
integration.

A release connection check executed one real 50-action model chunk and closed
normally. It was a connection check, not a full-task benchmark result.

## License

The [upstream model implementation](https://github.com/InternRobotics/InternVLA-A-series/tree/e6fc904f9edbfb14532e97095fc2372202517f76)
retains its [CC BY-NC-SA 4.0 license](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/LICENSE).
Weights and normalization data are downloaded separately under their own
terms, not bundled with the evaluator.
