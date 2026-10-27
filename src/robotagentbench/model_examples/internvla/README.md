# InternVLA-A1.5 example

Use `run` for the server-derived path. The earlier strict/no-projection candidate remains unchanged as a comparison in [STRICT_CANDIDATE.md](STRICT_CANDIDATE.md); its `serve`/`bridge` commands are **not** the server baseline.

## Reused behavior

The actual G1 model server, launcher, client loop and action adapter were read in full. [_server_projection.py](_server_projection.py) is byte-identical to the server ActionAdapter (SHA256 `6939b1be0848261afb03829f4c29de57d7ff58d6b3f3d3b69ca7493a5202ad45`). It preserves joint clipping, base clipping, inclusive 0.001m/s / 0.005rad/s deadbands, straight/turn exclusivity, braking, acceleration limits, audit records and commit of acknowledged rows.

[server_policy.py](server_policy.py) preserves measured state23, full native head1280×800 + wrist848×480 RGB, resize-pad224, state mean/std, padded32, BF16 optimized action-only inference, seed42, per-chunk policy reset, 50×32 predictions and NumPy mean/std unnormalization (epsilon1e-6) once. The 22 supervised channels remain 20 absolute native joint targets plus forward m/s and yaw rad/s. Torso lift is metres; other joints and continuous outer grippers are radians. No RoboTwin compaction, delta integration or gripper0–1 conversion is introduced. Worker control remains20Hz.

The real Discovery launcher replaced its training-index table with one public instruction row. This extraction preserves that lookup/assert behavior using `public:<task_id>` and the worker's current public instruction. No private answer, GT target list or scoring state is read. Separate context/map packets are preserved but, like the original server, not separately consumed. Household context already inside instruction remains inside that same model prompt.

An optional explicit `public_prompt` supports the old Short14 model-side training-prompt override. Do not assume those fixed historical prompts match newly randomized objects. Default `null` uses current worker instruction; adding separate context would be an explicit new prompt revision, not baseline parity.

See [SERVER_REUSE.md](SERVER_REUSE.md) for the source identities and model input/action contract.

## Install and configure once

Install RobotAgentBench in the operator, native-worker and separate model Python environments. Follow the [upstream model installation guide](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/tutorials/installation.md) for the model environment.

Download our fine-tuned model from **[AetherLabs-AI/InternVLA-A1.5-RAB](https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB)**:

```sh
python -m pip install huggingface_hub
hf download AetherLabs-AI/InternVLA-A1.5-RAB --local-dir weights/InternVLA-A1.5-RAB
```

If the model repository requires access, sign in with `hf auth login` and obtain access before downloading. Follow its model card for the checkpoint, matching normalization statistics, control schema and VLM paths. Pass their **absolute local paths** below; these files must belong to the same G1 model release. Generic base/RoboTwin weights are not interchangeable with this G1 checkpoint. The runner loads local files and never downloads or substitutes weights automatically.

For the native worker, use a complete, separately installed Isaac SDK and its matching Python interpreter as described in [Installation](../../../../docs/INSTALL.md). A Docker-only deployment additionally needs an explicit trusted container namespace; generic `--prepare-native` creates caches, not a host SDK installation.

```sh
python -m robotagentbench.model_examples internvla configure \
  --runtime /absolute/path/to/Runtime --sdk-site /absolute/path/to/native/site-packages \
  --launch /absolute/path/to/Runtime/source/saved/launch.json \
  --native-python /absolute/path/to/native-env/bin/python3.11 \
  --model-python /absolute/path/to/model-env/bin/python \
  --checkpoint /absolute/path/to/G1/pretrained_model \
  --stats /absolute/path/to/G1/stats.json --stats-key G1_omnipicker \
  --control-schema /absolute/path/to/G1/control_schema.json --vlm /absolute/path/to/local/VLM \
  --lock-root /absolute/path/to/shared/gpu_locks --worker-gpu 0 --model-gpu 1 \
  --task short01 --output /absolute/path/to/Runtime/source/run001 --config internvla-run.json
```

This checks local files, SDK directory inventory and the existing saved launch, then writes a fresh configuration and SHA receipt. It does not import Isaac/Torch, run a generator, sample a task, launch a worker or load weights. Artifact hashes are computed automatically; no manual SHA entry, worker argv or pipe-FD plumbing is required. `--task` is a public selection label for this saved launch, not a training index, and does not change its objects or targets.

For a materialized [saved configuration](../../../../docs/INSTANCES.md), use its
selected Runtime and launch inside the trusted read-only execution namespace.
Pass each prepared input directory with `--input-root`. If the selection
provides `operator.public_input_binding`, also pass
`--public-input-binding /absolute/path/to/operator/public_input_binding.json`.
This preserves the selected public instruction/context without exposing the
underlying subtask list or evaluator bindings.

The generated trusted argv selects `python -m robotagentbench.model_examples.native_worker` with the installed Runtime, SDK, saved launch, lock directory and GPU. Native output/report live at `{output}/native` and `{output}/native/native_report.json`. The **operator output must be inside this Runtime's source root**, must be fresh, and its parent must exist. Configuration and run enforce the workspace boundary. The launch must already be inside that same source root. Preserve the read-only namespace, native lease, initial admission, independent watchdog and delayed close/cleanup. For Docker retain `-i` without a TTY.

This automatic entry supports ordinary room/short raw22 owners only. Specialized Recovery/Robustness, cloth and coding owners require their dedicated bindings; this command does not bypass a bound-required rejection. Downloading assets/weights alone does not configure Isaac or every family. `server-template` remains available for an expert's explicit dedicated-worker integration; it leaves argv empty and is not a runnable deployment.

## Run one selected task

```sh
python -m robotagentbench.model_examples internvla run \
  --config internvla-run.json --task short01 \
  --output /absolute/path/to/Runtime/source/run001
```

The operator starts that existing worker, reads ready, creates its single public prompt binding, starts the model service, runs full-chunk prediction→original projection→execute→acknowledge, and closes owned processes. It reuses the project GPU-lease implementation for the model; the existing worker keeps its own native lease. Raw predictions, applied rows, projection reasons, acknowledgments and cleanup results are saved. A fresh output is mandatory; no automatic retries or repeated completed IDs.

For a pre-existing protected model service, set `model_endpoint` to its `http(s)://.../infer` URL. It must already implement the **original** infer/actions payload with the same public prompt binding, episode and step. Its checkpoint bytes/GPU lifecycle are external and not verified/stopped locally. This is not the old strict HTTP protocol, and no upstream standard HTTP API is implied.

Native cleanup is accepted only after original delayed closed acknowledgment, exit0 and the project's `shutdown_succeeded` report. An uncertain action is never resent. A hung/uncertain owner is left to the existing independent supervisor; no unrelated process/container is stopped. Cleanup completion is not task success.

## Verification

```sh
python -m unittest robotagentbench.model_examples.internvla.test_server_parity
python -m unittest robotagentbench.model_examples.internvla.test_configure
```

Source-checkout-only CPU parity tests compare the original snapshot against the extracted projection. Private source snapshots are not installed, so those checks skip when absent. These tests do not load weights or run a simulator.

The configuration tests use synthetic filesystem fixtures and check path/hash boundaries without native execution.

A release connection check with the original G1 checkpoint performed one real model inference and executed its 50-action chunk through the native worker, then closed both processes normally. This checks the connection, not full task success or performance across the 30 saved configurations. Record the checkpoint, statistics, code revision and selected episode when evaluating a different model download.

Official dependency: [InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series), inspected at `e6fc904f9edbfb14532e97095fc2372202517f76`, under its own [CC BY-NC-SA4.0 licence](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/LICENSE). No model implementation, weights, statistics or dataset is bundled.
