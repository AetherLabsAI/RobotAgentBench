# InternVLA-A1.5: G1 fine-tuned model example

This is an optional model client, not another simulator or evaluator. It connects an already admitted RobotAgentBench JSONL worker to a local InternVLA-A1.5 policy service. No model or GPU is loaded by imports, `template`, `verify`, or the CPU tests. The example has **not** been run with a real model as part of its release checks.

## Required model, not interchangeable base weights

Use an InternVLA-A1.5 checkpoint fine-tuned for `g1_omnipicker_jointpos20_basevw_v1`, with its matching `stats.json` and training `control_schema.json`. The generic `InternRobotics/InternVLA-A1.5-base` and RoboTwin checkpoints are not declared G1-compatible. A 32-wide tensor is padded model space, not a robot action definition.

The supported recipe is explicit:

- Three synchronized RGB inputs, `head`, `left`, `right`, retaining the full frame before the official resize-and-pad to 224×224.
- Measured state: the worker's 20 named command joints, then world base X, Y, and continuous unwrapped yaw (23 values). No zero-filled base state or quaternion-to-yaw re-interpretation.
- Model output: 50×32 padded normalized values; the bound training schema declares the leading 22 supervised channels. Official `UnNormalizeTransformFn` reverses mean/std normalization once, using its `1e-6` epsilon. No RoboTwin compaction or delta addition.
- Actions: 20 absolute native joint-position targets, followed by forward m/s and yaw rad/s. Torso lift is metres; other joints, including continuous outer grippers, are radians. Large gripper angle is open, small is closed; neither 0–1 normalization nor an observed grasp is implied.
- The existing worker executes at 20Hz. This client does not change physics rate, camera parameters, clocks, limits, guards or scoring.

Public `context` is passed through the official `language_memory` field. The current public instruction is used directly; no task-index table, private answer, target lookup or submitted plan is consulted. The allowlisted explored `navigation` map stays in the HTTP observation packet but this three-camera G1 policy does **not** consume it (`navigation_consumed=false`). Do not advertise map-conditioned inference.

## Install the model environment separately

Use the [official installation guide](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/tutorials/installation.md) in a separate environment; it documents Python 3.11, CUDA 12.8, Torch 2.10.0, Transformers 5.2.0, attention kernels, and the required Transformers integration. RobotAgentBench does not install or patch these packages. Use local model/VLM files; the example requests offline loading and does not download weights. Allocate an inference GPU through your operator's existing resource policy before starting `serve`; this module does not reserve or share GPUs.

## Configure and verify

```sh
python -m robotagentbench.model_examples.internvla template > internvla.json
```

Edit the checkpoint, stats, control-schema, local VLM, device and endpoint paths. Fill `artifacts_sha256` with the real SHA256 of checkpoint `config.json`, `model.safetensors`, matching statistics, and control schema. `verify` checks local bytes and the G1 codec; it does not load a policy. Sharded weights, alternative statistics/normalization, delta controls, unknown channel orders or different robot schemas intentionally fail closed and need a separately audited adapter.

```sh
python -m robotagentbench.model_examples.internvla verify --config internvla.json
python -m robotagentbench.model_examples.internvla serve --config internvla.json --port 8765
```

`serve` is the only command here that loads a model. Its `/infer` endpoint is this example's adapter protocol, **not an upstream standard HTTP API**. The service binds to loopback without authentication; for another host use an operator-protected tunnel, not a publicly exposed listener. Set client `--endpoint` to the resulting local/tunnel URL. `endpoint` in the configuration describes the intended client endpoint; `--port` controls the actual loopback listener.

## Attach the client to existing worker pipes

Your existing operator starts the authorized worker and hands its read/write pipes to the client. Keep Docker stdin open (`-i`, no TTY), preserve JSONL framing and keep worker diagnostics on stderr. No simulator process is started by this module.

```sh
python -m robotagentbench.model_examples.internvla bridge \
  --endpoint http://127.0.0.1:8765/infer \
  --worker-read-fd 3 --worker-write-fd 4 --max-calls 100
```

Descriptors 3 and 4 must already be inherited pipes; the command is not a shell pipeline that creates them. For programmatic integration use `bridge.run(worker.stdout, worker.stdin, Client(url, timeout), max_calls=...)`. The client reads the original `ready`, sends a UUID-bound `execute_actions` against the exact episode/step, waits for its result, and ends with the original `close`/`closed` exchange. Operator-side final scores are never sent to the policy.

Every row is validated before any row is sent: finite values, native measured joint limits, original base speed/acceleration, straight-or-turn exclusivity and stop-before-mode-switch. The worker remains the final authority for follower/collision/other native checks. There is no silent clipping, gripper conversion, deadband, base projection, action resampling or retry. A raw prediction that violates these constraints is rejected; valid tensor shape alone does not promise executable actions. Historical evaluation harnesses may have applied a separate base projection; this strict example does not claim numerical replay of those runs.

On transport failure or partial/uncertain execution the client stops without resending. The external worker owner/watchdog must settle ownership and cleanup. A failed client never claims the simulator was cleaned up.

## CPU-only checks

```sh
python -m unittest robotagentbench.model_examples.internvla.test_contract
```

These use synthetic observations/actions and a stub loopback HTTP service. They check allowlisting, state order, stale/normalized/nonfinite output rejection, whole-chunk rejection, map preservation and worker framing. They are not checkpoint inference, physical acceptance or benchmark results.

## Sources and scope

Official API references checked at commit `e6fc904f9edbfb14532e97095fc2372202517f76`:

- [RoboTwin inference](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/evaluation/RoboTwin/inference.py): configuration loading, transforms, `predict_action_chunk`; its robot-specific output compaction is deliberately not used here.
- [Transforms](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/transforms/core.py): resize/pad, state normalization, padding and unnormalization.
- [Chat transform](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py): task and public language-memory inputs.

The G1 order/units/state and 50×32→50×22 recipe were additionally checked against the project's existing G1 training control schema and actual historical A1.5 model-serving source/ready receipt, not the third-person trajectory replay. That source's training-task-index prompt lookup and GPU launcher are not imported. No upstream source, checkpoint, statistics, dataset or weights are bundled. Upstream InternVLA code has its own [CC BY-NC-SA 4.0 licence](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/LICENSE); installing that optional dependency does not change its terms.
