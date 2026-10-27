# InternVLA model interface

Start with the [installation and run guide](README.md). This reference describes
the G1 adapter used by `internvla run`.

## Inputs and actions

| Interface | Contract |
| --- | --- |
| State | 20 command-joint channels, world-frame XY and continuous yaw; float32 |
| Images | Head 1280×800 and two wrists 848×480, RGB; resize/pad to 224 |
| State transform | Mean/std normalization and padding to 32 channels |
| Inference | Action-only BF16, seed 42, policy reset for each chunk |
| Prediction | 1×50×32; first 22 channels unnormalized once as `value * (std + 1e-6) + mean` |
| Commands | 20 absolute joint targets, forward velocity in m/s, yaw rate in rad/s |
| Joint units | Torso lift in metres; angular joints and continuous outer grippers in radians |
| Execution | Original 20 Hz worker and action projection; no delta integration or gripper 0–1 conversion |
| Public text | Current worker instruction, bound to `public:<task_id>` |
| Extra observations | Map/context packets remain available but are not separately encoded by this adapter |

The same native camera dimensions apply to every task family, including
Robustness, Failure Recovery and tablecloth tasks. Full frames go directly to
the model's existing 224×224 resize/pad transform, with no extra crop or resize.

The original action projection retains joint/base clipping, inclusive
0.001 m/s and 0.005 rad/s deadbands, straight/turn exclusivity, braking and
acceleration limits. Projected rows are committed only after acknowledgment.
Raw predictions, applied commands and projection events are recorded separately.

## Prompt selection

By default `public_prompt` is `null`, so the current worker instruction is used.
Context already present in that instruction remains part of the model prompt.
An explicit override is available for reproducing a matching older experiment.
Do not reuse fixed historical object/colour prompts for randomized instances.
Ground-truth answers and private task bindings are not model inputs.

## Advanced deployments

For a pre-existing model service, set `model_endpoint` to its protected
`http(s)://.../infer` endpoint. It must implement this adapter's `infer/actions`
protocol with matching public-prompt, episode and step bindings. This is not
an upstream standard API. External checkpoint identity and GPU lifecycle are
the operator's responsibility; the client does not stop an external service.

`server-template` provides an empty worker-argv template for dedicated-worker
integration, not a runnable deployment. Retain the task's required bindings,
read-only inputs, GPU lease, admission and watchdog.

Native cleanup requires the original delayed `closed` acknowledgment, process
exit 0 and a `shutdown_succeeded` report. Uncertain actions are not resent;
the independent supervisor handles a hung owner. Cleanup is separate from
task success.

## Source compatibility

The action projection is unchanged from the server adapter:
SHA256 `6939b1be0848261afb03829f4c29de57d7ff58d6b3f3d3b69ca7493a5202ad45`.
Preprocessing, decoding and public-prompt routing retain the server recipe.
Deployment paths are supplied through configuration.

CPU configuration checks do not load a model or simulator:

```sh
python -m unittest robotagentbench.model_examples.internvla.test_configure
```

Source-parity tests additionally require original review snapshots. Those
snapshots are not runtime dependencies or part of the public package.
