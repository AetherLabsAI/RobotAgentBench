# Server source and packaging delta

This example extracts the original G1 policy-serving recipe. A release connection check subsequently performed one actual inference, executed 50 actions through the native worker, and closed the model and native processes normally. It is a short connection check, not a full task rollout or a success-rate measurement.

## Sources inspected in full

| Original role | SHA256 |
| --- | --- |
| Actual Discovery G1 serve_intern.py | a2e030601ee9531de6780a3a5220eb482ff523507acb440246590f56b9a5b3ea |
| Original shared ActionAdapter | 6939b1be0848261afb03829f4c29de57d7ff58d6b3f3d3b69ca7493a5202ad45 |
| Shared model/worker launcher and JSONL RPC | 83e734db2973b97e9399c97661a488af0441565b8e94390927116b0a305d4980 |
| Discovery rollout loop | 17b0a70eb6d05de9c29e8d0400ec5424ae61f68eaecee87b82a3c031f686dffe |
| Discovery public-prompt binding/control | c23ca2ba892a55b3901f4803be01c1c64ee7ebfddc870020ac768ac4fe168c96 |
| Original G1 training control schema | 256c212eb078ce43c218960859d1d3cd75cc7ce60b377e78a93f1a704624f03c |

The historical Discovery receipt recorded 72 actual model calls and 3,600 acknowledged action rows; native evaluation was valid, score0, original no-progress timeout. This is real policy-serving evidence, not the later third-person trajectory rendering. It does not establish performance for new tasks/models.

The Discovery model directory contained one rewritten public prompt row (SHA24a423e72f4ce53d9427a5ad84d7511231ac422fe3dfa2848bb8511dc7831f8a), not the old126-entry training table (SHA7e4a6bc3921da50b0549ed45a419a851587b15cd7de3bd4a75cb80fd3f898130). Thus retaining a public routing label and the exact current public instruction does not require training task indices or ground truth.

Private exact source snapshots are retained in this source-review checkout with non-package `.py.private` suffixes. They include historical deployment paths and are not installation inputs. Public code has no dependency on those paths.

## Behavior retained and explicit deltas

| Area | Retained | Packaging change / limitation |
| --- | --- | --- |
| State | named command-joint20 + worldXY + unwrapped yaw, float32 | Paths are operator parameters; native34 is allowlisted before inference |
| Images | head1280×800, wrists848×480; RGB CHW /255; original resize/pad224 | Smaller-camera profiles rejected by tested model path |
| Transform | original state Normalize default mean_std; chat tokenize_state; padded32 | VLM root parameterized; no new language_memory field |
| Model | optimized action-only; cuda BF16; seed42; reset/full predict_action_chunk | Separate installed model Python replaces server-specific Docker image/kernel mounts |
| Decode | shape1×50×32; NumPy first22 ×(std+1e-6)+mean | stats path/key parameterized; no manual SHA input |
| Prompt | public allowlist lookup and exact assertion | public:<taskID> routing label replaces Discovery-specific key; optional explicit public historical prompt override |
| Context/map | fields retained in public observation, original model ignores separate extras | No new context/map-conditioned claim; prompt revision required to newly consume them |
| Projection | complete original ActionAdapter bytes, raw/applied/events, pending/commit | none; earlier strict candidate is separate and not the baseline |
| Worker calls | full chunk once, UUID/episode/step binding, acknowledge executed rows, no auto retry | configure generates the project's native_worker argv for installed ordinary room/short raw22 saved launches; dedicated families remain unsupported by that entry |
| GPU lifecycle | reuse project UUID lease for model, worker retains original own lease | no old shared-GPU override; operator uses common existing lock root; external supervisor still required |
| Close | original close/closed and process exit | explicit native shutdown report required; uncertainty does not count as cleanup success |
| Endpoint | optional original infer/actions payload | external endpoint checkpoint/lease cannot be verified locally; no standard upstream HTTP or full parity claim |

The old Short14 matrix replaced model input with fixed training prompts. Some encode a specific target/color. These cannot silently stand for newer randomized tasks. Default current-worker-public prompt corresponds to the actual Discovery public-binding method; choose a historical public prompt only when deliberately reproducing the matching old task.

## CPU evidence

Two source-parity tests and one synthetic transport seam passed without importing Torch:

1. Byte SHA of both original and packaged projection equals6939b1…. The same synthetic50×22 rows cover gripper clipping, mixed base channels, deadband, acceleration and braking. Applied rows, reason events, pending rows and commit0/17/50 state are exactly equal. Invalid nonfinite final rows reject without pending state.
2. AST comparison of `mapping`, `state`, `sample`, `rgb`, `batch`, `actions` and `transform` assignments matches the original source. Canonicalization is limited to class self bindings and the explicit VLM path parameter. This preserves original default-mode normalization and NumPy epsilon arithmetic rather than merely asserting equivalent dimensions.

These CPU checks do not execute Torch transforms, image interpolation, weights or numerical model inference. They complement the short real-model connection check; they do not establish equivalence for a different checkpoint or environment.

The additional transport seam uses fake owner/model objects, not a checkpoint: one50-row chunk with tiny mixed base outputs is projected by the original adapter to zero base commands, sent once, acknowledged at step50 and committed. It checks public task routing, public context preservation and metadata allowlisting; it is not a native simulation or real model result.

Two additional CPU configuration tests use synthetic files and the real pure `native_worker.configuration` / `example.sdk_paths` helpers. They verify generated argv/report placeholders, automatic model-file hashes, fresh save/reload and source-workspace output rejection, without SDK imports, Torch or task sampling.
