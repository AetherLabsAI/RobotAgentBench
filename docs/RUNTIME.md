# Native control runtime

The control runtime attaches to an already initialized and accepted native task
session. It does not create a second simulator, choose task targets, replace a
task's scoring rules, or execute policy Python on the simulator host.

## Execution boundary

`simulation.worker.run_native_worker` runs in a dedicated subprocess with an
operator-created `NativeWorkspace`, `NativeRunOptions`, and trusted
`TaskBackend`. These objects are not loaded from agent RPCs. There is no
model-selectable Python module, file-system root, score override, or GPU-sharing
permission in the action protocol.

`simulation.original_backend.run_original_worker` routes recovery and atomic
coffee launches before entering the generic original-source binding. A trusted
operator-supplied handoff must verify its specialized source profile and replace
the process with its original interpreter. Missing handoffs, already-loaded
source namespaces, unsupported coding mode and returning handoffs are rejected;
none can fall back to the generic worker. The router creates no output or GPU
lease. It is not a substitute for the child owner's complete admission checks.

`simulation.admission` checks prepared launch identities and original task-owner
admission. It verifies availability and source pins without refreshing manifests
or substituting assets. Canonical input and output roots replace deployment-local
directory allowlists; output directories must be fresh. Launch, request, manifest,
ready pose when supplied, optional spawn receipt, and scene content are pinned
until startup. Generic standalone and room owners require their prepared ready
pose; specialized owners retain their own reset and must not acquire a generic
ready-pose requirement.
Room requests may be compiled by their original owner without rewriting their
frozen input files. Recovery and atomic-coffee tasks retain their separate native
interpreter handoffs; they cannot silently fall through to the generic owner.
Task Discovery alone receives the approved [N + 2 contact budget](BENCHMARK.md#task-discovery-contact-budget)
through its task-owned release binding. It consumes the same native finger
contact stream and retains original coverage scoring, initialization and clock;
the source's physical object catalogue is not inferred from agent commands.

`simulation.gpu` preserves the native occupancy thresholds and requires explicit
per-run approval for either GPU exception mode. It holds a nonblocking cooperative
owner lock and repeats occupancy checks after acquiring that lock. All workers
using a host GPU must share the operator-configured lock directory. The owner
does not terminate or move other processes.

The native scene owner verifies launch inputs and source pins, admits its GPU,
loads the prepared scene, restores the declared robot initial state, initializes
task readers, settles the scene, and accepts the measured initial state before
attaching a control host. The bootstrap helpers preserve the full head camera
at 1280 × 800 and both wrist cameras at 848 × 480.

`simulation.native_execution` preserves the native Isaac initialization and
control body: CPU/60 Hz startup, one robot ready-state restore, 240 settling
substeps at 60 Hz, measured initial acceptance, the original native joint/base
controllers and collision guard, and synchronized render-only observations.
Renderer and physics modules are imported only after `SimulationApp` starts.
Task readers, robot parameter alignment, context-specific factories, and code
tools must be supplied by their original trusted backend; missing bindings are
not replaced with generic implementations.

Each submitted action chunk is validated in full before its first native write.
Rows contain 20 physical-unit joint targets and two base-velocity commands.
Joint limits, gripper coupling, base acceleration limits, and straight/turn mode
constraints are enforced without clipping. Each accepted row advances one
20 Hz control interval using the task-selected physical substeps.

`runtime.action_host` and `runtime.action_driver` retain the raw-action path,
including the environment actor's pre-physics hook. The separate
`runtime.coding_action_host` and `runtime.coding_driver` retain the action
transport used by code tools, including robot-state reads without camera
rendering. These are distinct paths, not interchangeable drivers. The action
transport alone is not a policy sandbox or an inverse-kinematics tool.

## Original task backends

Local source, runtime and asset locations are configured through the read-only
bindings described in [runtime dependencies](DEPENDENCIES.md).

`simulation.original_backend.bind_original_backend` connects the real original
task registry and factories to `TaskBackend`. Ordinary standalone tasks use the
original ShortTask session. Waste selection retains its own native factory;
drink, snack and sofa selection retain the shared selection factory. Room
admission, construction and accepted-session adaptation remain with the original
room owner, including its separate LongTask and Task Discovery branches.

`simulation.source_tree.NativeSourceTree` loads the configured bundled source
profile with immutable SHA-256 pins. Managed runtime modules cannot fall back to an
unrelated installed package or unreviewed bytecode cache. Importing the binding
does not download sources, write source caches, initialize Isaac or allocate a
GPU. This is integrity checking of trusted host code, not a policy sandbox.
Keep one source binding active for the lifetime of one dedicated worker; do not
reuse its task objects after the binding closes or share it across workers.
Operator-reviewed namespace mappings can locate a restored original dependency
without editing its source bytes or changing its module identity. Relocation
does not rewrite embedded paths, refresh dependency manifests or enable a
specialized interpreter; those retain their own admission requirements.

Native controller, robot-alignment and collision-guard factories are loaded only
through their original runtime modules. The original collision-library source
and platform-specific native artifacts require a separate pinned `native`
dependency group. Code tools are optional and require their original broker,
starter resources and `coding` dependency group. Original sandbox ownership,
container-image and isolation checks are not replaced by this binding.

Specialized recovery and atomic-coffee interpreters are not attached through the
generic source environment. The [specialized operator](SPECIALIZED.md) supplies
the trusted fixed-family handoff after verifying its bundled source profile,
task-specific dependencies and prepared launch. Recovery and atomic coffee
retain their different original import precedence and interpreter environments;
missing configuration is rejected rather than falling back to ordinary-task
physics or scoring.

## Progress, results, and termination

The task session or room owns progress scores, timeout policy, final-state
checks, and termination. The transport delegates `begin_control`,
`after_physics`, and `finish_control`; it does not add a new wall-clock timeout
or recompute task scores. `runtime.room_session` adapts an accepted room while
preserving its native terminal outcome. Abandonment and execution errors are
not converted to valid zero-score evaluations.

Observation reads must not advance physics. Only the allowlisted public camera,
robot state, context, and explored-map fields pass through the observation filter. Request UUIDs prevent action
replay; a failed or uncertain native write faults the host rather than being
automatically retried.

## Model-visible observations

The generic native worker returns the same information on its raw-action and
code-tool paths. The initial `ready.contract.observation_fields` lists the
fields available in that episode:

| Field | Content |
| --- | --- |
| `instruction` | Original task/scenario instruction, without evaluator target bindings |
| `context` | Complete original public household memory or selection context; empty for tasks without supplied context |
| `images` | Synchronized base64 PNG images named `head`, `left`, and `right` |
| `joints` | Named, measured native joint state in the original physical units |
| `base_state` | Measured world position, quaternion, unwrapped yaw, linear and angular velocity |
| `navigation` | Explored-map PNG and metric metadata, using the original `openend.native_map_only.v1` packet |
| `episode_id`, `step`, `time_s`, `image_step` | Episode and control-frame identity |

The map is extracted from the admitted scene's static geometry at the original
0.04 m resolution. It is progressively revealed using the measured robot state,
original head-camera geometry, camera field of view and static occlusion. A small
region around the base is initialized as known. Unknown cells remain grey;
known free floor, static obstacles, clearance margins, measured base pose and
the executed trajectory are drawn with the original renderer. This is not RGB
SLAM. It supplies no route, `goto` action, target pose, or task annotation.
Generated `/World/RoomEpisode` objects and the robot are excluded from static
scene extraction, matching the original map-assisted evaluation. Internal full
grids and geometry files remain evaluator-side under `navigation/private_geometry`;
only the explored PNG and its allowlisted metadata are returned to the model.

OpenEnd and Task Discovery context comes from their original versioned public
input readers. Context Reasoning uses only its original public question text;
answer keys, object bindings, manifests and process scores are not observations.
The original instruction remains unchanged, including context already embedded
in that instruction. `context` also exposes the complete supplied text separately.

The lightweight code-tool robot-state read includes context but intentionally
does not render cameras or update the map; request a full observation for images
and the map. A map provider is episode-local, and repeated reads of one control
frame do not advance physics or append duplicate trajectory points. Map PNGs are
recorded alongside the episode. Specialized recovery, atomic-coffee and native
cloth owners keep their distinct original protocols; they do not fall through
to this generic worker or acquire synthetic observations from another scene.

## Shutdown boundary

The native owner's `simulation.shutdown.NativeResources` closes the host and session,
detaching physics and collision monitoring, detaching camera annotators,
destroying render products, checking source layers, closing the simulator, and
releasing its GPU lock. Each cleanup stage is attempted even if another fails;
partial camera initialization is tracked before attachment can fail. Cleanup
errors invalidate the shutdown acknowledgement but do not rewrite task scores.
`runtime.completion.CompletionStream` buffers the close
acknowledgement until the owner confirms successful native shutdown; closing
the action pipe alone is not a cleanup receipt.

The CLI provides catalog queries, read-only dependency checks, and the explicit
[native tablecloth operator](CLOTH.md) and [specialized owners](SPECIALIZED.md).
The cloth operator internally supplies
the trusted binding for its three fixed tasks; its default mode is non-native.
Native execution still needs complete task/backend dependency and asset
configuration and a dedicated worker environment. Specialized operators keep
their original task-specific owners and require an external job supervisor.
The CPU tests use deterministic doubles to validate these control and lifecycle
contracts; they do not establish physical task success or native simulator
compatibility.
