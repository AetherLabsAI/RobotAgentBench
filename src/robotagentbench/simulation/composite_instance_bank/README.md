# Composite instance bank

Task-owned CPU preparation and immutable loading for seven non-cloth long tasks and four OpenEnd scenarios. It does not run a policy, start Isaac Sim, change scoring, or change physical/control/camera settings.

| Paper task | Preparation identity | Native identity |
| --- | --- | --- |
| Long 1: Stock the tray (Multi) | `long_task_15` | existing `stock_wooden_tray` multi-target alias |
| Long 2: Tidy the sofa area | `long_task_16` | `long_task_16` |
| Long 3: Tidy the table | `long_task_17` | `long_task_17` |
| Long 5: Find and pack an item | `long_task_19` | `long_task_19` |
| Long 6: Prepare a meal | `long_task_20` | `long_task_20` |
| Long 7: Prepare coffee and bread | `long_task_21` | `long_task_21` |
| Long 8: Pack bags and baskets | `long_task_22` | `long_task_22` |
| Dinner, guests, leaving, tidying | `prepare_dinner`, `host_guests`, `prepare_to_leave`, `tidy_room` | existing `openend` owner |

Long 4 / runtime 18 is the separate cloth owner and is not generated here.

## Preparation

Use a fresh bank directory containing `OWNER.json` with
`{"owner":"RobotAgentBench-composite-instance-bank-v31"}`. Configure the existing room installation inside that directory. The original source view must remain under the bank's root; shared backend/profile directories must not be used as writable preparation outputs.

`source_install.install_preparers(bank_root, source_root)` installs only the exact original preparation dependencies listed in `source_manifest.py`. Existing different files are rejected, never overwritten. The original path allowlists still apply: outputs are created under each original task's `generated_rooms` directory inside this private source view.

Small original initialization data are embedded in Python so a wheel does not depend on undeclared JSON/USD package-data globs. Original Python preparers are in the `original_sources` package. No original score or physical implementation is copied into a new scorer.

The shelf composer additionally requires the external archive described by `resource_contract.CONTRACT`. After safe extraction, call
`source_install.install_generation_resources(bank_root, extracted_resource_root)`.
This verifies all 215 exact input files and preserves their third-party licenses. It copies task-local inputs, not packages into the SDK or global Python environment. The original Linux/CPython 3.11 native dependency ABI is retained. Archive publication/asset-index integration is a separate release step; no network download or installation occurs here.

The configured CPU namespace must expose the original read-only asset/data aliases and the original Node/USD/schema dependencies. In particular, do not replace original full scene-layer data with a partial code-only directory at the same logical path. Additional fridge interior resources must retain their original 22-file manifest and hashes; they are shared with the existing fridge preparation resource package.

With an explicit operator configuration containing `root`, `source_root`, `sdk_site`, `node`, and approved task `templates`:

```sh
python -B -m robotagentbench.simulation.composite_instance_bank.worker \
  --config /path/to/bank/config.json --task long_task_16 --seed 1
```

`templates.configuration(root=..., source_root=..., sdk_site=..., node=...)`
returns this configuration without writing or sampling. Its eleven original
request templates and public v4 `context`/`prompt` inputs are bundled with the
package; a private staging configuration is not needed. Serialize the returned
dictionary as the explicit `config.json` inside the owned bank directory.

Generate requested seeds directly. Do not silently replace a rejected seed, widen a range, change an original gate, or relabel an existing scene as a new sample. The reviewed bank uses actual seeds 1–30, the existing long-task layout-4 templates and OpenEnd layout-43 templates, with original seeded object/station placement. Distinct initial configurations do not imply distinct room layouts.

## Daily situations for OpenEnd

To prepare a **new** context-rich cohort, pass
`context_profile="daily_situations_v1"` to `templates.configuration(...)`.
The worker selects the matching public text **and** scene requirements before
composition. Existing saved configurations and the original profile are not
rewritten. Long tasks are unaffected.

Each scenario has six context/scene variants, each used with five of the original
seeds 1–30. These are not thirty different stories; some variants differ only
in observable scene state and use identical public text.
The unchanged household memory is followed by a separate **Today's situation**
section. The complete context and high-level request are visible to the policy;
target bindings, the operator rationale, and the scene request are not included.

| Scenario | Context / scene variation |
| --- | --- |
| Dinner | Emma's leftovers; Alex's usual meal; Ryan's meal plus coffee; a table spill; a cancelled dinner plan; skipping Ryan's usual coffee tonight |
| Guests | Alex's, Emma's, or Ryan's preferences; a changed visitor; two visitors with different preferences; a spill before the visit |
| Leaving | Rain during a library trip, shopping trip, or park walk; dry weather; an outdated rain forecast; rain only after returning home |
| Tidying | Ordinary tidying; a spill; used cans; discarded paper and a used cup; mixed waste; mixed waste plus a spill |

Stains, used packaging, empty containers, and cleanliness in this table are
**scene variation**, not facts disclosed in `Today's situation`. The agent
must discover them through camera observations. Dinner/guest variants that
differ only in stain presence share the same public text; all six tidying
variants likewise use one neutral request and context. General conditional
household rules remain available, without identifying the current scene state.

Weather is supplied as contextual information, not a new simulated rain effect.
Rain during an outdoor trip selects the existing `umbrella_transfer` task:
the folded umbrella starts on the sofa and is packed into the open bag. Its
existing process maximum is 9, plus the existing 1-point target contact credit.
The current forecast and the trip's time window matter: merely mentioning rain
does not always require an umbrella. No new scoring rule is introduced.

New preparations write `public_input_binding.json`, pinned to the actual
`launch.json` and saved `public_input.json`. Pass it through the existing
`--public-input-binding` model-client option. Do **not** launch a context-rich
case using only the native legacy `prompt_id`: it is retained for admission,
not as a replacement for the separately saved daily context.

The 0.2.1 saved-configuration release corrects only public context from the
0.2.0 cohort, preserving every physical state and task target. Both earlier
archives remain unchanged. Use the 0.2.1 manifest in the [configuration guide](../../../../docs/INSTANCES.md)
to download the saved states instead of generating them again.

## Saved instances and read-only loading

Each successful instance has a real USD scene, original ready/spawn state, request, manifest, native launch, and hash-bound `instance.json`. The native launch remains the output of the original preparer/admitter. Server-only authoring requests and target bindings are not policy inputs.

```sh
python -B -m robotagentbench.simulation.composite_instance_bank.bank \
  /path/to/bank/records/long_task_16_seed001/instance.json
```

`bank.load()` imports no task generator or SDK. It checks saved artifact hashes and reads the same launch; it does not sample again. The original owner still enforces complete source/asset pins and live initial gates at native admission.

OpenEnd instances also save the existing v4 `public_input.json` separately. Pass that explicit public input through the appropriate policy-client binding. The legacy native frozen-context bundle is not rewritten here; generating a room does not establish that the generic model-context routing has been updated. Only the controlled `context` and `prompt` are model-visible, not the scene request or instance metadata.

`actual_scene_cpu_admitted` means real scene composition, original static CPU admission, and saved-input reload passed. It is **not** native settling, manipulation success, a completed scenario, or a model score. Existing disabled/live gate behavior is preserved, including the known original Long-6 fridge initial-gate limitation. No gate is enabled or bypassed by this bank.

## Release boundaries

Global task catalogs, backend profiles, shared assets, scoring, and global packaging/asset indexes are not modified. Task-local source/data installation hooks are provided; any generic CLI/catalog/asset-index integration remains a separate explicit integration step.

Saved launches retain original absolute logical resource names. Loading a saved
instance is immutable, not path migration: a relocated installation still needs
the reviewed read-only resource namespace. The external shelf-helper archive and
the shared 22-file fridge resource package must be made available by the release
asset installer; this module does not publish them or silently download them.

Original project code follows the repository's Apache-2.0 license. Scene/calibration data retain the project data policy. Third-party geometry/native dependencies retain their included original licenses and are not relicensed as Apache code.

Private host operators, dispatch ledgers, server paths and validation receipts belong only under ignored `private/`. Private Python operator files use `.py.private` so the repository's recursive `*.py` source-distribution rule cannot include them inadvertently.
