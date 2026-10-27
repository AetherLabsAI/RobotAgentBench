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
