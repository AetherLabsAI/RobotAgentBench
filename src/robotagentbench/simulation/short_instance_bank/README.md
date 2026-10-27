# Short-task saved instance bank

This module prepares and read-only reloads the original short-task instances
1–16. It does not run a model, Isaac, robot actions, or scoring. Short tasks
17–18 use their independent cloth owner and are not synthesized here.

`bank.py` invokes the unchanged original `prepare_task.prepare` and the original
`probe_random_layout.request_for`, starting with seeds 1–30 and its existing
random-layout mode. The saved index contains the request, original launch and
generated-file hashes, actual composed object transforms, semantic bindings,
and a physical-state fingerprint that excludes seed/path/time. A different seed
alone is not counted as a distinct instance. `prepared_cpu_not_native_accepted`
is explicitly not native initial acceptance or task success. Failures have no
process score and are retained; the same rejected seed is never silently
replaced or retried. To fill a bank with 30 valid static initial configurations,
the operator may inspect later original candidate seeds up to 90. Candidate
seeds and final episode ordinals are separate, and all rejections remain in the
ledger. This is initial-geometry admission, not selection based on model scores.

Preparation requires an independently owned Runtime and its original source and
resource namespace. An existing unsettled intent is never redispatched. A new
attempt for the same seed is only for a diagnosed preparation infrastructure
failure, not a way to bypass original rejection. New candidate seeds are
separate, once-only static preparations.

```text
python -m robotagentbench.simulation.short_instance_bank.cpu_entry \
  --sdk-site <existing-py311-USD-site> --prepare \
  --runtime <independently-owned-Runtime> --task <task_name> \
  --attempt <new-attempt> --seed 1

python -m robotagentbench.simulation.short_instance_bank.cpu_entry \
  --sdk-site <existing-py311-USD-site> --load <saved-index.json> \
  --task <task_name> --seed 1 --verify-scene
```

Omit `--seed` only after representative preparations succeed, to prepare seeds
1–30. Preserve the original namespace when reloading; do not rewrite launch
paths. The task names and paper numbering are the ordered `bank.TASKS` tuple.

For a bank with already verified saved instances, `--reuse-index` imports their
unchanged launch hashes. `--seed-start 31 --seed-end 90 --target-count 30`
prepares only later candidates and stops once 30 distinct actual object states
have been saved. The loader never calls this preparation path.

`source_install.py` contains exact original, previously audited missing
preparation source identities. It adds only absent files to the independent
view, refuses conflicts, and never replaces evaluators, physics, component
records, or shared source. The fridge scene/sources identities are anchored in
the existing `reviewed_initial_inputs.v1` record; the small living-room wrapper
is byte-identical to the original scene layer. Preparation helper versions must
match each original component record, rather than repinning that record.

Exact small Python/layer resources are included in this package. The original
fridge data provider (22 members plus its original manifest) and sofa initializer
data (two members) are separately required and described by
`resource_manifest.py`; installation rejects absent or different members.
The bank operator preserves the original manifests alongside those data, and
does not treat a code license as an asset/robot license grant.

Seeded paper-object preparation also reuses the exact data embedded by
`composite_instance_bank.source_manifest`; the installer adds only missing
members with matching original hashes. Existing source records are not repinned.

Deployment-specific host scripts, source-provider recipes, CPU logs and bank
indexes are private operator material. They are not the public Python API and
are not evidence of model/native acceptance. Private scripts deliberately do
not use a `.py` extension so they are excluded by the source-package Python
inclusion rule.
