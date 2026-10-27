# Saved public scenario input

The saved OpenEnd bank contains the selected v4 household memory and scenario
request in each `public_input.json`. The unchanged original room owner still
uses its older frozen context bundle when no explicit binding is supplied.
Generating or selecting a v4 instance alone does not replace that default.

For a saved room instance, the trusted operator can save
`robotagentbench.simulation.public_input.from_selection(selection)` as a new
binding JSON below an explicit read-only input root. Its exact schema is:

```json
{
  "schema": "robotagentbench.public_input_binding.v1",
  "source": "/original/logical/case/public_input.json",
  "sha256": "<exact saved public-file SHA256>",
  "launch_sha256": "<exact unchanged saved-launch SHA256>"
}
```

The execution namespace must map the saved original logical paths and the
binding file read-only; the worker's `NativeWorkspace.input_roots` must include
those paths. This does not rewrite a saved launch, manifest, scene, or source
pin. Cross-machine asset relocation still requires the normal namespace plan.

- Raw example worker: add `--public-input-binding /input/public-binding.json`.
- Coding worker configuration: optionally add
  `"public_input_binding": "/input/public-binding.json"`.
- Trusted Python worker: pass `public_input_binding=Path(...)` to
  `run_original_worker`/`run_native_worker`/`admit_prepared_launch`.

The argument is operator configuration, not an action/RPC option. The public
file may contain exactly `instruction`+`context` or original v4
`prompt`+`context`, all text. Additional keys, private targets/answers, changed
hashes, wrong launches, outside-root paths, and symlinks fail closed. There is
no fallback to request targets, scene manifests, task lists, or evaluator data.

Original room admission, registration, source checks, initial acceptance,
physical execution, score policy, timeout, and cleanup are unchanged. Only the
admitted public instruction and context are replaced; the original room's
existing accepted-session instruction boundary delivers that instruction to
both raw and coding observations. The binding file, public file, and original
launch remain pinned in `AdmittedLaunch.verify_inputs()` before GPU admission
and again before startup. Other owner families reject this optional binding
rather than ignore it. With the argument omitted, the old behavior is unchanged.

CPU tests use synthetic public text and frozen-owner doubles; they are not a
new native or model evaluation. Existing v37 deployed files remain unchanged.
