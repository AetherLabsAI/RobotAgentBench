# Shelf scene dependencies

Saved shelf scenes and room compositions use the original scene preparation
helpers bundled under `backends/profiles/room/vendor/shelf_generation`. Their
source bytes and sampling rules are unchanged. The pinned original file list
is retained in `simulation/composite_instance_bank/resource_contract.py`.

The Python source and accompanying notices ship with the evaluator. Native
Linux CPython 3.11 dependency payloads ship in shared asset release 0.1.6;
the installer checks their hashes and places them in the task's own runtime.
They are not installed into the user's Python environment or Isaac SDK.
The original helper paths are restored within the task's read-only compatibility
tree alongside the room assets; those paths are not required on the host.

The included python-fcl, Rtree, and Shapely distributions retain their original
licenses under their respective `*.dist-info/licenses/` directories. Those
licenses, rather than the benchmark's own asset license, apply to these
third-party dependencies.
