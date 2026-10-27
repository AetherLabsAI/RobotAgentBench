# Preparation-only recovered source and data

These are exact original project bytes, not replacement implementations. They
are installed only when absent in a task-owned source view; any existing
different byte identity is an error. Original component-source records and
evaluators are not repinned or replaced.

| Original task-relative source | SHA-256 | Provenance |
| --- | --- | --- |
| fridge_to_heated_box/scene.py | `9571e8b0610e29e19b6e296728d321a8c59c5cf41088bab5b53e7b0911c77f2e` | Existing reviewed_initial_inputs.v1 |
| fridge_to_heated_box/sources.py | `514820a911cfeae7f88db6faf06d7d1aba432125f90e9aa4d7b27fa0640ee6d2` | Existing reviewed_initial_inputs.v1 |
| sofa_to_drawer/prepare_initial.py | `7b91601fc4b6af33b76660b5b3947249f3ffa7aa3388714ddaa47926299b8c9c` | Current original initializer, approved missing-export recovery |
| sofa_to_drawer/initial_station.py | `0c9c465e34d8f1f3ecf444645d5f6d13d8e7fced09682a4eb9b96b717400bdfe` | Direct original dependency of that initializer |
| sofa_to_drawer/prepare_scene.py | `79f075dbfb5fd67390e0f0b95416c8b4e6ab81362be12c98bb986ee612a8e9ec` | Same original preparation source used by the composite-task bank |
| sofa_to_coffee_table/prepare_initial.py | `12218ed5e81e7b2c288dfeb4853e55a4bac7e23e6f33ce2fffe4fa165c6fccba` | Existing generic-current source inventory |

The original living-room wrapper is unchanged (`1b29b947…7ad26e`), and its
original sublayers remain separately required. `cloth_wrapper_resources.py`
holds the 20 original cloth-clear wrapper bytes and exact identities from the
previously reviewed wrapper list; no mass, geometry, or collision opinions were
edited. It restores missing layers, rather than generating new wrappers.

`resource_manifest.py` records the original 22 fridge interior resources and
their original manifest identity (`18b83a03…be433e`), plus the two original sofa
initializer data identities. The fridge resources were missing from the
installed source view, not alternative geometry versions. Each copied original
resource matched its prior recorded hash. The operator's data packages retain
the original manifest unmodified. Asset authorization follows the project asset
policy; the code license is not a new asset or robot license grant.

The separate CPU-composer dependency provider retains its existing reviewed
215-file receipt (`49890683…6f4a8a`), including its original FCL/Rtree/Shapely
dependencies. It is a read-only provider, not a new scene generator or a change
to the original collision/sampling algorithms. Deployment paths, host wrappers,
output hashes, and CPU receipts stay in private operator records.

All saved instances remain **CPU prepared, not native initial accepted**.
Original CPU launch admission and scene validation run in the original
preparer; this is not Isaac settling or a model/trajectory evaluation.
