# Third-party notices

RobotAgentBench's original evaluation code is licensed under the
[Apache License 2.0](../LICENSE). Third-party software, assets, datasets, and
model weights retain their respective licenses and terms. The project's code
license does not replace or extend those terms.

## Simulator and dependencies

Simulator runtimes and separately installed Python or native dependencies are
obtained from their upstream distributions. Their upstream licenses and notices
continue to apply. Installing a dependency does not grant permission to
redistribute its source, binaries, or bundled assets under RobotAgentBench's
license.

The selected native environment includes these distributions. This is not a
complete license inventory of the simulator's transitive components or linked
libraries; retain the notices shipped with the exact installed distributions.

| External resource | License evidence | Distribution in this code package |
| --- | --- | --- |
| Isaac Sim 5.1.0 and Omniverse Kit | [Version-specific licensing](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses-isaac-sim.html): Isaac Sim source and separately licensed Kit, models and textures have distinct terms | Not bundled; install upstream distributions under their applicable terms |
| NumPy 1.26.0 | [BSD-3-Clause](https://numpy.org/doc/1.26/license.html); binary distributions can carry additional notices | Not bundled |
| SciPy 1.15.3 | [BSD-3-Clause and bundled-component notices](https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt) | Not bundled |
| Pillow 11.3.0 | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/11.3.0/LICENSE) | Not bundled |
| python-fcl 0.7.0.11 | [BSD-3-Clause](https://github.com/BerkeleyAutomation/python-fcl/blob/v0.7.0.11/LICENSE); linked native libraries have their own notices | Not bundled |

The project does not relicense or redistribute an installed SDK, its native
material files, or collision-library binaries. Do not infer permission to
redistribute them from the license of a Python wrapper or of this repository.

## Robot, scene, and object assets

Robot models, meshes, textures, room layouts, and task-object resources are
separate from the evaluation source code. Any separately supplied asset package
retains its own attribution, license, and access requirements. The presence of
an asset identifier or loading interface in this repository does not grant
redistribution rights to the referenced asset.

Team-authored rooms, task objects, initialization records, question banks and
standard answers in the [asset release](https://huggingface.co/datasets/DylanHel/RobotAgentBench-assets)
use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This is separate
from the evaluation code's Apache-2.0 license. Embedded third-party materials
and their adaptations keep their original terms.

The AgiBot G1 / GenieSim robot resources retain the upstream
[GenieSimAssets CC BY-NC-SA 4.0 terms](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE),
including their attribution, non-commercial and share-alike requirements.
The asset release includes the original license text, resource attribution and
integration-change notice. These resources are not relicensed under CC BY 4.0
or Apache-2.0. Refer to `LICENSE.md`, `NOTICE.md` and `LICENSES/` in the asset
repository; its license archive is downloaded by the asset installer.

The source distribution includes the repository's rendered documentation
images and short overview video. These previews are not simulator geometry,
textures, episode state, control trajectories, or a redistributable SDK/robot
asset bundle. Their inclusion does not relicense the referenced native assets.

The human-intervention resource has an upstream attribution record:
[fps arms (rigged only)](https://opengameart.org/content/fps-arms-rigged-only),
by para, released under CC0 and crediting the MakeHuman team. This identifies
that upstream resource only, not every derived rig or other human model.
The arm asset is not included in this code package.

## Data, model weights, and services

Third-party datasets and pretrained weights are obtained separately and remain
subject to their original terms. Users provide any required service accounts
and credentials; private credentials are not part of this source distribution.

## Included third-party source

The cloth profile includes Python source from Shapely 2.1.2 under its BSD
3-Clause license. Its copyright and license are retained at
`src/robotagentbench/backends/profiles/cloth/geometry_deps/shapely-2.1.2.dist-info/licenses/LICENSE.txt`.
The separate native environment supplies Shapely's compiled components and GEOS.

Where source files are distributed under another license, their original
copyright, license, and attribution notices must be retained. Notices associated
with those files take precedence over the project's default code license for
the affected material.
