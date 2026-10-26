"""Import-safe prim-path contract for the canonical TableclothBasket.

This module is intentionally stdlib-only because ``playback_video3.py`` reads
the paths before Isaac's SimulationApp makes ``pxr`` importable.
"""

BASKET_PATH = "/World/Assets/TableclothBasket"
BASKET_MODEL_PATH = f"{BASKET_PATH}/model"
BASKET_RISER_PATH = f"{BASKET_PATH}/riser"
ASSET_MESH_RELATIVE_PATH = (
    "Meshes/Sketchfab_model/_dc85028501a4347a20df6a75b3e3044_fbx/"
    "RootNode/pCube2/pCube2_Texture_0/pCube2_Texture_0"
)
BASKET_COLLISION_PATH = (
    f"{BASKET_MODEL_PATH}/Geometry/{ASSET_MESH_RELATIVE_PATH}"
)

__all__ = [
    "ASSET_MESH_RELATIVE_PATH",
    "BASKET_COLLISION_PATH",
    "BASKET_MODEL_PATH",
    "BASKET_PATH",
    "BASKET_RISER_PATH",
]
