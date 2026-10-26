# -*- coding: utf-8 -*-
"""Fail-closed runtime policy for retail assets used by shelf reproduce tasks.

Runtime approval is carried by a public, Git-versioned release receipt.  The
receipt contains no candidate images, REAL frames, contact sheets, or private
paths.  Approval is necessary but not sufficient: the live scene must also
point at a reviewed refined bundle.  Old ``photo_assets`` with a matching
product name are therefore not silently accepted.

There is deliberately no environment-variable override for this policy.
Importing a newly refined asset requires updating the checked-in bundle revisions
and its scene reference here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable

try:
    from real2sim.production.git_provenance import source_revision
except ModuleNotFoundError:
    production = Path(__file__).resolve().parents[1] / "production"
    if str(production) not in sys.path:
        sys.path.insert(0, str(production))
    from git_provenance import source_revision


ARTIFACT_ID = "63d876d3-1d89-4911-bd1d-181f2f63ec30"
ARTIFACT_URL = f"https://claude.ai/code/artifact/{ARTIFACT_ID}"
DOCUMENT_SNAPSHOT = "2026-07-24"
RELEASE_RECEIPT_SCHEMA = "real2sim.shelf_asset_release_receipt.v1"
RELEASE_RECEIPT_PATH = (
    Path(__file__).resolve().parent / "shelf_asset_release_receipt.json"
)

SHELF = "/World/Assets/ShelfItems"
TIER2 = "/World/Assets/Tier2Items"
from machine_paths import DEMO_ROOT as DEFAULT_DEMO_ROOT  # noqa: E402

READY = "ready"
NEEDS_QUALITY_UPGRADE = "needs_quality_upgrade"
NEEDS_REFINED_ASSET_IMPORT = "needs_refined_asset_import"
NEEDS_TASK_INTEGRATION = "needs_task_integration"
NOT_IN_QC_DOCUMENT = "not_in_qc_document"
NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class DocumentAsset:
    asset_id: str
    name: str
    category: str
    document_status: str
    expected_dimensions_m: tuple[float, ...] | None = None
    quality_note: str = ""


def _doc(
    asset_id: str,
    name: str,
    category: str,
    document_status: str,
    dimensions: tuple[float, ...] | None = None,
    note: str = "",
) -> DocumentAsset:
    return DocumentAsset(
        asset_id=asset_id,
        name=name,
        category=category,
        document_status=document_status,
        expected_dimensions_m=dimensions,
        quality_note=note,
    )


# Exactly the twenty cards visible in the current artifact.  The header chips
# claim one additional bottle and one additional box; those stale counts are
# not permission to infer unlisted assets.
DOCUMENT_ASSETS = {
    # cans
    "sprite": _doc(
        "sprite", "Sprite", "can", READY, (0.066, 0.122), "清晰/无缝"
    ),
    "coke_usa": _doc(
        "coke_usa", "Coca-Cola (USA)", "can",
        NEEDS_QUALITY_UPGRADE, (0.066, 0.122), "略重影"
    ),
    "grapefruit_sparkling": _doc(
        "grapefruit_sparkling", "Grapefruit Sparkling", "can",
        NEEDS_QUALITY_UPGRADE, (0.066, 0.122), "略软"
    ),
    "starbucks_espresso": _doc(
        "starbucks_espresso", "Starbucks Espresso", "can",
        NEEDS_QUALITY_UPGRADE, (0.054, 0.103), "反光略糊"
    ),
    "red_bull_pink_slim": _doc(
        "red_bull_pink_slim", "Red Bull Pink (slim)", "can",
        READY, (0.053, 0.168), "尺寸对/可读"
    ),
    "red_bull_slim": _doc(
        "red_bull_slim", "Red Bull (slim)", "can",
        NEEDS_QUALITY_UPGRADE, None, "反光重影"
    ),
    # tubes and bottle
    "pringles_bbq": _doc(
        "pringles_bbq", "Pringles BBQ (USA)", "tube",
        READY, (0.075, 0.230), "清晰"
    ),
    "pringles_sour_cream": _doc(
        "pringles_sour_cream", "Pringles Sour Cream (USA)", "tube",
        READY, (0.075, 0.230), "清晰"
    ),
    "gatorade_orange": _doc(
        "gatorade_orange", "Gatorade Orange", "bottle",
        NEEDS_QUALITY_UPGRADE, (0.070, 0.210), "底模好；标签略糊"
    ),
    # boxes
    "coke_mini_fridge_pack": _doc(
        "coke_mini_fridge_pack", "Coca-Cola mini fridge-pack", "box",
        READY, (0.285, 0.115, 0.105), "好"
    ),
    "sprite_mini_fridge_pack": _doc(
        "sprite_mini_fridge_pack", "Sprite mini fridge-pack", "box",
        READY, (0.285, 0.115, 0.105), "好"
    ),
    "honey_maid_handi_snacks": _doc(
        "honey_maid_handi_snacks", "Honey Maid Handi-Snacks (10 PACKS)",
        "box", READY, (0.235, 0.125, 0.045), "重拍/正面清晰"
    ),
    "unreal_smores_kit": _doc(
        "unreal_smores_kit", "UNREAL Peanut-Butter S'mores Kit", "box",
        READY, (0.255, 0.180, 0.065), "重拍/正背清晰"
    ),
    "red_bull_box": _doc(
        "red_bull_box", "Red Bull box", "box",
        READY, (0.110, 0.110, 0.175), "好"
    ),
    "clif_zbar_protein": _doc(
        "clif_zbar_protein", "CLIF Z-Bar Protein", "box",
        READY, (0.135, 0.135, 0.040), "重拍/已修复"
    ),
    # bags
    "viva_cacao_powder": _doc(
        "viva_cacao_powder", "Viva Cacao Powder", "bag",
        READY, (0.160, 0.230, 0.060), "好"
    ),
    "lindt": _doc(
        "lindt", "Lindt", "bag", NEEDS_QUALITY_UPGRADE,
        (0.150, 0.220, 0.070), "顶部略开"
    ),
    "skittles_share": _doc(
        "skittles_share", "Skittles (share)", "bag",
        NEEDS_QUALITY_UPGRADE, (0.200, 0.260, 0.060),
        "仅单张照片；顶部略开"
    ),
    "skittles_littles": _doc(
        "skittles_littles", "Skittles Littles", "bag",
        READY, (0.150, 0.200, 0.050), "重拍/抠图干净"
    ),
    "chomps_beef_pouch": _doc(
        "chomps_beef_pouch", "CHOMPS beef pouch", "bag",
        NEEDS_QUALITY_UPGRADE, (0.120, 0.210, 0.045), "重拍；袋型偏窄"
    ),
}


@dataclass(frozen=True)
class RefinedBundle:
    asset_id: str
    revision: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class ReviewedLocalAsset:
    asset_id: str
    name: str
    category: str
    revision: str


# Products that are visible in audited historical REAL primary-camera frames
# but absent from the external twenty-card document use a separate fail-closed
# registry.  This does not add inferred cards to DOCUMENT_ASSETS.
REVIEWED_LOCAL_ASSETS = {
    "gatorade_zero_blue": ReviewedLocalAsset(
        asset_id="gatorade_zero_blue",
        name="Gatorade Zero Glacier Freeze (blue, orange cap)",
        category="bottle",
        revision=(
            "gatorade-zero-blue-refined-v3-imagegen-translucent-solid-cap-"
            "2026-07-30"
        ),
    ),
    "diet_coke_fifa_can": ReviewedLocalAsset(
        asset_id="diet_coke_fifa_can",
        name="Diet Coke silver/red football-edition can",
        category="can",
        revision="diet-coke-fifa-refined-v4-imagegen-sealed-2026-07-30",
    ),
    "good2grow_strawberry_kiwi": ReviewedLocalAsset(
        asset_id="good2grow_strawberry_kiwi",
        name=(
            "Good2Grow Strawberry Kiwi kids juice "
            "(black Spider-Man-style character topper)"
        ),
        category="bottle",
        revision="good2grow-strawberry-kiwi-refined-v4-2026-07-29",
    ),
    "oreo_gf": ReviewedLocalAsset(
        asset_id="oreo_gf",
        name="Gluten Free Double Stuf OREO pillow pack",
        category="bag",
        revision="oreo-gf-refined-v1-realframe-2026-08-02",
    ),
    "cayman_jacked_margarita": ReviewedLocalAsset(
        asset_id="cayman_jacked_margarita",
        name="Cayman Jacked Margarita Extra Strength 23.5 fl oz can",
        category="can",
        revision="cayman-jacked-margarita-refined-v3-sealed-2026-07-30",
    ),
}


# These are the only local bundles that can currently be proven to be the
# refined versions used by the shelf tasks. List every render dependency;
# missing files fail closed and Git owns repository source identity.
REFINED_BUNDLES = {
    "coke_mini_fridge_pack": RefinedBundle(
        asset_id="coke_mini_fridge_pack",
        revision="coke-mini-fridge-pack-retained-v1-2026-07-30",
        files=(
            "photo_assets/coke_mini_box/asset.usda",
            "photo_assets/textures/coke_mini_box_face_front.png",
            "photo_assets/textures/coke_mini_box_face_back.png",
            "photo_assets/textures/coke_mini_box_face_end.png",
        ),
    ),
    "sprite_mini_fridge_pack": RefinedBundle(
        asset_id="sprite_mini_fridge_pack",
        revision="sprite-mini-fridge-pack-retained-v1-2026-07-30",
        files=(
            "photo_assets/sprite_mini_box/asset.usda",
            "photo_assets/textures/sprite_mini_box_face_front.png",
            "photo_assets/textures/sprite_mini_box_face_back.png",
            "photo_assets/textures/sprite_mini_box_face_end.png",
        ),
    ),
    "unreal_smores_kit": RefinedBundle(
        asset_id="unreal_smores_kit",
        revision="unreal-smores-kit-retained-v1-2026-07-30",
        files=(
            "photo_assets/smores_kit_box/asset.usda",
            "photo_assets/textures/smores_kit_box_face_front.png",
            "photo_assets/textures/smores_kit_box_face_end.png",
        ),
    ),
    "gatorade_zero_blue": RefinedBundle(
        asset_id="gatorade_zero_blue",
        revision=(
            "gatorade-zero-blue-refined-v3-imagegen-translucent-solid-cap-"
            "2026-07-30"
        ),
        files=(
            "newcans/bottles/gatorade_zero_blue/asset.usda",
            "newcans/bottles/gatorade_zero_blue/"
                "gatorade_bottle_geo.usdc",
            "newcans/bottles/gatorade_zero_blue/textures/"
                "gatorade_zero_glacier_freeze_label_imagegen_v1.png",
        ),
    ),
    "diet_coke_fifa_can": RefinedBundle(
        asset_id="diet_coke_fifa_can",
        revision="diet-coke-fifa-refined-v4-imagegen-sealed-2026-07-30",
        files=(
            "newcans/cans/diet_coke_fifa/asset.usda",
            "newcans/cans/diet_coke_fifa/soda_can_detail_geo.usdc",
            "newcans/cans/diet_coke_fifa/textures/"
                "diet_coke_fifa_wrap.png",
        ),
    ),
    "good2grow_strawberry_kiwi": RefinedBundle(
        asset_id="good2grow_strawberry_kiwi",
        revision="good2grow-strawberry-kiwi-refined-v4-2026-07-29",
        files=(
            "newcans/bottles/good2grow_strawberry_kiwi/asset.usda",
            "newcans/bottles/good2grow_strawberry_kiwi/textures/"
                "good2grow_strawberry_kiwi_wrap_v1.png",
        ),
    ),
    "oreo_gf": RefinedBundle(
        asset_id="oreo_gf",
        revision="oreo-gf-refined-v1-realframe-2026-08-02",
        files=(
            "newcans/bags/oreo_gf_pack/asset.usda",
            "newcans/bags/oreo_gf_pack/textures/front.png",
            "newcans/bags/oreo_gf_pack/textures/back.png",
        ),
    ),
    "cayman_jacked_margarita": RefinedBundle(
        asset_id="cayman_jacked_margarita",
        revision="cayman-jacked-margarita-refined-v3-sealed-2026-07-30",
        files=(
            "newcans/cans/cayman_jacked_margarita/asset.usda",
            "newcans/cans/cayman_jacked_margarita/"
                "soda_can_detail_geo.usdc",
            "newcans/cans/cayman_jacked_margarita/textures/basecolor.png",
        ),
    ),
    "coke_usa": RefinedBundle(
        asset_id="coke_usa",
        revision="coke-usa-refined-v5-imagegen-sealed-2026-07-30",
        files=(
            "newcans/cans/coke_usa_fifa/asset.usda",
            "newcans/cans/coke_usa_fifa/soda_can_detail_geo.usdc",
            "newcans/cans/coke_usa_fifa/textures/basecolor.png",
        ),
    ),
    "starbucks_espresso": RefinedBundle(
        asset_id="starbucks_espresso",
        revision="starbucks-espresso-refined-v4-sealed-2026-07-30",
        files=(
            "newcans/cans/starbucks_espresso_light/asset.usda",
            "newcans/cans/starbucks_espresso_light/"
                "soda_can_detail_geo.usdc",
            "newcans/cans/starbucks_espresso_light/textures/basecolor.png",
        ),
    ),
    "grapefruit_sparkling": RefinedBundle(
        asset_id="grapefruit_sparkling",
        revision="grapefruit-sparkling-refined-v4-sealed-2026-07-30",
        files=(
            "newcans/cans/grapefruit_sparkling/asset.usda",
            "newcans/cans/grapefruit_sparkling/"
                "soda_can_detail_geo.usdc",
            "newcans/cans/grapefruit_sparkling/textures/basecolor.png",
        ),
    ),
    "red_bull_pink_slim": RefinedBundle(
        asset_id="red_bull_pink_slim",
        revision="red-bull-pink-refined-v4-sealed-2026-07-30",
        files=(
            "newcans/cans/red_bull_pink_slim/asset.usda",
            "newcans/cans/red_bull_pink_slim/soda_can_detail_geo.usdc",
            "newcans/cans/red_bull_pink_slim/textures/basecolor.png",
        ),
    ),
    "red_bull_slim": RefinedBundle(
        asset_id="red_bull_slim",
        revision="red-bull-silver-refined-v4-sealed-2026-07-30",
        files=(
            "newcans/cans/red_bull_silver_slim/asset.usda",
            "newcans/cans/red_bull_silver_slim/"
                "soda_can_detail_geo.usdc",
            "newcans/cans/red_bull_silver_slim/textures/basecolor.png",
        ),
    ),
    "honey_maid_handi_snacks": RefinedBundle(
        asset_id="honey_maid_handi_snacks",
        revision="honey-maid-refined-v2-2026-07-28",
        files=(
            "newcans/boxes/honey_maid_handi_snacks/asset.usda",
            "newcans/boxes/honey_maid_handi_snacks/textures/front.png",
            "newcans/boxes/honey_maid_handi_snacks/textures/back.png",
            "newcans/boxes/honey_maid_handi_snacks/textures/left.png",
            "newcans/boxes/honey_maid_handi_snacks/textures/right.png",
            "newcans/boxes/honey_maid_handi_snacks/textures/top.png",
            "newcans/boxes/honey_maid_handi_snacks/textures/bottom.png",
        ),
    ),
    "skittles_share": RefinedBundle(
        asset_id="skittles_share",
        revision="skittles-share-refined-v2-2026-07-28",
        files=(
            "newcans/bags/skittles_share/asset.usda",
            "newcans/bags/skittles_share/textures/front.png",
            "newcans/bags/skittles_share/textures/back.png",
        ),
    ),
    "lindt": RefinedBundle(
        asset_id="lindt",
        revision="lindt-lindor-milk-refined-v2-2026-07-28",
        files=(
            "newcans/bags/lindt_lindor_milk/asset.usda",
            "newcans/bags/lindt_lindor_milk/textures/front.png",
            "newcans/bags/lindt_lindor_milk/textures/back.png",
        ),
    ),
    "skittles_littles": RefinedBundle(
        asset_id="skittles_littles",
        revision="skittles-littles-refined-v3-2026-07-29",
        files=(
            "newcans/bags/skittles_littles/asset.usda",
            "newcans/bags/skittles_littles/textures/front.png",
            "newcans/bags/skittles_littles/textures/back.png",
        ),
    ),
    "chomps_beef_pouch": RefinedBundle(
        asset_id="chomps_beef_pouch",
        revision="chomps-beef-pouch-refined-v3-2026-07-29",
        files=(
            "newcans/bags/chomps_beef_pouch/asset.usda",
            "newcans/bags/chomps_beef_pouch/textures/front.png",
            "newcans/bags/chomps_beef_pouch/textures/back.png",
        ),
    ),
    "clif_zbar_protein": RefinedBundle(
        asset_id="clif_zbar_protein",
        revision="clif-zbar-protein-refined-v1-2026-07-27",
        files=(
            "newcans/boxes/clif_zbar_protein/asset.usda",
            "newcans/boxes/clif_zbar_protein/textures/"
                "zbar_box_face_front_refined.png",
            "newcans/boxes/clif_zbar_protein/textures/"
                "zbar_box_face_back_refined.png",
            "newcans/boxes/clif_zbar_protein/textures/"
                "zbar_box_face_end.png",
        ),
    ),
    "gatorade_orange": RefinedBundle(
        asset_id="gatorade_orange",
        revision=(
            "gatorade-orange-refined-v5-imagegen-translucent-solid-cap-"
            "2026-07-30"
        ),
        files=(
            "newcans/bottles/gatorade_orange/asset.usda",
            "newcans/bottles/gatorade_orange/gatorade_bottle_geo.usdc",
            "newcans/bottles/gatorade_orange/textures/"
                "gatorade_zero_orange_label_imagegen_v1.png",
        ),
    ),
    "sprite": RefinedBundle(
        asset_id="sprite",
        revision="sprite-refined-v6-imagegen-large-mark-sealed-2026-07-30",
        files=(
            "newcans/soda_can_355ml/variants/sprite/sprite.usda",
            "newcans/soda_can_355ml/variants/sprite/"
                "soda_can_detail_geo.usdc",
            "newcans/soda_can_355ml/variants/sprite/textures/basecolor.png",
        ),
    ),
    "pringles_bbq": RefinedBundle(
        asset_id="pringles_bbq",
        revision="pringles-bbq-newcans-2026-07-24",
        files=(
            "newcans/tubes/pringles_bbq/asset.usda",
            "newcans/tubes/pringles_bbq/textures/label.png",
        ),
    ),
    # The QC artifact explicitly retains these existing shelf-task models:
    # dimensions match, required front/back/end views are present, and their
    # current paths are declared here so an unrelated legacy asset cannot be
    # substituted under the same prim name.
    "pringles_sour_cream": RefinedBundle(
        asset_id="pringles_sour_cream",
        revision="pringles-sour-cream-existing-qc-2026-07-24",
        files=(
            "photo_assets/pringles_green/asset.usda",
            "photo_assets/textures/pringles_green_wrap.png",
        ),
    ),
    "red_bull_box": RefinedBundle(
        asset_id="red_bull_box",
        revision="red-bull-box-existing-qc-2026-07-24",
        files=(
            "photo_assets/redbull_pink_box/asset.usda",
            "photo_assets/textures/redbull_pink_box_face_front.png",
            "photo_assets/textures/redbull_pink_box_face_back.png",
            "photo_assets/textures/redbull_pink_box_face_end.png",
        ),
    ),
    "viva_cacao_powder": RefinedBundle(
        asset_id="viva_cacao_powder",
        revision="viva-cacao-existing-qc-2026-07-24",
        files=(
            "photo_assets/cacao_bag/asset.usda",
            "photo_assets/textures/cacao_bag_bagf.png",
            "photo_assets/textures/cacao_bag_bagb.png",
        ),
    ),
}


@dataclass(frozen=True)
class TaskAsset:
    task_id: str
    asset_id: str | None
    root: str
    prims: tuple[str, ...]
    integration_status: str
    integration_note: str = ""
    expected_reference: str | None = None


def _task(
    task_id: str,
    asset_id: str | None,
    root: str,
    prims: Iterable[str],
    status: str,
    note: str = "",
    reference: str | None = None,
) -> TaskAsset:
    return TaskAsset(
        task_id=str(task_id),
        asset_id=asset_id,
        root=root,
        prims=tuple(prims),
        integration_status=status,
        integration_note=note,
        expected_reference=reference,
    )


SPRITE_REFERENCE = (
    "./newcans/soda_can_355ml/variants/sprite/sprite.usda"
)
COKE_USA_REFERENCE = "./newcans/cans/coke_usa_fifa/asset.usda"
STARBUCKS_ESPRESSO_REFERENCE = (
    "./newcans/cans/starbucks_espresso_light/asset.usda"
)
GRAPEFRUIT_SPARKLING_REFERENCE = (
    "./newcans/cans/grapefruit_sparkling/asset.usda"
)
RED_BULL_PINK_REFERENCE = (
    "./newcans/cans/red_bull_pink_slim/asset.usda"
)
HONEY_MAID_REFERENCE = (
    "./newcans/boxes/honey_maid_handi_snacks/asset.usda"
)
LINDT_REFERENCE = "./newcans/bags/lindt_lindor_milk/asset.usda"
SKITTLES_SHARE_REFERENCE = "./newcans/bags/skittles_share/asset.usda"
PRINGLES_BBQ_REFERENCE = "./newcans/tubes/pringles_bbq/asset.usda"
GATORADE_ORANGE_REFERENCE = (
    "./newcans/bottles/gatorade_orange/asset.usda"
)
CHOMPS_BEEF_POUCH_REFERENCE = (
    "./newcans/bags/chomps_beef_pouch/asset.usda"
)
OREO_GF_REFERENCE = (
    "./newcans/bags/oreo_gf_pack/asset.usda"
)
CLIF_ZBAR_PROTEIN_REFERENCE = (
    "./newcans/boxes/clif_zbar_protein/asset.usda"
)
SKITTLES_LITTLES_REFERENCE = (
    "./newcans/bags/skittles_littles/asset.usda"
)
GATORADE_ZERO_BLUE_REFERENCE = (
    "./newcans/bottles/gatorade_zero_blue/asset.usda"
)
DIET_COKE_FIFA_REFERENCE = (
    "./newcans/cans/diet_coke_fifa/asset.usda"
)
GOOD2GROW_STRAWBERRY_KIWI_REFERENCE = (
    "./newcans/bottles/good2grow_strawberry_kiwi/asset.usda"
)
CAYMAN_JACKED_MARGARITA_REFERENCE = (
    "./newcans/cans/cayman_jacked_margarita/asset.usda"
)

# Only retail tasks belong here.  An absent task is non-retail and remains
# runnable.  Retaining blocked tasks in the canonical registry preserves the
# historical taxonomy while this table controls execution.
TASK_ASSETS = {
    "6": _task(
        "6", "gatorade_orange", SHELF, ("gatorade_m",), READY,
        reference=GATORADE_ORANGE_REFERENCE,
    ),
    "8": _task(
        "8", "coke_usa", SHELF, ("coke_c3",), READY,
        reference=COKE_USA_REFERENCE,
    ),
    "9": _task(
        "9", "gatorade_zero_blue", SHELF, ("gatorade_t",),
        NEEDS_TASK_INTEGRATION,
        "the asset identity is reviewed, but this older pick recording is not "
        "part of the current fixed rerender scope",
        reference=GATORADE_ZERO_BLUE_REFERENCE,
    ),
    "10": _task("10", "sprite", SHELF, ("sprite_c1", "sprite_c2"), READY,
                reference=SPRITE_REFERENCE),
    "14": _task(
        "14", "gatorade_orange", SHELF, ("gatorade_m",), READY,
        reference=GATORADE_ORANGE_REFERENCE,
    ),
    "15": _task(
        "15", "gatorade_zero_blue", SHELF, ("gatorade_t",),
        NEEDS_TASK_INTEGRATION,
        "the asset identity is reviewed, but the old small-table exemplar "
        "still conflicts with the current CoffeeTable destination contract",
        reference=GATORADE_ZERO_BLUE_REFERENCE,
    ),
    "16": _task(
        "16", "gatorade_orange", SHELF, ("gatorade_m",), READY,
        reference=GATORADE_ORANGE_REFERENCE,
    ),
    "17": _task(
        "17", "gatorade_zero_blue", SHELF, ("gatorade_t",), READY,
        reference=GATORADE_ZERO_BLUE_REFERENCE,
    ),
    "18": _task("18", "sprite", SHELF, ("sprite_c1",), READY,
                reference=SPRITE_REFERENCE),
    "19": _task(
        "19", "diet_coke_fifa_can", SHELF, ("diet_coke",), READY,
        reference=DIET_COKE_FIFA_REFERENCE,
    ),
    # Older/new-batch IDs are not all present in task_registry yet, but are
    # still enumerated so legacy runners cannot treat them as non-retail.
    "21": _task("21", "sprite", SHELF, ("sprite_c1", "sprite_c2"), READY,
                reference=SPRITE_REFERENCE),
    "22": _task(
        "22", "gatorade_zero_blue", SHELF, ("gatorade_t",), READY,
        reference=GATORADE_ZERO_BLUE_REFERENCE,
    ),
    "23": _task(
        "23", "gatorade_orange", SHELF, ("gatorade_m",), READY,
        reference=GATORADE_ORANGE_REFERENCE,
    ),
    "24": _task(
        "24", "diet_coke_fifa_can", SHELF, ("diet_coke",),
        NEEDS_TASK_INTEGRATION,
        "the Diet Coke asset is reviewed, but this newer recording still "
        "needs a verified initializer/calibration",
        reference=DIET_COKE_FIFA_REFERENCE,
    ),
    "25": _task(
        "25", "sprite", SHELF, ("sprite_c1", "sprite_c2"),
        NEEDS_TASK_INTEGRATION,
        "refined Sprite exists, but this newer place recording has no "
        "verified initializer/calibration",
    ),
    "26": _task(
        "26", "gatorade_orange", SHELF, ("gatorade_m",), READY,
        reference=GATORADE_ORANGE_REFERENCE,
    ),
    "35": _task(
        "35", "chomps_beef_pouch", TIER2, ("chomps_r",), READY,
        reference=CHOMPS_BEEF_POUCH_REFERENCE,
    ),
    "36": _task("36", "pringles_bbq", TIER2, ("pring_bbq",), READY,
                reference=PRINGLES_BBQ_REFERENCE),
    "37": _task(
        "37", "honey_maid_handi_snacks", TIER2, ("honeymaid",), READY,
        reference=HONEY_MAID_REFERENCE,
    ),
    "38": _task(
        "38", "skittles_littles", TIER2, ("skit_littles",), READY,
        reference=SKITTLES_LITTLES_REFERENCE,
    ),
    "39": _task(
        "39", "clif_zbar_protein", TIER2, ("zbar_t",), READY,
        reference=CLIF_ZBAR_PROTEIN_REFERENCE,
    ),
    "40": _task("40", "oreo_gf", TIER2, ("oreo_gf",), READY,
                "reviewed-local identity from the small Task 40 REAL head camera; "
                "the prim was added to Tier2Items on 2026-08-02",
                reference=OREO_GF_REFERENCE),
    # Raw head frames are authoritative over the stale NEW_TASKS.md labels:
    # 06ded902 clearly reads "10 PACKS / Honey Maid / S'MORES /
    # HANDI-SNACKS"; 036f057e clearly shows the CLIF Z-Bar Protein box.
    "41": _task(
        "41", "honey_maid_handi_snacks", TIER2, ("honeymaid",), READY,
        reference=HONEY_MAID_REFERENCE,
    ),
    "42": _task("42", "pringles_bbq", TIER2, ("pring_bbq",), READY,
                reference=PRINGLES_BBQ_REFERENCE),
    "43": _task(
        "43", "chomps_beef_pouch", TIER2, ("chomps_r",), READY,
        reference=CHOMPS_BEEF_POUCH_REFERENCE,
    ),
    "44": _task(
        "44", "skittles_littles", TIER2, ("skit_littles",), READY,
        reference=SKITTLES_LITTLES_REFERENCE,
    ),
    "45": _task("45", "oreo_gf", TIER2, ("oreo_gf",), READY,
                "same reviewed-local oreo_gf identity as small Task 40",
                reference=OREO_GF_REFERENCE),
    "46": _task(
        "46", "clif_zbar_protein", TIER2, ("zbar_t",), READY,
        reference=CLIF_ZBAR_PROTEIN_REFERENCE,
    ),
    "47": _task(
        "47", "good2grow_strawberry_kiwi", SHELF, ("good2grow",), READY,
        reference=GOOD2GROW_STRAWBERRY_KIWI_REFERENCE,
    ),
    "48": _task(
        "48", "starbucks_espresso", SHELF, ("starbucks",), READY,
        "the refined short-can asset is scene-bound and the snack initializer "
        "derives a per-episode grasp anchor from the exact trajectory",
        reference=STARBUCKS_ESPRESSO_REFERENCE,
    ),
    "50": _task(
        "50", "grapefruit_sparkling", SHELF, ("grapefruit_can",), READY,
        "the refined can is scene-bound and calibrated by the exact episode "
        "trajectory through the shared snack initializer",
        reference=GRAPEFRUIT_SPARKLING_REFERENCE,
    ),
    "51": _task(
        "51", "red_bull_pink_slim", SHELF, ("red_bull_pink",), READY,
        "the canonical shelf alias uses the same reviewed Pink Edition "
        "asset as the reviewed table target and is initialized per episode",
        reference=RED_BULL_PINK_REFERENCE,
    ),
    "85": _task(
        "85", "cayman_jacked_margarita", SHELF, ("cayman_jacked",), READY,
        "the reviewed alcohol-can identity used by small Task 88 place is "
        "reused for the corresponding shelf pick without substituting a "
        "different beverage",
        reference=CAYMAN_JACKED_MARGARITA_REFERENCE,
    ),
    "86": _task(
        "86", "good2grow_strawberry_kiwi", SHELF, ("good2grow",), READY,
        reference=GOOD2GROW_STRAWBERRY_KIWI_REFERENCE,
    ),
    "87": _task(
        "87", "grapefruit_sparkling", SHELF, ("grapefruit_can",), READY,
        reference=GRAPEFRUIT_SPARKLING_REFERENCE,
    ),
    "88": _task(
        "88", "cayman_jacked_margarita", SHELF, ("cayman_jacked",), READY,
        reference=CAYMAN_JACKED_MARGARITA_REFERENCE,
    ),
    "89": _task(
        "89", "starbucks_espresso", SHELF, ("starbucks",), READY,
        reference=STARBUCKS_ESPRESSO_REFERENCE,
    ),
}


# Audit mapping for every retail prim currently authored in scene_room.usda.
# Unknown future children are also deactivated by ``usd_scene_filter_source``.
SCENE_PRIM_ASSETS = {
    SHELF: {
        "lindt": "lindt",
        "gatorade_t": "gatorade_zero_blue",
        "cacao": "viva_cacao_powder",
        "honeymaid_t": "honey_maid_handi_snacks",
        "skittles_sh": "skittles_share",
        "pring_g": "pringles_sour_cream",
        "pring_p": "pringles_bbq",
        "pring_r": None,
        "pring_cup": None,
        "diet_coke": "diet_coke_fifa_can",
        "coke_usa": "coke_usa",
        "starbucks": "starbucks_espresso",
        "red_bull_pink": "red_bull_pink_slim",
        "good2grow": "good2grow_strawberry_kiwi",
        "grapefruit_can": "grapefruit_sparkling",
        "cayman_jacked": "cayman_jacked_margarita",
        "zbar2": "clif_zbar_protein",
        "littles2": "skittles_littles",
        "sprite_c1": "sprite",
        "coke_c3": "coke_usa",
        "teal_box2": None,
        "rb_box": "red_bull_box",
        "sprite_c2": "sprite",
        "sprite_box": "sprite_mini_fridge_pack",
        "gatorade_m": "gatorade_orange",
        "gatorade_b": None,
    },
    TIER2: {
        "chomps_r": "chomps_beef_pouch",
        "smores_k": "unreal_smores_kit",
        "pring_smoky": None,
        "pring_bbq": "pringles_bbq",
        "honeymaid": "honey_maid_handi_snacks",
        "skit_littles": "skittles_littles",
        "lays": None,
        "zbar_t": "clif_zbar_protein",
        "chomps_t": None,
        "pring_cup2": None,
        "oreo_gf": None,
    },
}

READY_SCENE_PRIMS = {
    SHELF: frozenset((
        "cacao",
        "coke_c3",
        "diet_coke",
        "gatorade_m",
        "gatorade_t",
        "good2grow",
        "grapefruit_can",
        "cayman_jacked",
        "starbucks",
        "red_bull_pink",
        "pring_g",
        "rb_box",
        "sprite_c1",
        "sprite_c2",
        "sprite_box",
    )),
    TIER2: frozenset((
        "chomps_r", "honeymaid", "oreo_gf", "pring_bbq", "skit_littles",
        "smores_k", "zbar_t",
    )),
}

READY_PRIM_BINDINGS = {
    SHELF: {
        "cacao": (
            "viva_cacao_powder",
            "./photo_assets/cacao_bag/asset.usda",
        ),
        "coke_c3": ("coke_usa", COKE_USA_REFERENCE),
        "diet_coke": (
            "diet_coke_fifa_can",
            DIET_COKE_FIFA_REFERENCE,
        ),
        "gatorade_m": (
            "gatorade_orange",
            GATORADE_ORANGE_REFERENCE,
        ),
        "gatorade_t": (
            "gatorade_zero_blue",
            GATORADE_ZERO_BLUE_REFERENCE,
        ),
        "good2grow": (
            "good2grow_strawberry_kiwi",
            GOOD2GROW_STRAWBERRY_KIWI_REFERENCE,
        ),
        "grapefruit_can": (
            "grapefruit_sparkling",
            GRAPEFRUIT_SPARKLING_REFERENCE,
        ),
        "cayman_jacked": (
            "cayman_jacked_margarita",
            CAYMAN_JACKED_MARGARITA_REFERENCE,
        ),
        "starbucks": (
            "starbucks_espresso",
            STARBUCKS_ESPRESSO_REFERENCE,
        ),
        "red_bull_pink": (
            "red_bull_pink_slim",
            RED_BULL_PINK_REFERENCE,
        ),
        "pring_g": (
            "pringles_sour_cream",
            "./photo_assets/pringles_green/asset.usda",
        ),
        "rb_box": (
            "red_bull_box",
            "./photo_assets/redbull_pink_box/asset.usda",
        ),
        "sprite_c1": ("sprite", SPRITE_REFERENCE),
        "sprite_c2": ("sprite", SPRITE_REFERENCE),
        "sprite_box": (
            "sprite_mini_fridge_pack",
            "./photo_assets/sprite_mini_box/asset.usda",
        ),
    },
    TIER2: {
        "chomps_r": (
            "chomps_beef_pouch",
            CHOMPS_BEEF_POUCH_REFERENCE,
        ),
        "honeymaid": (
            "honey_maid_handi_snacks",
            HONEY_MAID_REFERENCE,
        ),
        "oreo_gf": ("oreo_gf", OREO_GF_REFERENCE),
        "pring_bbq": ("pringles_bbq", PRINGLES_BBQ_REFERENCE),
        "skit_littles": (
            "skittles_littles",
            SKITTLES_LITTLES_REFERENCE,
        ),
        "smores_k": (
            "unreal_smores_kit",
            "./photo_assets/smores_kit_box/asset.usda",
        ),
        "zbar_t": (
            "clif_zbar_protein",
            CLIF_ZBAR_PROTEIN_REFERENCE,
        ),
    },
}


@dataclass(frozen=True)
class AssetDecision:
    task_id: str
    applicable: bool
    usable: bool
    status: str
    asset_id: str | None
    document_name: str | None
    root: str | None
    prims: tuple[str, ...]
    reason: str
    asset_revision: str | None = None
    artifact_url: str = ARTIFACT_URL

    def to_dict(self) -> dict:
        return asdict(self)


def load_release_approval_receipt() -> tuple[dict, dict]:
    """Load the public approval receipt and record its Git source revision."""
    path = RELEASE_RECEIPT_PATH
    if not path.is_file():
        raise RuntimeError(f"shelf release receipt is missing: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid shelf release receipt: {exc}") from exc
    expected = {
        "schema": RELEASE_RECEIPT_SCHEMA,
        "status": "approved_for_runtime",
        "document_snapshot": DOCUMENT_SNAPSHOT,
        "private_evidence_distributed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"shelf release receipt has invalid {key!r}")
    assets = receipt.get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError("shelf release receipt has no asset map")
    active_assets = {
        asset_id
        for bindings in READY_PRIM_BINDINGS.values()
        for asset_id, _reference in bindings.values()
    }
    if set(assets) != active_assets:
        missing = sorted(active_assets - set(assets))
        extra = sorted(set(assets) - active_assets)
        raise RuntimeError(
            "shelf release receipt asset set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    try:
        revision = source_revision((path, Path(__file__).resolve()))
    except (RuntimeError, ValueError):
        revision = {
            "schema": "real2sim.external_source_file.v1",
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
        }
    return receipt, revision


def verify_release_approval(
    asset_id: str,
) -> tuple[bool, str, str | None]:
    """Verify public runtime approval without opening private QC evidence."""
    try:
        receipt, _receipt_revision = load_release_approval_receipt()
    except RuntimeError as exc:
        return False, str(exc), None
    entry = receipt["assets"].get(asset_id)
    if not isinstance(entry, dict):
        return False, f"no release approval for {asset_id}", None
    required = {
        "name",
        "category",
        "approval_basis",
        "approval_revision",
        "runtime_bundle_revision",
    }
    if set(entry) != required:
        return False, f"release approval fields are invalid for {asset_id}", None
    if entry["approval_basis"] not in {
        "document_snapshot",
        "reviewed_upgrade",
        "reviewed_local_asset",
    }:
        return False, f"release approval basis is invalid for {asset_id}", None
    if not all(
        isinstance(entry[key], str) and entry[key]
        for key in required
    ):
        return False, f"release approval values are invalid for {asset_id}", None
    bundle = REFINED_BUNDLES.get(asset_id)
    if bundle is None or entry["runtime_bundle_revision"] != bundle.revision:
        return False, f"release approval bundle revision mismatch for {asset_id}", None
    document = DOCUMENT_ASSETS.get(asset_id)
    if document is not None and (
        entry["name"] != document.name or entry["category"] != document.category
    ):
        return False, f"release approval identity mismatch for {asset_id}", None
    return (
        True,
        "public release approval receipt verified",
        entry["approval_revision"],
    )


def asset_has_approval(asset_id: str) -> tuple[bool, str, str | None]:
    """Accept only assets present in the Git-versioned release receipt."""
    return verify_release_approval(asset_id)


def verify_refined_bundle(
    asset_id: str,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> tuple[bool, str, str | None]:
    """Verify every declared USD/texture path and return its reviewed revision."""
    bundle = REFINED_BUNDLES.get(asset_id)
    if bundle is None:
        return False, f"no reviewed refined bundle for {asset_id}", None
    root = Path(demo_root)
    ok, reason, approval_revision = verify_release_approval(asset_id)
    if not ok or approval_revision is None:
        return False, reason, None
    for relative in bundle.files:
        path = root / relative
        if not path.is_file():
            return False, f"refined asset dependency is missing: {path}", None
    return True, "reviewed refined bundle paths verified", bundle.revision


def _prim_metadata(scene_text: str, prim: str) -> str | None:
    pattern = (
        rf'\b(?:def|over)\s+Xform\s+"{re.escape(prim)}"\s*'
        rf"\((.*?)\)\s*\{{"
    )
    match = re.search(pattern, scene_text, flags=re.DOTALL)
    return match.group(1) if match else None


def verify_scene_references(
    task: TaskAsset,
    scene_path: Path | str,
) -> tuple[bool, str]:
    """Bind a ready task to the intended refined reference in the live scene."""
    if not task.expected_reference:
        return False, f"task {task.task_id} has no reviewed scene reference"
    path = Path(scene_path)
    try:
        scene_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return False, f"cannot inspect live scene {path}: {exc}"
    expected = f"@{task.expected_reference}@"
    for prim in task.prims:
        metadata = _prim_metadata(scene_text, prim)
        if metadata is None:
            return False, f"live scene is missing retail prim {task.root}/{prim}"
        if expected not in metadata:
            return (
                False,
                f"{task.root}/{prim} does not reference the reviewed asset "
                f"{task.expected_reference}",
            )
    return True, "live scene references the reviewed refined asset"


def verify_ready_scene_assets(
    scene_path: Path | str,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> tuple[bool, str, dict[str, str]]:
    """Verify every shelf item that the scene filter intends to leave active."""
    path = Path(scene_path)
    try:
        scene_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return False, f"cannot inspect live scene {path}: {exc}", {}

    revisions: dict[str, str] = {}
    for root, bindings in READY_PRIM_BINDINGS.items():
        for prim, (asset_id, expected_reference) in bindings.items():
            if asset_id not in revisions:
                ok, reason, revision = verify_refined_bundle(
                    asset_id, demo_root=demo_root
                )
                if not ok or revision is None:
                    return False, reason, {}
                revisions[asset_id] = revision
            metadata = _prim_metadata(scene_text, prim)
            if metadata is None:
                return (
                    False,
                    f"live scene is missing approved retail prim {root}/{prim}",
                    {},
                )
            if f"@{expected_reference}@" not in metadata:
                return (
                    False,
                    f"{root}/{prim} does not reference approved document "
                    f"asset {expected_reference}",
                    {},
                )
    return True, "all active retail scene asset paths are verified", revisions


def require_ready_scene_assets(
    scene_path: Path | str,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> dict[str, str]:
    ok, reason, revisions = verify_ready_scene_assets(
        scene_path, demo_root=demo_root
    )
    if not ok:
        raise RuntimeError(f"shelf scene whitelist failed: {reason}")
    return revisions


def task_asset_decision(
    task_id: str,
    *,
    verify_runtime: bool = False,
    scene_path: Path | str | None = None,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> AssetDecision:
    """Return a fail-closed task decision without mutating the scene."""
    tid = str(task_id)
    task = TASK_ASSETS.get(tid)
    if task is None:
        return AssetDecision(
            task_id=tid,
            applicable=False,
            usable=True,
            status=NOT_APPLICABLE,
            asset_id=None,
            document_name=None,
            root=None,
            prims=(),
            reason="task does not use a shelf retail asset",
        )

    document = DOCUMENT_ASSETS.get(task.asset_id) if task.asset_id else None
    reviewed_local = (
        REVIEWED_LOCAL_ASSETS.get(task.asset_id) if task.asset_id else None
    )
    document_name = (
        document.name
        if document
        else reviewed_local.name if reviewed_local else None
    )
    status = task.integration_status
    reason = task.integration_note
    if document is None and reviewed_local is None:
        status = NOT_IN_QC_DOCUMENT
        reason = reason or "asset has no visible card in the QC document"
    elif reviewed_local is not None:
        approved, approval_reason, _approval_revision = asset_has_approval(
            task.asset_id
        )
        if not approved:
            status = NEEDS_QUALITY_UPGRADE
            reason = approval_reason
        elif status == READY:
            reason = approval_reason
        else:
            reason = (
                reason
                or f"reviewed local {reviewed_local.name} bundle is not integrated"
            )
    elif document.document_status != READY:
        approved, approval_reason, _approval_revision = asset_has_approval(
            task.asset_id
        )
        if not approved:
            status = NEEDS_QUALITY_UPGRADE
            reason = (
                f"{document.name} is marked for quality improvement and its "
                f"local remediation is not verified: {approval_reason}"
            )
        else:
            reason = (
                approval_reason
                if status == READY
                else reason or f"refined {document.name} bundle is not integrated"
            )
    elif status != READY:
        reason = reason or f"refined {document.name} bundle is not integrated"

    revision = None
    if status == READY and verify_runtime:
        ok, bundle_reason, revision = verify_refined_bundle(
            task.asset_id, demo_root=demo_root
        )
        if not ok:
            status = NEEDS_REFINED_ASSET_IMPORT
            reason = bundle_reason
        else:
            effective_scene = (
                Path(scene_path)
                if scene_path is not None
                else Path(demo_root) / "scene_room.usda"
            )
            refs_ok, refs_reason = verify_scene_references(task, effective_scene)
            if not refs_ok:
                status = NEEDS_REFINED_ASSET_IMPORT
                reason = refs_reason
                revision = None
            else:
                reason = f"{bundle_reason}; {refs_reason}"
    elif status == READY:
        bundle = REFINED_BUNDLES.get(task.asset_id)
        revision = bundle.revision if bundle else None
        approved, approval_reason, _approval_revision = asset_has_approval(
            task.asset_id
        )
        if not approved:
            status = NEEDS_QUALITY_UPGRADE
            revision = None
            reason = approval_reason
        else:
            reason = f"{approval_reason}; refined bundle paths are declared"

    return AssetDecision(
        task_id=tid,
        applicable=True,
        usable=status == READY,
        status=status,
        asset_id=task.asset_id,
        document_name=document_name,
        root=task.root,
        prims=task.prims,
        reason=reason,
        asset_revision=revision,
    )


def require_task_asset_ready(
    task_id: str,
    *,
    verify_runtime: bool = False,
    scene_path: Path | str | None = None,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> AssetDecision:
    """Raise before any subprocess/scene mutation when a retail asset is blocked."""
    decision = task_asset_decision(
        task_id,
        verify_runtime=verify_runtime,
        scene_path=scene_path,
        demo_root=demo_root,
    )
    if decision.applicable and not decision.usable:
        raise RuntimeError(
            f"shelf asset blocked ({decision.status}): {decision.reason}"
        )
    return decision


def task_is_runnable(task_id: str) -> bool:
    """Static scheduler filter; direct entrypoints perform runtime verification."""
    decision = task_asset_decision(task_id)
    return not decision.applicable or decision.usable


def blocked_retail_tasks(task_ids: Iterable[str]) -> dict[str, dict]:
    output = {}
    for task_id in task_ids:
        decision = task_asset_decision(str(task_id))
        if decision.applicable and not decision.usable:
            output[str(task_id)] = decision.to_dict()
    return output


def usable_prims(root: str) -> tuple[str, ...]:
    return tuple(sorted(READY_SCENE_PRIMS.get(root, ())))


def usable_task_prims(root: str) -> tuple[str, ...]:
    """Return prims that are approved as targets, not merely static scenery."""
    return tuple(sorted({
        prim
        for task in TASK_ASSETS.values()
        if task.root == root and task_is_runnable(task.task_id)
        for prim in task.prims
    }))


def is_retail_prim_path(path: str) -> bool:
    return any(
        path == root or path.startswith(root + "/")
        for root in (SHELF, TIER2)
    )


def require_nonretail_prim_path(path: str, operation: str) -> None:
    """Keep task-unaware generic USD writers away from all retail models."""
    if is_retail_prim_path(path):
        raise RuntimeError(
            f"{operation} is task-unaware and cannot mutate retail prim {path}; "
            "use init_task.py with a canonical task id"
        )


def require_ready_task_prim(
    root: str,
    prim: str,
    *,
    scene_path: Path | str,
    demo_root: Path | str = DEFAULT_DEMO_ROOT,
) -> tuple[str, str]:
    """Verify a legacy prim-name entrypoint against the target allow-list."""
    if prim not in usable_task_prims(root):
        mapped = SCENE_PRIM_ASSETS.get(root, {}).get(prim)
        if mapped is None:
            reason = "model is not present in the QC document"
        else:
            document = DOCUMENT_ASSETS.get(mapped)
            reviewed_local = REVIEWED_LOCAL_ASSETS.get(mapped)
            reason = (
                f"{document.name} is not an integrated ready task asset"
                if document is not None
                else (
                    f"{reviewed_local.name} is not an integrated ready task asset"
                    if reviewed_local is not None
                    else "model has no visible QC-document card"
                )
            )
        raise RuntimeError(
            f"shelf asset blocked for {root}/{prim}: {reason}"
        )
    binding = READY_PRIM_BINDINGS.get(root, {}).get(prim)
    if binding is None:
        raise RuntimeError(
            f"approved task prim has no reviewed binding: {root}/{prim}"
        )
    asset_id, reference = binding
    ok, reason, revision = verify_refined_bundle(
        asset_id, demo_root=demo_root
    )
    if not ok or revision is None:
        raise RuntimeError(reason)
    task = _task(
        "legacy-prim-entrypoint",
        asset_id,
        root,
        (prim,),
        READY,
        reference=reference,
    )
    ok, reason = verify_scene_references(task, scene_path)
    if not ok:
        raise RuntimeError(reason)
    require_ready_scene_assets(scene_path, demo_root=demo_root)
    return asset_id, revision


def usd_scene_filter_source(stage_name: str = "s") -> str:
    """Generate a USD snippet that hides every non-ready direct shelf child.

    Iterating ``GetAllChildren`` makes unknown future shelf items fail closed as
    well.  Ready paths are then explicitly reactivated and asserted.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stage_name):
        raise ValueError("stage_name must be a Python identifier")
    specs = {
        root: tuple(sorted(names))
        for root, names in READY_SCENE_PRIMS.items()
    }
    return f"""
_real2sim_ready_retail = {specs!r}
for _real2sim_root_path, _real2sim_ready_names in _real2sim_ready_retail.items():
    _real2sim_root = {stage_name}.GetPrimAtPath(_real2sim_root_path)
    assert _real2sim_root and _real2sim_root.IsValid(), (
        'missing retail root ' + _real2sim_root_path)
    _real2sim_ready_names = set(_real2sim_ready_names)
    for _real2sim_child in list(_real2sim_root.GetAllChildren()):
        _real2sim_child.SetActive(
            _real2sim_child.GetName() in _real2sim_ready_names)
    for _real2sim_name in sorted(_real2sim_ready_names):
        _real2sim_prim = {stage_name}.GetPrimAtPath(
            _real2sim_root_path + '/' + _real2sim_name)
        assert _real2sim_prim and _real2sim_prim.IsValid(), (
            'missing approved refined retail prim '
            + _real2sim_root_path + '/' + _real2sim_name)
        if not _real2sim_prim.IsActive():
            _real2sim_prim.SetActive(True)
"""


def policy_snapshot() -> dict:
    """Strict JSON-ready manifest for reports and resume signatures."""
    return {
        "schema": "real2sim.shelf_asset_policy.v2",
        "artifact_id": ARTIFACT_ID,
        "artifact_url": ARTIFACT_URL,
        "document_snapshot": DOCUMENT_SNAPSHOT,
        "visible_card_count": len(DOCUMENT_ASSETS),
        "document_assets": {
            key: asdict(value) for key, value in DOCUMENT_ASSETS.items()
        },
        "release_receipt": load_release_approval_receipt()[0],
        "source_revision": load_release_approval_receipt()[1],
        "ready_scene_prims": {
            root: sorted(names) for root, names in READY_SCENE_PRIMS.items()
        },
        "ready_prim_bindings": READY_PRIM_BINDINGS,
        "task_assets": {
            key: asdict(value) for key, value in TASK_ASSETS.items()
        },
        "refined_bundles": {
            key: asdict(value) for key, value in REFINED_BUNDLES.items()
        },
    }
