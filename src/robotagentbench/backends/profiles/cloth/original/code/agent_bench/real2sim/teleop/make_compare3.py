"""Three-view REAL vs SIM comparison video for one reproduce episode.

Top row = the real recording's three primary cameras (head_color, hand_left, hand_right),
bottom row = the sim playback rendered from the matching onboard camera prims
(playback_video3.py). Both sides are sampled on the SAME 20 fps grid extract_episode.py
uses, so column t is the same instant in real and sim.

Orchestrates the full per-episode flow (same stages as reproduce_batch_v6):
  1. extract_episode (playback_traj.npz)            [this env]
  2. refine_episode  (scene placement for this uid) [subprocess]
  3. playback_video3.py in the Isaac container      [docker exec]
  4. decode the real cams on the grid + compose     [this env, PyAV]

    conda run -n base python real2sim/teleop/make_compare3.py <uuid> [--skip-render]

Output: dataset/sim_episodes_repro/<uid8>_compare3.mp4
"""
import argparse, base64, fcntl, filecmp, json, math, os, re, shlex, shutil, subprocess, sys, tempfile, uuid
from pathlib import Path

import cv2
import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
from machine_paths import (                                    # noqa: E402
    CONTAINER_NAME as _MACHINE_CONTAINER,
    DEMO_ROOT as _MACHINE_DEMO_ROOT,
    RECORD_ROOT as _MACHINE_RECORD_ROOT,
    RENDER_GPU as _MACHINE_RENDER_GPU,
    MESH311_PYTHON,
)
DEMO = str(_MACHINE_DEMO_ROOT)
REC = str(_MACHINE_RECORD_ROOT)
sys.path.insert(0, os.path.join(REPO, "dataset"))
from convert_episode import convert_episode                      # noqa: E402
from decode_images import camera_frames, resample_to_grid, write_mp4  # noqa: E402
from managed_place_lock import (                                # noqa: E402
    PLACE_LOCK,
    SCENE_TRANSACTION_LOCK as _SCENE_TRANSACTION_LOCK,
    acquire_or_inherit_lock,
    lock_pass_fds,
)
from managed_subprocess import run_argv                          # noqa: E402
from place_sidecar import validate_place_sidecar_pair            # noqa: E402
from init_place import (                                        # noqa: E402
    coffee_table_calibration_from_environment,
    destination_for_task,
    zone_for_task,
)
from bag_place_contract import (                                # noqa: E402
    BAG_PLACE_MAX_APPROACH_OFFSET_M,
    BAG_PLACE_SCENE_FILENAME,
    BAG_PLACE_TASK_ID_SET,
)
from task_registry import (                                    # noqa: E402
    REGISTRY,
    contact_prims,
    track_prim,
)
from episode_appearance_randomization import (                 # noqa: E402
    randomization_enabled as appearance_randomization_enabled,
    run_appearance_subprocess,
)
sys.path.insert(0, os.path.join(REPO, "real2sim", "production"))
from git_provenance import source_revision                         # noqa: E402
import house_layout_canary_hook                                  # noqa: E402
from kitchen_tableware_randomization import (                  # noqa: E402
    run_tableware_subprocess,
)
from episode_scene_randomization import (                      # noqa: E402
    context_object_descriptors,
    infer_grasp_hand_from_hdf5,
    randomization_enabled,
    randomization_identity,
    reconcile_hand,
    run_randomization_subprocess,
)
from pick_provenance import (                                  # noqa: E402
    TASK53_CONTACT_SCENE_CALIBRATION_FILENAME,
    task53_contact_scene_calibration_identity,
    task53_episode_calibration,
)
from shelf_asset_policy import (                                # noqa: E402
    READY,
    require_ready_scene_assets,
    require_task_asset_ready,
)
from robot_positions import (                                   # noqa: E402
    COFFEE_TABLE_CURRENT_BOTTOM_M,
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    COFFEE_TABLE_CURRENT_SIZE_XYZ_M,
    COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
    robot_position,
    translate_xy,
)
from check_success import (                                    # noqa: E402
    CURRENT_LAYOUT_SUCCESS_TASKS,
    current_layout_success_contract,
    current_layout_trace_errors,
    distractor_audit_errors,
)
from task29_full_reproduce import (                              # noqa: E402
    CUP_ROOT_PATHS as TASK29_FULL_CUP_ROOT_PATHS,
    EPISODE_UID as TASK29_FULL_EPISODE_UID,
    FINAL_POSES as TASK29_FULL_INITIAL_POSES,
    SOURCE_CUP_PATH as TASK29_FULL_SOURCE_CUP_PATH,
    TASK29_BIN_OFFSET_X_M,
    TASK29_BIN_OFFSET_Y_M,
    TASK29_COFFEE_TABLE_CENTER_XY_M,
    TASK29_COFFEE_TABLE_OFFSET_XY_M,
    TASK29_RETIRED_COFFEE_TABLE_VISUAL_PATH,
    TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
    TASK29_GRIP_CLOSE_MAX_FORCE,
    TASK29_GRIP_OPEN_LIMIT_RAD,
    TASK29_GRIP_PHASES,
    TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD,
    TASK29_PAPER_PLATE_BOTTOM_M,
    TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M,
    TASK29_PAPER_PLATE_PATH,
    TASK29_PAPER_PLATE_TOP_M,
    TASK29_PAPER_PLATE_TRANSLATE_XYZ_M,
    TASK29_RIM_COMBINE_MODE,
    TASK29_RIM_DYNAMIC_FRICTION,
    TASK29_RIM_GRASP_WORLD_OFFSET_XYZ_M,
    TASK29_RIM_GRIP_CLOSE_MAX_FORCE,
    TASK29_RIM_GRIP_CLOSE_RAMP_FRAMES,
    TASK29_RIM_STATIC_FRICTION,
    TASK29_SETTLED_CUP_TRANSITIONS,
    TASK29_TARGET_COMBINE_MODE,
    TASK29_TARGET_DYNAMIC_FRICTION,
    TASK29_TARGET_STATIC_FRICTION,
)

CELL_W, CELL_H, BAND = 432, 270, 26
REAL_CAMS = ("head_color", "hand_left", "hand_right")
SIM_DIRS = ("head", "left", "right")
FPS = 20.0
# Shelf-origin snack/tier2 trajectories are physically the same operation as
# the generic pick family: a dynamic target must stop both distal pads at its
# visible surface.  Keep the instrumentation decision centralized so no shelf
# render can silently fall back to lift-only evidence.
PICK_CONTACT_FAMILIES = frozenset({"pick", "snack", "tier2"})
SHELF_SURFACE_STOP_TASK_IDS = frozenset({
    "6", "8", "9", "10", "21", "22",
    "35", "36", "37", "38", "39", "40",
})
RUNTIME_SOURCE = Path(REPO) / "real2sim" / "teleop" / "playback_video3.py"
RUNTIME_DEST = Path(DEMO) / "usdz" / "playback_video3.py"
PD_RUNTIME_SOURCE = Path(REPO) / "real2sim" / "teleop" / "playback_pd.py"
PD_RUNTIME_DEST = Path(DEMO) / "usdz" / "playback_pd.py"
ROBOT_COLLISION_PROFILE_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "robot_collision_profile.py"
)
ROBOT_COLLISION_PROFILE_DEST = (
    Path(DEMO) / "usdz" / "robot_collision_profile.py"
)
GEOMETRY_SOURCE = Path(REPO) / "real2sim" / "teleop" / "place_collision_geometry.py"
GEOMETRY_DEST = Path(DEMO) / "usdz" / "place_collision_geometry.py"
ALIGN_SOURCE = Path(REPO) / "real2sim" / "teleop" / "align_grasp.py"
ALIGN_DEST = Path(DEMO) / "usdz" / "align_grasp.py"
PAD_GEOMETRY_SOURCE = Path(REPO) / "real2sim" / "teleop" / "pad_geometry.py"
PAD_GEOMETRY_DEST = Path(DEMO) / "usdz" / "pad_geometry.py"
PICK_PROVENANCE_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "pick_provenance.py"
)
PICK_PROVENANCE_DEST = Path(DEMO) / "usdz" / "pick_provenance.py"
TASK53_CALIBRATION_SOURCE = (
    Path(REPO)
    / "real2sim" / "teleop"
    / TASK53_CONTACT_SCENE_CALIBRATION_FILENAME
)
TASK53_CALIBRATION_DEST = (
    Path(DEMO) / "usdz" / TASK53_CONTACT_SCENE_CALIBRATION_FILENAME
)
CAMERA_CALIBRATION_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "camera_pose_calibration.py"
)
CAMERA_CALIBRATION_DEST = (
    Path(DEMO) / "usdz" / "camera_pose_calibration.py"
)
ROBOT_POSITIONS_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "robot_positions.py"
)
ROBOT_POSITIONS_DEST = Path(DEMO) / "usdz" / "robot_positions.py"
GROUP_SHIFT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "group_shift.py"
)
GROUP_SHIFT_DEST = Path(DEMO) / "usdz" / "group_shift.py"
TASK30_CONTRACT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "task30_throw_reproduce.py"
)
TASK30_CONTRACT_DEST = (
    Path(DEMO) / "usdz" / "task30_throw_reproduce.py"
)
TASK69_CONTRACT_SOURCE = (
    Path(REPO)
    / "real2sim"
    / "teleop"
    / "task69_clear_tablecloth_reproduce.py"
)
TASK69_CONTRACT_DEST = (
    Path(DEMO) / "usdz" / "task69_clear_tablecloth_reproduce.py"
)
BAG_PLACE_CONTRACT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "bag_place_contract.py"
)
BAG_PLACE_CONTRACT_DEST = Path(DEMO) / "usdz" / "bag_place_contract.py"
TASK6_DRAWER_CONTRACT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop"
    / "task6_drawer_sequence_contract.py"
)
TASK6_DRAWER_CONTRACT_DEST = (
    Path(DEMO) / "usdz" / "task6_drawer_sequence_contract.py"
)
TASK6_HAT_CALIBRATION_SOURCE = (
    Path(REPO) / "real2sim" / "teleop"
    / "calibrate_task6_hat_30cm_action.py"
)
TASK6_HAT_CALIBRATION_DEST = (
    Path(DEMO) / "usdz" / "calibrate_task6_hat_30cm_action.py"
)
TASK29_CONTRACT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "task29_full_reproduce.py"
)
TASK29_CONTRACT_DEST = (
    Path(DEMO) / "usdz" / "task29_full_reproduce.py"
)
DUAL_BIN_LAYOUT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "dual_trash_bin_layout.py"
)
DUAL_BIN_LAYOUT_DEST = (
    Path(DEMO) / "usdz" / "dual_trash_bin_layout.py"
)
TABLECLOTH_BASKET_CONTRACT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "tablecloth_basket_contract.py"
)
TABLECLOTH_BASKET_CONTRACT_DEST = (
    Path(DEMO) / "usdz" / "tablecloth_basket_contract.py"
)
TASK30_PAPER_ASSET_SOURCE_DIR = (
    Path(REPO)
    / "real2sim"
    / "assets"
    / "genie"
    / "iros_crumpled_paper_000"
)
TASK30_PAPER_ASSET_DEST_DIR = (
    Path(DEMO) / "usdz" / "iros_crumpled_paper_000"
)
TASK30_PAPER_ASSET_RUNTIME_FILES = tuple(
    (
        TASK30_PAPER_ASSET_SOURCE_DIR / relative,
        TASK30_PAPER_ASSET_DEST_DIR / relative,
        f"task30_paper/{relative}",
    )
    for relative in (
        "Aligned.usd",
        "Aligned.usda",
        "textures/Paper.jpg",
        "textures/Paper_Met.jpg",
        "textures/Paper_Nor.jpg",
    )
)
SHELF_POLICY_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "shelf_asset_policy.py"
)
SHELF_POLICY_DEST = Path(DEMO) / "usdz" / "shelf_asset_policy.py"
GIT_PROVENANCE_SOURCE = (
    Path(REPO) / "real2sim" / "production" / "git_provenance.py"
)
GIT_PROVENANCE_DEST = Path(DEMO) / "usdz" / "git_provenance.py"
SHELF_ENFORCER_SOURCE = (
    Path(REPO) / "real2sim" / "teleop" / "enforce_shelf_asset_policy.py"
)
SHELF_ENFORCER_DEST = (
    Path(DEMO) / "usdz" / "enforce_shelf_asset_policy.py"
)
SHELF_RELEASE_RECEIPT_SOURCE = (
    Path(REPO) / "real2sim" / "teleop"
    / "shelf_asset_release_receipt.json"
)
SHELF_RELEASE_RECEIPT_DEST = (
    Path(DEMO) / "usdz" / "shelf_asset_release_receipt.json"
)


SCENE_PATH = Path(DEMO) / "scene_room.usda"
KITCHEN_TABLEWARE_ENABLE_ENV = "REAL2SIM_KITCHEN_TABLEWARE_RANDOMIZATION"
KITCHEN_TABLEWARE_ASSET_ROOT = (
    Path(DEMO) / "kitchen_random" / "assets" / "tableware"
)
BAG_PLACE_SCENE_PATH = Path(DEMO) / BAG_PLACE_SCENE_FILENAME
TASK73_SCENE_PATH = BAG_PLACE_SCENE_PATH
SCENE_TRANSACTION_LOCK = Path(_SCENE_TRANSACTION_LOCK)
MANIFEST_PATH = Path(REPO) / "dataset" / "manifest_new.jsonl"
LEGACY_MANIFEST_PATH = Path(REPO) / "dataset" / "manifest.jsonl"
FRAME_POLICY_MANIFEST = (
    Path(DEMO) / "cam_preview" / "video3" / "asset_policy.json"
)
EPISODE_INPUT_SCHEMA = "real2sim.episode_input.v1"
RECORDLESS_INPUT_SCHEMA = "real2sim.recordless_trajectory_input.v1"
FRAME_PROVENANCE_SCHEMA = "real2sim.video3.provenance.v3"
OUTPUT_PROVENANCE_SCHEMA = "real2sim.compare.provenance.v2"
SIM_ONLY_PROVENANCE_SCHEMA = "real2sim.sim_only_video.v1"
TRAY_TEXTURE_SOURCE = Path(REPO) / "real2sim" / "teleop" / "assets" / "tray_coffee_baseColor.jpg"
TRAY_TEXTURE_DEST = Path(DEMO) / "usdz" / "tray_coffee_baseColor.jpg"
CUSHION_TEXTURE_SOURCE = (
    Path(REPO)
    / "real2sim" / "teleop" / "assets"
    / "cushion_blue_gray_stripes.png"
)
CUSHION_TEXTURE_DEST = (
    Path(DEMO) / "usdz" / "cushion_blue_gray_stripes.png"
)
TASK53_CUSHION_TEXTURE_SOURCE = (
    Path(REPO)
    / "real2sim" / "teleop" / "assets"
    / "task53_cushion_herringbone.png"
)
TASK53_CUSHION_TEXTURE_DEST = (
    Path(DEMO) / "usdz" / "task53_cushion_herringbone.png"
)
TASK55_WALLET_TEXTURE_SOURCE = (
    Path(REPO)
    / "real2sim" / "teleop" / "assets"
    / "task55_wallet_herringbone_lowfreq_v1.png"
)
TASK55_WALLET_TEXTURE_DEST = (
    Path(DEMO)
    / "usdz" / "wallet_tex"
    / "task55_wallet_herringbone_lowfreq_v1.png"
)
TRAY_TEXTURE_TASK_IDS = frozenset({"42", "43", "44", "46"})
LEGACY_CUSHION_TEXTURE_TASK_IDS = frozenset({"11", "12", "13"})


def task_texture_runtime_files(task_id=None):
    """Return only the external textures that the selected task can consume.

    ``None`` retains the historical all-runtime inventory for callers that do
    not select a task.  A real replay always supplies its task id, so unrelated
    owner assets neither block that replay nor become part of its source
    identity.
    """
    task = None if task_id is None else str(task_id)
    files = []
    if task is None or task in TRAY_TEXTURE_TASK_IDS:
        files.append((
            TRAY_TEXTURE_SOURCE,
            TRAY_TEXTURE_DEST,
            "tray_coffee_baseColor.jpg",
        ))
    if task is None or task in LEGACY_CUSHION_TEXTURE_TASK_IDS:
        files.append((
            CUSHION_TEXTURE_SOURCE,
            CUSHION_TEXTURE_DEST,
            "cushion_blue_gray_stripes.png",
        ))
    if task is None or task == "53":
        files.append((
            TASK53_CUSHION_TEXTURE_SOURCE,
            TASK53_CUSHION_TEXTURE_DEST,
            "task53_cushion_herringbone.png",
        ))
    if task is None or task == "55":
        files.append((
            TASK55_WALLET_TEXTURE_SOURCE,
            TASK55_WALLET_TEXTURE_DEST,
            "task55_wallet_herringbone_lowfreq_v1.png",
        ))
    return tuple(files)
SCENE_RUNTIME_LAYERS = (
    (
        Path(DEMO).parent
        / "assets_living_room" / "trash_bin" / "Aligned_bag_red.usd",
        "assets_living_room/trash_bin/Aligned_bag_red.usd",
    ),
    (
        Path(DEMO).parent
        / "assets_living_room" / "trash_bin" / "trash_cans_raw.usd",
        "assets_living_room/trash_bin/trash_cans_raw.usd",
    ),
)
BAG_PLACE_RUNTIME_LAYERS = (
    (
        Path(DEMO)
        / "house_task_assets/task73/tote_bag/usd/tote_bag_manual_skin_v21_zup.usd",
        "bag_place/tote_bag_manual_skin_v21_zup.usd",
    ),
    (
        Path(DEMO)
        / "house_task_assets/task73/tote_bag/usd/tote_bag_manual_skin_v21.usd",
        "bag_place/tote_bag_manual_skin_v21.usd",
    ),
    (
        Path(DEMO)
        / "house_task_assets/task73/bottle_02/bottle_02.usdc",
        "bag_place/bottle_02.usdc",
    ),
    (
        Path(DEMO)
        / "house_task_assets/task73/tote_bag/textures/leather_basecolor.png",
        "bag_place/leather_basecolor.png",
    ),
    (
        Path(DEMO)
        / "house_task_assets/task73/tote_bag/textures/leather_roughness.png",
        "bag_place/leather_roughness.png",
    ),
    (
        Path(DEMO)
        / "house_task_assets/task73/tote_bag/textures/leather_normal.png",
        "bag_place/leather_normal.png",
    ),
    (Path(DEMO) / "usdz/umbrella.usdz", "bag_place/umbrella.usdz"),
    (
        Path(DEMO) / "usdz/glasses_case.usdz",
        "bag_place/glasses_case.usdz",
    ),
    (Path(DEMO) / "usdz/wallet.usda", "bag_place/wallet.usda"),
)
TASK73_RUNTIME_LAYERS = BAG_PLACE_RUNTIME_LAYERS
from machine_paths import ROBOT_ASSET_ROOT               # noqa: E402
ROBOT_ASSET = ROBOT_ASSET_ROOT / "robot.usda"
CONTAINER_NAME = _MACHINE_CONTAINER
CONTAINER_MAIN = "/geniesim/main"
CONTAINER_ASSETS = "/geniesim/main/source/geniesim/assets"
# These are the exact layers referenced by robot.usda and bind the live pad
# collision/linkage/camera mount to every runtime and resume identity.  The
# demo container bind-mounts this sibling assets directory at
# /geniesim/main/source/geniesim/assets.
ROBOT_RUNTIME_LAYERS = (
    (ROBOT_ASSET, "robot/G1_omnipicker/robot.usda"),
    (
        ROBOT_ASSET_ROOT / "configuration" / "Cam.usd",
        "robot/G1_omnipicker/configuration/Cam.usd",
    ),
    (
        ROBOT_ASSET_ROOT / "configuration" / "robot_base.usd",
        "robot/G1_omnipicker/configuration/robot_base.usd",
    ),
    (
        ROBOT_ASSET_ROOT / "configuration" / "robot_physics.usd",
        "robot/G1_omnipicker/configuration/robot_physics.usd",
    ),
    (
        ROBOT_ASSET_ROOT / "configuration" / "robot_sensor.usd",
        "robot/G1_omnipicker/configuration/robot_sensor.usd",
    ),
    (
        ROBOT_ASSET_ROOT / "configuration" / "robot_robot.usd",
        "robot/G1_omnipicker/configuration/robot_robot.usd",
    ),
)
KITCHEN_RANDOM_SOURCE_ROOT = (
    Path(REPO) / "real2sim" / "production" / "assets"
    / "kitchen_tableware"
)
KITCHEN_RANDOM_RUNTIME_FILES = (
    (
        Path(REPO) / "real2sim" / "production"
        / "kitchen_material_randomization.py",
        Path(DEMO) / "kitchen_material_randomization.py",
        "kitchen/kitchen_material_randomization.py",
    ),
    (
        Path(REPO) / "real2sim" / "production"
        / "central_table_material_random_options.json",
        Path(DEMO) / "central_table_material_random_options.json",
        "kitchen/central_table_material_random_options.json",
    ),
    (
        Path(REPO) / "real2sim" / "production"
        / "stove_material_random_options.json",
        Path(DEMO) / "stove_material_random_options.json",
        "kitchen/stove_material_random_options.json",
    ),
    (
        Path(REPO) / "real2sim" / "production" / "assets"
        / "central_table" / "T_Worktop_Light_BC.png",
        Path(DEMO) / "constructor_kitchen_final_v1"
        / "door_style_preview_assets" / "T_Worktop_Light_BC.png",
        "kitchen/T_Worktop_Light_BC.png",
    ),
    (
        Path(REPO) / "real2sim" / "production"
        / "kitchen_tableware_randomization.py",
        Path(DEMO) / "kitchen_random" / "generate_kitchen_tableware.py",
        "kitchen/generate_kitchen_tableware.py",
    ),
    (
        Path(REPO) / "real2sim" / "production" / "kitchen_randomization.py",
        Path(DEMO) / "kitchen_random" / "kitchen_randomization.py",
        "kitchen/kitchen_randomization.py",
    ),
) + tuple(
    (
        KITCHEN_RANDOM_SOURCE_ROOT / source_relative,
        KITCHEN_TABLEWARE_ASSET_ROOT / destination_relative,
        f"kitchen/tableware/{destination_relative}",
    )
    for source_relative, destination_relative in (
        *((
            f"bowl_{index:03d}/exports/bowl_{index:03d}.usda.source",
            f"bowl_{index:03d}/exports/bowl_{index:03d}.usda",
        ) for index in range(1, 7)),
        *((
            f"bowl_{index:03d}/textures/bowl_{index:03d}_albedo.png",
            f"bowl_{index:03d}/textures/bowl_{index:03d}_albedo.png",
        ) for index in range(2, 7)),
        *((
            f"plate_{index:03d}/exports/plate_{index:03d}.usda.source",
            f"plate_{index:03d}/exports/plate_{index:03d}.usda",
        ) for index in range(1, 7)),
        *((
            f"plate_{index:03d}/textures/plate_{index:03d}_albedo.png",
            f"plate_{index:03d}/textures/plate_{index:03d}_albedo.png",
        ) for index in range(1, 7)),
        (
            "cutlery_set_v1/chopstick_001/exports/"
            "chopstick_001.usda.source",
            "cutlery_set_v1/chopstick_001/exports/chopstick_001.usda",
        ),
        (
            "cutlery_set_v1/chopstick_001/exports/"
            "chopstick_001_visual.usda.source",
            "cutlery_set_v1/chopstick_001/exports/"
            "chopstick_001_visual.usda",
        ),
    )
)
MANAGED_HOST_SOURCES = (
    Path(__file__).resolve(),
    Path(REPO) / "real2sim" / "teleop"
    / "episode_scene_randomization.py",
    Path(REPO) / "real2sim" / "teleop"
    / "episode_distractor_policy_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "episode_appearance_randomization.py",
    Path(REPO) / "real2sim" / "teleop"
    / "episode_appearance_policy.json",
    Path(REPO) / "real2sim" / "teleop" / "init_pick.py",
    Path(REPO) / "real2sim" / "teleop" / "prepare_native_paper_cup.py",
    Path(REPO) / "real2sim" / "teleop" / "paper_cup_collision_geometry.py",
    Path(REPO) / "real2sim" / "teleop" / "init_task55_context.py",
    Path(REPO) / "real2sim" / "teleop" / "init_task56_context.py",
    Path(REPO) / "real2sim" / "teleop" / "init_task57_context.py",
    Path(REPO) / "real2sim" / "teleop" / "init_task59_context.py",
    Path(REPO) / "real2sim" / "teleop"
    / "init_place_current_context.py",
    Path(REPO) / "real2sim" / "teleop" / "robot_positions.py",
    Path(REPO) / "real2sim" / "teleop" / "group_shift.py",
    BAG_PLACE_CONTRACT_SOURCE,
    Path(REPO) / "real2sim" / "teleop" / "dual_trash_bin_layout.py",
    Path(REPO) / "real2sim" / "teleop"
    / "set_canonical_coffee_table_size.py",
    Path(REPO) / "real2sim" / "teleop"
    / "set_canonical_tablecloth_basket_height.py",
    TABLECLOTH_BASKET_CONTRACT_SOURCE,
    Path(REPO) / "real2sim" / "teleop"
    / "task55_secondary_context_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "task56_contact_scene_calibration_v2.json",
    Path(REPO) / "real2sim" / "teleop"
    / "task56_secondary_context_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "task59_support_calibration_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "task57_secondary_context_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "task59_secondary_context_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "place_current_context_v1.json",
    Path(REPO) / "real2sim" / "teleop"
    / "current_place_exemplars_v1.json",
    Path(REPO) / "real2sim" / "teleop" / "init_task.py",
    Path(REPO) / "real2sim" / "teleop" / "init_snack.py",
    Path(REPO) / "real2sim" / "teleop" / "init_tier2.py",
    Path(REPO) / "real2sim" / "teleop" / "init_place.py",
    Path(REPO) / "real2sim" / "teleop" / "init_throw.py",
    Path(REPO) / "real2sim" / "teleop" / "refine_episode.py",
    Path(REPO) / "real2sim" / "teleop" / "init_task29_full.py",
    Path(REPO) / "real2sim" / "teleop" / "task29_full_reproduce.py",
    Path(REPO) / "real2sim" / "teleop" / "task30_throw_reproduce.py",
    Path(REPO) / "real2sim" / "teleop" / "task29_three_cup_usd.py",
    Path(REPO) / "real2sim" / "teleop" / "task_registry.py",
    Path(REPO) / "real2sim" / "teleop" / "place_sidecar.py",
    Path(REPO) / "real2sim" / "teleop" / "managed_place_lock.py",
    Path(REPO) / "real2sim" / "teleop" / "managed_subprocess.py",
    Path(REPO) / "real2sim" / "teleop" / "generate_cushion_texture.py",
    SHELF_POLICY_SOURCE,
    SHELF_ENFORCER_SOURCE,
)
CRASH_DUMP_LINE = re.compile(
    r"^.*(carb\.crashreporter-breakpad|\[Fatal\] \[carb).*$", re.MULTILINE)


def _without_crash_dump(text: str, keep: int = 6) -> str:
    """Drop the breakpad stack dump, keeping its first few lines.

    The dump is emitted after the traceback that caused it, so tailing raw
    stderr shows only symbol names. The head lines carry the signal, the fault
    address and the faulting module, which is what a genuine mid-run crash
    needs; the remaining hundreds of frames only evict the exception.
    """
    kept, dropped = [], 0
    for line in text.splitlines(keepends=True):
        if CRASH_DUMP_LINE.match(line.rstrip("\n")):
            dropped += 1
            if dropped <= keep:
                kept.append(line)
            continue
        kept.append(line)
    if dropped > keep:
        kept.append(f"... {dropped - keep} further crash-dump lines dropped\n")
    return "".join(kept)


RESERVED_SIM_ENV = {
    "ACTIVE_HAND",
    "CUDA_VISIBLE_DEVICES",
    "EPISODE_UID",
    "HOLD_HAND",
    "HOLD_MODE",
    "HOLD_PRIM",
    "HOLD_SURFACE_Z",
    "INSTANCE",
    "METRICS_ONLY",
    "PICK_CONTACT_INSTRUMENTATION",
    "PLACE_RUN_ID",
    "PLACE_TASK_ID",
    "PLACE_DESTINATION_KIND",
    "PLACE_DESTINATION_BODY_PATH",
    "PLACE_DESTINATION_BOTTOM_Z",
    "PLACE_ZONE",
    "RUNTIME_SOURCE_REVISION",
    "SCENE_RANDOMIZATION_SEED",
    "REAL2SIM_APPEARANCE_RANDOMIZATION",
    "REAL2SIM_KITCHEN_TABLEWARE_RANDOMIZATION",
    "TASK_OWNED_ARM_INITIAL_DELTA_RAD",
    "TASK38_INIT_ROBOT_STATE",
    "DISTRACTOR_TRACK_PRIMS",
    "REAL2SIM_TASK_ID",
    "REAL2SIM_TARGET_PRIM",
    "REAL2SIM_RECOVERED_PLAN",
    "GRIP_WIDTH",
    "PLACE_APPROACH_OFFSET_XYZ",
    "RELEASE_RETREAT",
    "SEAT_LONG",
    "SEAT_MID",
    "UP_AXIS",
    "SCENE_USDA",
    "POLICY_SCENE_USDA",
    "HEAD_ADJ",
    "HEAD_CAMERA_LOCAL_X_OFFSET_M",
    "HEAD_CAMERA_LOCAL_Y_OFFSET_M",
    "HEAD_CAMERA_LOCAL_Z_OFFSET_M",
    "HEAD_CAMERA_MOUNT_PITCH_DEG",
    "HEAD_FOCAL_SCALE",
    "HEAD_PRINCIPAL_X_PX",
    "HEAD_PRINCIPAL_Y_PX",
    "WRIST_FOCAL_SCALE",
}
ALLOWED_SIM_ENV = {
    "APPROACH_MIN_OPEN_MM",
    "BOOK_ITERS",
    "CLAMP_STEPS",
    "CONTACT_FORCE_THR",
    "CUSHION_CONTACT_DAMPING",
    "CUSHION_CONTACT_STIFFNESS",
    "TASK53_DYNAMIC_FRICTION",
    "TASK53_MIN_TORSIONAL_PATCH_RADIUS",
    "TASK53_REAL_VISUALS",
    "TASK53_STATIC_FRICTION",
    "TASK53_TORSIONAL_PATCH_RADIUS",
    "TASK55_GRIP_ADVANCE_FRAMES",
    "TASK55_PAD_CONTACT_DAMPING",
    "TASK55_PAD_CONTACT_STIFFNESS",
    "TASK55_PAD_DYNAMIC_FRICTION",
    "TASK55_PAD_STATIC_FRICTION",
    "TASK56_BOX_COLLISION",
    "TASK56_MIN_TORSIONAL_PATCH_RADIUS",
    "TASK56_TORSIONAL_PATCH_RADIUS",
    "TASK59_MIN_TORSIONAL_PATCH_RADIUS",
    "TASK59_TORSIONAL_PATCH_RADIUS",
    "TASK73_BEND",
    "TASK73_COLLISION_ITERATION_MULTIPLIER",
    "TASK73_COLLISION_PAIR_UPDATE_FREQUENCY",
    "TASK73_DENSITY",
    "TASK73_FRICTION",
    "TASK73_LINEAR_DAMPING",
    "TASK73_POISSON",
    "TASK73_SETTLING_DAMPING",
    "TASK73_SHEAR",
    "TASK73_SOLVER_POSITION_ITERATIONS",
    "TASK73_STRETCH",
    "TASK73_THICKNESS",
    "TASK73_YOUNGS",
    "GRASP_LONG_AXIS_SIGN",
    "GRASP_LONG_CAMERA_BLEND",
    "GRASP_LONG_PLANE_ROTATION_DEG",
    "GAP_PER_RAD",
    "GRIP_CONTACT",
    "GRIP_DAMP",
    "GRIP_FORCE",
    "GRIP_PAST",
    "GRIP_SQUEEZE",
    "GRIP_STIFF",
    "GLASSES_CASE_BLACK_VISUAL",
    "UMBRELLA_DARK_VISUAL",
    "HOLD_THR",
    "LIFT_OFFSET",
    "LIMIT",
    "PHYS_HZ",
    "POST_STEPS",
    "PREOPEN_MARGIN",
    "SEAT_OFF",
}

TASK73_GRASP_CALIBRATIONS = {
    "012e2786-cd7b-4993-93c2-8ebf3f3771a4": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.03,
        "SEAT_MID": 0.0,
    },
    "0366cec8-5ee0-4989-8eea-33c504934978": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.03,
        "SEAT_MID": 0.0,
    },
    "07bcc657-42d6-4506-ae76-7c5ca5e8bca2": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": -0.05,
        "SEAT_MID": 0.0,
    },
    "118506ff-44e1-48d4-a901-9d65e86fdce8": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": -0.025,
        "SEAT_MID": 0.02,
    },
    "12212ad3-87d7-4535-8710-ecd1e9acc7bb": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": -0.05,
        "SEAT_MID": 0.0,
    },
    "13cd844c-6422-4ec7-bcc8-6edb0b28f9c6": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 1.0,
        "SEAT_LONG": 0.08,
        "SEAT_MID": 0.0,
    },
    "15bdaf36-d223-4d5a-b0af-166d3bd2af98": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.03,
        "SEAT_MID": 0.0,
    },
    "1d7cdc7b-b652-4d54-9226-cf1aacbd2c41": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.08,
        "SEAT_MID": 0.0,
    },
    "2058a978-2265-4b8e-ab47-488402140226": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": -0.03,
        "SEAT_MID": 0.0,
    },
    "2237d9aa-54f9-4549-b6c8-6c33a2097963": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.08,
        "SEAT_MID": 0.0,
    },
    "27e0f92b-ea62-45ae-a8c6-db85168a6be5": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.08,
        "SEAT_MID": 0.0,
    },
    "4ec374e7-317e-48d5-833d-f4fbc5ba8277": {
        "GRASP_LONG_AXIS_SIGN": 1.0,
        "GRASP_LONG_CAMERA_BLEND": 0.0,
        "SEAT_LONG": 0.05,
        "SEAT_MID": 0.0,
    },
}


def task73_grasp_calibration_environment(episode_uid):
    """Return the bounded per-episode grasp controls handed off on l2."""
    calibration = TASK73_GRASP_CALIBRATIONS.get(str(episode_uid), {})
    environment = {}
    allowed = {
        "SEAT_LONG",
        "SEAT_MID",
        "GRASP_LONG_AXIS_SIGN",
        "GRASP_LONG_CAMERA_BLEND",
        "GRASP_LONG_PLANE_ROTATION_DEG",
    }
    for key, value in calibration.items():
        if key not in allowed:
            raise ValueError(f"unsupported task73 grasp calibration {key}")
        if isinstance(value, bool) or type(value) not in (int, float):
            raise ValueError(f"task73 grasp calibration {key} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"task73 grasp calibration {key} is not finite")
        if key == "GRASP_LONG_CAMERA_BLEND" and not 0.0 <= numeric <= 1.0:
            raise ValueError(
                "task73 GRASP_LONG_CAMERA_BLEND must be within [0, 1]")
        if key == "GRASP_LONG_AXIS_SIGN" and numeric not in (-1.0, 1.0):
            raise ValueError(
                "task73 GRASP_LONG_AXIS_SIGN must be exactly -1 or +1")
        if (
            key == "GRASP_LONG_PLANE_ROTATION_DEG"
            and not -90.0 <= numeric <= 90.0
        ):
            raise ValueError(
                "task73 GRASP_LONG_PLANE_ROTATION_DEG must be within "
                "[-90, 90]")
        environment[key] = str(numeric)
    return environment

TASK_SIM_ENV_DEFAULTS = {
    # small Task 58 does not merely prefer this value: its task-bound runtime
    # (playback_video3_task58_current.py:46-49) REFUSES to run without it, so
    # leaving it unregistered made the task permanently unrunnable from the
    # queue -- the requirement lived only in whoever remembered to pass it.
    # -20 mm is the measured waist correction for the TV cabinet top.
    "58": {"LIFT_OFFSET": "-0.02"},
    # The soft-wallet pad material is a property of the wallet and the pads,
    # not of any one recording: all four reviewed episodes carried these same
    # four values. Kept per-episode they read as "this task needs contact
    # tuning per recording", which is the single most expensive claim a
    # fifty-episode batch can inherit. GRIP_PAST stays per-episode because it
    # genuinely differs (-0.15 on three, -0.22 on one).
    "55": {
        "TASK55_PAD_CONTACT_DAMPING": "300",
        "TASK55_PAD_CONTACT_STIFFNESS": "3000",
        "TASK55_PAD_DYNAMIC_FRICTION": "4",
        "TASK55_PAD_STATIC_FRICTION": "4",
    },
    # Articulated-drawer bill place: the pull hand's recorded closure drives
    # to FULL-SHUT against the ~17 mm front wall; the unbounded penetration
    # demand makes the solver eject the drawer westward ("pushed closed",
    # the dominant round-3 failure).  A NEGATIVE closure bound stops the
    # pads ~12 mm apart — a firm friction pinch without the crush (task-91
    # anti-eject precedent).  Positive GRIP_PAST=0.20 was separately
    # measured worse (0.160 -> 0.103 m); the LEFT bill hold is clamp-managed
    # and unaffected by this bound.
    # 2026-08-03: the bill is carried into the drawer at +46.4 deg. A
    # 253 x 206 mm sheet rotated 46 deg has a 324 x 325 mm footprint
    # against a 311 x 224 mm drawer opening, so it cannot enter at any
    # depth -- measured: it reached 0.275 (62 mm inside) then was pushed
    # back to 0.337, the rim. -46 deg cancels that carry angle and the
    # sheet settles at 0.216, i.e. 12 mm off the drawer floor. The value
    # is the measured carry angle, not a swept constant.
    "28": {
        "GRIP_PAST": "-0.10",
        "GRASP_LONG_PLANE_ROTATION_DEG": "-46",
    },
    # Real frame 0 shows a fabric cushion loading the rear portion of the
    # target book. These values are the video-verified compliant-contact and
    # recorded-closure calibration for that exact small-task context.
    "12": {
        "CUSHION_CONTACT_DAMPING": "20",
        "CUSHION_CONTACT_STIFFNESS": "300",
        "GRIP_PAST": "0.30",
    },
    "53": {
        "BOOK_ITERS": "16",
        "PHYS_HZ": "60",
    },
    # The old accepted rung used 240 Hz.  The global solver clock is now 60 Hz,
    # so that result is audit-only until the exact action passes a 60 Hz replay;
    # keep only the geometry/material and iteration settings here.
    "56": {
        "BOOK_ITERS": "32",
        "GRIP_PAST": "-0.05",
        "PHYS_HZ": "60",
        "TASK56_BOX_COLLISION": "1",
        "TASK56_MIN_TORSIONAL_PATCH_RADIUS": "0.003",
        "TASK56_TORSIONAL_PATCH_RADIUS": "0.006",
    },
    # The real hard cases in small Tasks 57 and 59 are black. Keep the
    # generated collision/rigid asset unchanged and correct only its rendered
    # lid/wall material composition.
    "57": {
        "GLASSES_CASE_BLACK_VISUAL": "1",
        # Installing the recorded frame-zero robot state prevents the shared
        # room's legacy pose from rolling the rounded case during warm-up.
        # At the default 500 N/rad drive, however, closure trails the recorded
        # hand by several frames and one pad pushes the case away.  The bounded
        # 700 N/rad drive yields 129 consecutive bilateral-contact frames,
        # 19.7/23.3 N peak pad forces, and a 285.7 mm retained lift without
        # visible pad interpenetration.
    },
    "59": {
        "GLASSES_CASE_BLACK_VISUAL": "1",
        "GRIP_PAST": "-0.15",
        "TASK59_MIN_TORSIONAL_PATCH_RADIUS": "0.003",
        "TASK59_TORSIONAL_PATCH_RADIUS": "0.006",
    },
    # Same folded Umbrella as Task 54; start from its accepted contact rung
    # (65 mm cylinder stalls the pads, so the softer closure envelope applies).
    "90": {
        "GRIP_PAST": "-0.25",
        "PHYS_HZ": "60",
        "BOOK_ITERS": "16",
        "UMBRELLA_DARK_VISUAL": "1",
    },
    # White-shell GlassesCase on the sofa seat.  The recorded closure goes to
    # a full-shut command; unbounded it squeezes the 60 mm case out sideways
    # (trace: gap -> 1.5 mm, case ejected 30 mm at frames 140-150).  Bound the
    # closure past-contact depth like the Task 57 gallery rung and keep the
    # 700 N/rad drive so closure does not trail the recorded hand.
    "91": {
        "GRIP_PAST": "-0.28",
    },
    # Wallet-out-of-drawer: the pads meet the standing pillow near its
    # tapered top seam, so the closure needs to drive past the seam to seat
    # on the fabric body (real pads bite 3-4 cm deep); 700 N/rad keeps the
    # closure from trailing the recorded hand.
    # Historical A/B at 120 Hz with BOOK_ITERS=16/GRIP_FORCE=60 regressed the
    # catch from 16.1 mm to 0.1 mm.  That solver-rate result is audit-only under
    # the global 60 Hz contract; keep the minimal drive envelope.
    "92": {
        # The wallet is 40 mm across the pinch axis but the recorded closure
        # only ever reaches the command that held a SOFT pouch; against the
        # rigid proxy that leaves the pads short and the wallet slides out
        # during the lift (23.7 mm up, then dropped).  Drive further past
        # the recorded command so the pads actually load the 40 mm faces.
        "GRIP_PAST": "0.35",
    },
}

EPISODE_SIM_ENV_DEFAULTS = {
    # This umbrella grasp sits at the shallow edge of the stalled-pad
    # envelope and intermittently slips during the retract on some scene
    # orderings (217 mm lift then drop).  One rung deeper closure retains it.
    ("90", "0119c688-ae3f-4512-aeb1-d07c0e4d91ea"): {
        "GRIP_PAST": "-0.20",
    },
    # Audited-l2 publication canary with the same filled pouch and current
    # frame-zero context.  Start from the contact material rung proven on the
    # visually matching 7c7a3f0d episode; the new run must independently pass
    # contact and REAL3/SIM3 review before these values are accepted.
    ("55", "01ed7e25-5dc6-4f5b-bfea-5ceca2d34143"): {
        "GRIP_PAST": "-0.15",
    },
    # Video/trace-reviewed small Task 55 soft-pouch episode.  On the explicit
    # frame-zero context and current shared runtime, 4 kN/m retains bilateral
    # contact for only 11 frames and drops after a 37.6 mm transient lift.
    # The bounded 3 kN/m rung yields 128 bilateral frames (114 consecutive), a
    # 278.7 mm retained lift, and 33.1/56.1 N peak pad forces.
    ("55", "7c7a3f0d-d368-4286-9274-e4f8041c9859"): {
        "GRIP_PAST": "-0.15",
    },
    # The second and third reviewed pouches use the same compliant material
    # rung; their episode-bound centre/yaw controls are still independently
    # contact- and video-reviewed.  On the second episode, -0.15
    # over-compresses the pouch at 106.1/78.0 N.  The bounded -0.22 closure
    # keeps 94 consecutive bilateral frames and a 259.0 mm retained lift at
    # 45.2/21.3 N with clear pad boundaries in all three views.
    ("55", "f60be28b-c295-4a31-9f07-cc57f4270739"): {
        "GRIP_PAST": "-0.22",
    },
    # The third reviewed recording shows the same filled fabric pouch and uses
    # the same bounded compliant-pad material.  Its grasp-centre/yaw controls
    # remain episode-bound below and are tuned from contact evidence.
    ("55", "6fa97ae0-cf4e-49ca-9fe0-1712e0d83d89"): {
        "GRIP_PAST": "-0.15",
    },
    # The three reviewed small Task 56 contact contexts need different final
    # pad gaps.  The former shared -0.05 rad over-closure produced sustained
    # 200--430 N impulses despite visually successful lifts.  These
    # trace-reviewed values retain long bilateral lifts without visible pad
    # penetration on the current robot station and explicit frame-zero
    # context: Note -0.20; Wallet -0.22; Book -0.25.  The older Wallet -0.26
    # setting left a 50 mm physical pad gap around the 47.55 mm contact box and
    # produced only sequential one-sided taps.
    ("56", "003d1033-5f4f-4030-9484-0f2aa53495e6"): {
        "GRIP_PAST": "-0.20",
        "LIFT_OFFSET": "-0.011601128",
    },
    ("56", "07ec98c4-9e86-4abe-a301-43ec193068e6"): {
        "GRIP_PAST": "-0.22",
        "LIFT_OFFSET": "-0.0117105",
    },
    ("56", "0a5c5dba-3cf7-4f9f-9d18-8117670e37cd"): {
        "GRIP_PAST": "-0.25",
        "LIFT_OFFSET": "-0.011075992",
    },
    # Trace-reviewed small Task 57: at the exact recorded closure trajectory,
    # -0.30 rad leaves a 58 mm gap but loses the rounded 60 mm case during the
    # final acceleration; -0.28 rad leaves 54.7 mm, 125 bilateral-contact
    # frames, and a 295.7 mm final lift without visible pad interpenetration.
    ("57", "0138ae6f-3d50-4539-b71c-90a80c961a95"): {
        "GRIP_PAST": "-0.28",
    },
}

TASK_INIT_COMMAND_DEFAULTS = {
    # Articulated coffee-table drawer place: the right hand pulls the
    # black_box_2 drawer open on the shared prismatic-slide physics, the
    # left hand drops the held bill in.  IH_YAW=90 points the bill's long
    # axis into the drawer (a cross-slot bill bridges the wall lips).
    "28": (
        "EPISODE_UID={episode_uid} PLACE_TASK=28 IH_YAW=90 "
        "python real2sim/teleop/init_place.py"
    ),
    "29": "python real2sim/teleop/init_task29_full.py",
    # Shelf drink picks initialise through the snack family: one fixed base on
    # the shelf mid-axis, only the target drink moves to the episode's grasp
    # point.  init_snack takes the template NAME, not the task id.
    "8": "python real2sim/teleop/init_snack.py coke",
    # Retail snack tier (tier 2) picks go through their own initializer: the
    # board is 43 cm below the drink tier and each product maps to one prim.
    # 2026-07 tier2 batch anchor for the Honey Maid box (PASS, left hand).
    # 2026-08-02: the shelf station moved +0.030 m toward the shelf, and every
    # tier-2 anchor below was recorded against the old one. The replayed arm is
    # relative to the base, so the hand now reaches 30 mm further in y and the
    # anchor has to follow it by the same amount to stay on the object.
    # Recorded-orientation staging: the real box is parked at an angle and the
    # arm was recorded sweeping in along it; face-on staging put a 45 mm-thin
    # box across that path and the approach knocked it over (never one passing
    # run in any configuration). Deriving the box yaw from the recorded pinch
    # axis is what first made this task pass: 151.3 mm held 3.6 s. The pinned
    # anchor is gone because the align pass supplies both position and
    # orientation per episode -- which is also what a 50-episode batch needs.
    "37": "python real2sim/teleop/init_tier2.py 37",
    # 35/39 read low (44.3 / 5.2 mm) since the shelf station moved +30 mm --
    # their pre-move passes (121.7 / 158.2) are the baseline to recover via
    # anchor re-calibration, and that work should happen WITHOUT the yaw
    # variable stacked on top. Yaw stays off here until they are re-solved;
    # 37 proves the two fixes can then be combined (it passes 151.3 at the
    # same new station with yaw on).
    "35": "T2_ALIGN_YAW=0 python real2sim/teleop/init_tier2.py 35",
    "39": "T2_ALIGN_YAW=0 python real2sim/teleop/init_tier2.py 39",
    "41": "PLACE_TASK=41 python real2sim/teleop/init_place.py",
    # The 2026-07 tier2 batch solved this episode's anchor (PASS, 92-98 mm
    # lift); re-deriving it from align_grasp lands 66 cm behind the board.
    # T2_KIN_OVERRIDE takes the pre-layout-shift frame the batch recorded.
    "40": (
        "T2_KIN_OVERRIDE=-2.022,1.387,0.856 T2_HAND_OVERRIDE=R "
        "python real2sim/teleop/init_tier2.py 40"
    ),

    "53": (
        "EPISODE_UID={episode_uid} PICK_TASK=53 "
        "PICK_TASK53_REAL_SCENE=1 BITE_Z={task53_bite_z} "
        "python real2sim/teleop/init_pick.py"
    ),
    "55": "PICK_TASK=55 python real2sim/teleop/init_pick.py",
    "56": (
        "EPISODE_UID={episode_uid} PICK_TASK=56 "
        "PICK_PAD_AXIS_OFFSET=-0.0075 PICK_BITE=-0.010 "
        "python real2sim/teleop/init_pick.py"
    ),
    "57": (
        "EPISODE_UID={episode_uid} PICK_TASK=57 "
        "python real2sim/teleop/init_pick.py"
    ),
    "78": (
        "EPISODE_UID={episode_uid} PICK_TASK=78 "
        "python real2sim/teleop/init_pick.py"
    ),
    "59": (
        "EPISODE_UID={episode_uid} PICK_TASK=59 "
        "PICK_GLASSES_CASE_SIDE_REST=1 "
        "PICK_GLASSES_CASE_LONG_AXIS_OFFSET=0.015 "
        "PICK_PAD_AXIS_OFFSET=0.0025 BITE_Z=0.030 "
        "python real2sim/teleop/init_pick.py"
    ),
    "90": (
        # BITE_Z=0.030 (Task 59 precedent): the real folded canopy is a
        # ~9-10 cm soft bundle, the rigid asset a 65 mm cylinder, so the
        # recorded closure height skims the asset top (+48..60 mm above the
        # object centre) and the catch is nondeterministic at the default
        # 12 mm bite.  The deeper bite converts it into a deterministic wrap.
        # EDGE_OVERHANG=0.01: the default 55 mm riser shift runs almost
        # perpendicular to the umbrella's long axis, so the cylinder tips and
        # rolls off the riser during settle (persistent misses on the
        # high-grasp episodes).  The fingers stall at a 48 mm pad gap well
        # above the riser top, so a centred riser cannot block them.
        "EPISODE_UID={episode_uid} PICK_TASK=90 BITE_Z=0.030 "
        "EDGE_OVERHANG=0.01 "
        "python real2sim/teleop/init_pick.py"
    ),
    "91": (
        "EPISODE_UID={episode_uid} PICK_TASK=91 "
        "python real2sim/teleop/init_pick.py"
    ),
    # Find the wallet inside the coffee-table black drawer (2026-07-31
    # batch): right hand pulls black_box_2 open on the shared articulated
    # drawer physics, left hand lifts the wallet out.
    "92": (
        "EPISODE_UID={episode_uid} PICK_TASK=92 "
        "python real2sim/teleop/init_pick.py"
    ),
}

EPISODE_INIT_COMMAND_DEFAULTS = {
    # Left-hand sofa episode: the approach path skims the case and pushes it
    # 119 mm before closure on some scene orderings.  Shift the case a small
    # bounded distance along the pinch axis away from the pushing pad.
    ("91", "0409f50d-e759-46b0-8c58-5c818c90a390"): (
        "EPISODE_UID={episode_uid} PICK_TASK=91 "
        "PICK_PAD_AXIS_OFFSET=-0.010 "
        "python real2sim/teleop/init_pick.py"
    ),
    # Audited-l2 episode selected after direct REAL3 frame-zero/final review.
    # Its target presents the same nearly horizontal edge as 7c7a3f0d.  The
    # contact trace and comparison video remain authoritative for whether the
    # inherited yaw seed and bounded +7.5 mm centre are retained.
    ("55", "01ed7e25-5dc6-4f5b-bfea-5ceca2d34143"): (
        "EPISODE_UID={episode_uid} PICK_TASK=55 "
        "PICK_TASK55_SOFT_WALLET_COLLISION=1 "
        "PICK_INITIAL_YAW_REFERENCE_DEG=-170 "
        "PICK_PAD_AXIS_OFFSET=0.0075 "
        "python real2sim/teleop/init_pick.py"
    ),
    # The reviewed real target is a filled fabric pouch.  The bounded soft
    # collision envelope and +7.5 mm pad-axis centre yield balanced bilateral
    # loading on the canonical CoffeeTable; the generic rigid Wallet remains
    # the fail-closed default for every other small Task 55 episode.
    ("55", "7c7a3f0d-d368-4286-9274-e4f8041c9859"): (
        "EPISODE_UID={episode_uid} PICK_TASK=55 "
        "PICK_TASK55_SOFT_WALLET_COLLISION=1 "
        "PICK_INITIAL_YAW_REFERENCE_DEG=-170 "
        "PICK_PAD_AXIS_OFFSET=0.0075 "
        "python real2sim/teleop/init_pick.py"
    ),
    # This real episode presents the same undirected pinch on the -38-degree
    # representative.  The antipodal +142-degree solution never establishes
    # simultaneous pad contact, so bind the observed face/stripe orientation.
    ("55", "f60be28b-c295-4a31-9f07-cc57f4270739"): (
        "EPISODE_UID={episode_uid} PICK_TASK=55 "
        "PICK_TASK55_SOFT_WALLET_COLLISION=1 "
        "PICK_INITIAL_YAW_REFERENCE_DEG=-38 "
        "python real2sim/teleop/init_pick.py"
    ),
    # The third frame-zero target has the same nearly horizontal long edge as
    # the first reviewed pouch.  Start from its visually measured -176-degree
    # representative; the bounded pad-axis offset is refined from the current
    # contact trace rather than inherited from another episode.
    ("55", "6fa97ae0-cf4e-49ca-9fe0-1712e0d83d89"): (
        "EPISODE_UID={episode_uid} PICK_TASK=55 "
        "PICK_TASK55_SOFT_WALLET_COLLISION=1 "
        "PICK_INITIAL_YAW_REFERENCE_DEG=-176 "
        "python real2sim/teleop/init_pick.py"
    ),
    # On the current shared robot station, centre/old -0.26 closure leaves the
    # Wallet-supported Remote between a 50 mm pad gap with only sequential
    # one-sided taps.  A bounded +5 mm pad-axis centre with -0.22 closure
    # yields 120 consecutive bilateral frames, a 220.5 mm lift, and balanced
    # 43.4/43.3 N peak forces without visible pad penetration.
    ("56", "07ec98c4-9e86-4abe-a301-43ec193068e6"): (
        "EPISODE_UID={episode_uid} PICK_TASK=56 "
        "PICK_PAD_AXIS_OFFSET=0.005 PICK_BITE=-0.010 "
        "python real2sim/teleop/init_pick.py"
    ),
    # The target remains directly on the calibrated, visible CoffeeTable.
    # Contact tracing on that real surface showed that -9 mm left the far pad
    # as a one-sided pusher (one bilateral frame, 1.2 mm lift).  The reviewed
    # The former -17 mm correction was accepted before the CoffeeTable layout
    # shift and now misses the case.  On the current canonical robot/table
    # layout, -22 mm yields 126 consecutive bilateral-contact frames, a
    # 283.7 mm lift, and 17.1/20.1 N peak pad forces without visible pad
    # interpenetration.  It changes neither robot, table, nor object height.
    ("57", "0138ae6f-3d50-4539-b71c-90a80c961a95"): (
        "EPISODE_UID={episode_uid} PICK_TASK=57 "
        "PICK_PAD_AXIS_OFFSET=-0.022 BITE_Z=0.012 "
        "python real2sim/teleop/init_pick.py"
    ),
}

# The original sofa pipeline predates task_registry.py and still uses its
# dedicated init/refine scripts. These exact robot-template IDs keep a small,
# explicit compatibility surface instead of trusting an arbitrary target
# returned by a legacy initializer.
LEGACY_RUNTIME_TARGET_PRIMS = {
    "11": "/World/Assets/Note",
    "12": "/World/Assets/Book",
    "13": "/World/Assets/Book",
}


def run(command, timeout=None, *, cwd=None, env=None):
    """Run an argv command without exposing caller-controlled values to a host shell."""
    return run_argv(command, timeout=timeout, cwd=cwd, env=env)


def parse_env_assignments(text):
    """Parse a shell-like list of KEY=VALUE tokens, but never accept commands."""
    try:
        tokens = shlex.split(text or "")
    except ValueError as exc:
        raise ValueError(f"invalid environment assignment syntax: {exc}") from exc
    result = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"expected environment assignment, got {token!r}")
        if key in result:
            raise ValueError(f"duplicate environment assignment {key}")
        if "\x00" in value:
            raise ValueError(f"NUL is not allowed in environment value {key}")
        result[key] = value
    return result


def parse_command_spec(text):
    """Split ``KEY=value command args`` into an argv and explicit env updates."""
    try:
        tokens = shlex.split(text or "")
    except ValueError as exc:
        raise ValueError(f"invalid command syntax: {exc}") from exc
    updates = {}
    index = 0
    while index < len(tokens):
        key, separator, value = tokens[index].partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            break
        if key in updates:
            raise ValueError(f"duplicate environment assignment {key}")
        if "\x00" in value:
            raise ValueError(f"NUL is not allowed in environment value {key}")
        updates[key] = value
        index += 1
    command = tokens[index:]
    if not command:
        raise ValueError("init command contains no executable")
    return command, updates


def run_command_spec(text, *, timeout, cwd, extra_env=None):
    command, updates = parse_command_spec(text)
    child_env = os.environ.copy()
    for key, value in (extra_env or {}).items():
        value = str(value)
        if key in updates and updates[key] != value:
            raise ValueError(
                f"init command {key} conflicts with current episode identity")
        child_env[key] = value
    child_env.update(updates)
    # Older callers spell their base-env init as ``conda run -n base
    # python ...``. Once this process owns the managed OFD lock, that wrapper
    # is unsafe because it closes pass_fds before spawning the Python child.
    # The current interpreter is already that base Python.
    if (lock_pass_fds(child_env)
            and command[:5] == ["conda", "run", "-n", "base", "python"]):
        command = [sys.executable, *command[5:]]
    return run(command, timeout=timeout, cwd=cwd, env=child_env)


def resolve_episode_dir(identifier, record_root=None):
    """Resolve an exact UUID or an unambiguous safe UUID prefix."""
    if (not isinstance(identifier, str)
            or not re.fullmatch(r"[0-9A-Fa-f-]{4,64}", identifier)):
        raise ValueError(f"invalid episode id/prefix {identifier!r}")
    root = Path(record_root or REC)
    # Production callers normally pass a complete UUID.  Resolve that one
    # directory directly before enumerating the record root: the audited
    # archive can live on a read-only external exFAT volume, where listing the
    # large root is both unnecessary and capable of blocking in exfat_lookup.
    # Prefix lookup deliberately retains the exhaustive ambiguity check below.
    try:
        canonical = str(uuid.UUID(identifier))
    except ValueError:
        canonical = None
    if canonical is not None:
        exact_candidates = [root / identifier]
        canonical_path = root / canonical
        if canonical_path not in exact_candidates:
            exact_candidates.append(canonical_path)
        try:
            for candidate in exact_candidates:
                if candidate.is_dir():
                    return candidate
        except OSError as exc:
            raise ValueError(
                f"cannot resolve exact episode under {root}: {exc}") from exc
    try:
        matches = sorted(
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith(identifier)
        )
    except OSError as exc:
        raise ValueError(f"cannot enumerate record root {root}: {exc}") from exc
    exact = [path for path in matches if path.name == identifier]
    if len(exact) == 1:
        return exact[0]
    if len(matches) != 1:
        raise ValueError(
            f"episode id {identifier!r} matched {len(matches)} directories")
    return matches[0]


def resolve_episode_task_id(episode, manifest_paths=None):
    """Resolve task identity from raw metadata and cross-check all manifests.

    Old recordings live only in ``manifest.jsonl`` while newer ones live in
    ``manifest_new.jsonl``.  The raw episode's ``meta_info.json`` is primary;
    every manifest row that exists for the exact UUID must agree with it.
    """
    episode_path = Path(episode) if isinstance(episode, (str, Path)) else None
    has_episode_directory = bool(
        episode_path is not None and episode_path.is_dir()
    )
    episode_uid = episode_path.name if has_episode_directory else str(episode)
    try:
        canonical_uid = str(uuid.UUID(str(episode_uid)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"invalid exact episode UUID {episode_uid!r}") from exc

    authoritative_task = None
    if has_episode_directory:
        metadata_path = episode_path / "meta_info.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read raw episode metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"raw episode metadata is not an object: {metadata_path}")
        raw_task = metadata.get("task_id")
        if isinstance(raw_task, bool) or not isinstance(raw_task, (int, str)):
            raise ValueError(f"raw episode metadata has invalid task_id: {metadata_path}")
        authoritative_task = str(raw_task)

    if manifest_paths is None:
        paths = (MANIFEST_PATH, LEGACY_MANIFEST_PATH)
        explicit_manifests = False
    elif isinstance(manifest_paths, (str, Path)):
        paths = (Path(manifest_paths),)
        explicit_manifests = True
    else:
        paths = tuple(Path(path) for path in manifest_paths)
        explicit_manifests = True

    task_ids = set()
    for path in paths:
        try:
            stream = path.open(encoding="utf-8")
        except OSError as exc:
            if not explicit_manifests and not path.exists():
                continue
            raise ValueError(f"cannot read task manifest {path}: {exc}") from exc
        with stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid manifest JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"manifest row at {path}:{line_number} is not an object"
                    )
                if record.get("uuid") != canonical_uid:
                    continue
                task_id = record.get("task_id")
                if isinstance(task_id, bool) or not isinstance(
                        task_id, (int, str)):
                    raise ValueError(
                        f"manifest task_id at {path}:{line_number} is invalid"
                    )
                task_ids.add(str(task_id))

    if authoritative_task is not None:
        conflicts = sorted(task_ids - {authoritative_task})
        if conflicts:
            raise ValueError(
                f"raw episode task {authoritative_task} conflicts with "
                f"manifest task ids {conflicts} for {canonical_uid}"
            )
        return authoritative_task
    if len(task_ids) != 1:
        raise ValueError(
            f"episode {canonical_uid} maps to {len(task_ids)} canonical "
            f"task ids in manifests: {sorted(task_ids)}"
        )
    return next(iter(task_ids))


def runtime_target_prim(task_id, init_result=None):
    """Return the canonical runtime target for registry or legacy sofa IDs."""
    task_id = str(task_id)
    init_result = dict(init_result or {})
    if task_id in REGISTRY:
        entry = REGISTRY[task_id]
        if entry.get("prim"):
            expected = track_prim(task_id)
        else:
            # Shelf drink picks resolve their exact can/bottle at runtime.
            # Accept that initializer result only when it is one of the
            # canonical, policy-approved prims for this exact task.
            declared = init_result.get("track_prim")
            decision = require_task_asset_ready(task_id)
            allowed = {
                f"{decision.root}/{prim}"
                for prim in decision.prims
            }
            if not declared:
                raise ValueError(
                    f"task {task_id} requires a declared runtime target")
            if declared not in allowed:
                raise ValueError(
                    "init target disagrees with canonical task identity: "
                    f"{declared} not in {sorted(allowed)}")
            return declared
    elif task_id in LEGACY_RUNTIME_TARGET_PRIMS:
        expected = LEGACY_RUNTIME_TARGET_PRIMS[task_id]
    else:
        raise ValueError(
            f"task {task_id} has no canonical runtime target")
    declared = init_result.get("track_prim")
    if declared and declared != expected:
        raise ValueError(
            "init target disagrees with canonical task identity: "
            f"{declared} != {expected}")
    return expected


def verify_episode_asset_policy(task_id):
    """Verify the task target and every shelf model before shared mutation."""
    try:
        decision = require_task_asset_ready(
            str(task_id),
            verify_runtime=True,
            scene_path=SCENE_PATH,
            demo_root=DEMO,
        )
        revisions = require_ready_scene_assets(
            SCENE_PATH,
            demo_root=DEMO,
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return decision, revisions


def validate_retail_init_result(result, episode_task, decision):
    """Reject a custom initializer that swaps or omits the approved model."""
    if not decision.applicable:
        return
    if not isinstance(result, dict):
        raise ValueError("retail init result must be a dictionary")
    if str(result.get("task", "")) != str(episode_task):
        raise ValueError(
            f"retail init task mismatch: episode task={episode_task}, "
            f"init task={result.get('task')!r}"
        )
    if result.get("asset_id") != decision.asset_id:
        raise ValueError(
            f"retail init asset mismatch for task {episode_task}: "
            f"expected {decision.asset_id!r}, got {result.get('asset_id')!r}"
        )
    if result.get("asset_status") != READY:
        raise ValueError(
            f"retail init did not verify a ready refined asset: "
            f"{result.get('asset_status')!r}"
        )
    if result.get("asset_revision") != decision.asset_revision:
        raise ValueError(
            f"retail init asset revision mismatch for task {episode_task}"
        )


def validate_task29_full_init_result(
    result,
    *,
    episode_task,
    episode_uid,
):
    """Reject partial/legacy small Task 29 initializers before playback."""

    if str(episode_task) != "29":
        return
    if not isinstance(result, dict):
        raise ValueError("small Task 29 init result must be a dictionary")
    expected_identity = {
        "family": "throw",
        "task": "29",
        "target": "PaperCup",
        "track_prim": TASK29_FULL_SOURCE_CUP_PATH,
        "required_count": 3,
        "probe_scope": "full_three_cup_continuous",
        "episode_uid": TASK29_FULL_EPISODE_UID,
    }
    if str(episode_uid) != TASK29_FULL_EPISODE_UID:
        raise ValueError(
            "small Task 29 current full reproduce requires reviewed episode "
            f"{TASK29_FULL_EPISODE_UID}"
        )
    for key, expected in expected_identity.items():
        if result.get(key) != expected:
            raise ValueError(
                f"small Task 29 full init {key} mismatch: "
                f"expected {expected!r}, got {result.get(key)!r}"
            )
    scene = result.get("scene")
    if (
        not isinstance(scene, dict)
        or scene.get("schema") != "real2sim.small_task29_full_scene.v4"
        or scene.get("episode_uid") != TASK29_FULL_EPISODE_UID
        or scene.get("task_id") != "29"
        or scene.get("source_cup_path") != TASK29_FULL_SOURCE_CUP_PATH
        or scene.get("source_cup_active") is not False
    ):
        raise ValueError("small Task 29 full scene identity is invalid")
    # 2026-08-02 moved the task-29 station to table_pick + the reviewed
    # bin-follow delta (task29_full_reproduce.ROBOT_BASE_XY); the initializer
    # and check_success both consume that constant, so the validator must
    # bind to the same source instead of the pre-delta shared station.
    import task29_full_reproduce as _task29_contract
    expected_station_xy = _task29_contract.ROBOT_BASE_XY
    expected_station_quat = _task29_contract.ROBOT_ORIENT_WXYZ
    try:
        robot_base = np.asarray(scene["robot_base_xy"], dtype=float)
        robot_orient = np.asarray(scene["robot_orient_wxyz"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "small Task 29 full scene robot pose is invalid") from exc
    if (
        robot_base.shape != (2,)
        or robot_orient.shape != (4,)
        or not np.allclose(
            robot_base,
            np.asarray(expected_station_xy, dtype=float),
            atol=1e-8,
            rtol=0.0,
        )
        or not np.allclose(
            robot_orient,
            np.asarray(expected_station_quat, dtype=float),
            atol=1e-7,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "small Task 29 robot is not at the release-aligned station")
    table = scene.get("coffee_table")
    try:
        table_size = np.asarray(table["world_size_xyz_m"], dtype=float)
        table_center = np.asarray(table["world_center_xy_m"], dtype=float)
        table_min = np.asarray(table["world_aabb_min_xyz_m"], dtype=float)
        table_max = np.asarray(table["world_aabb_max_xyz_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "small Task 29 current CoffeeTable descriptor is invalid") from exc
    expected_table_min = np.asarray((
        TASK29_COFFEE_TABLE_CENTER_XY_M[0]
        - COFFEE_TABLE_CURRENT_SIZE_XYZ_M[0] / 2.0,
        TASK29_COFFEE_TABLE_CENTER_XY_M[1]
        - COFFEE_TABLE_CURRENT_SIZE_XYZ_M[1] / 2.0,
        COFFEE_TABLE_CURRENT_BOTTOM_M,
    ))
    expected_table_max = expected_table_min + np.asarray(
        COFFEE_TABLE_CURRENT_SIZE_XYZ_M)
    if (
        not isinstance(table, dict)
        or table.get("prim_path") != "/World/Assets/CoffeeTable"
        or table.get("robot_collision_filtered") is not False
        or not isinstance(table.get("enabled_collision_paths"), list)
        or not table["enabled_collision_paths"]
        or table_size.shape != (3,)
        or table_center.shape != (2,)
        or table_min.shape != (3,)
        or table_max.shape != (3,)
        or not np.allclose(
            table_size,
            np.asarray(COFFEE_TABLE_CURRENT_SIZE_XYZ_M),
            atol=0.002,
            rtol=0.0,
        )
        or not np.allclose(
            table_center,
            np.asarray(TASK29_COFFEE_TABLE_CENTER_XY_M),
            atol=0.002,
            rtol=0.0,
        )
        or not np.allclose(
            table_min, expected_table_min, atol=0.002, rtol=0.0)
        or not np.allclose(
            table_max, expected_table_max, atol=0.002, rtol=0.0)
        or not np.allclose(
            np.asarray(
                table.get("task_local_offset_xy_m"), dtype=float),
            np.asarray(TASK29_COFFEE_TABLE_OFFSET_XY_M),
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "small Task 29 CoffeeTable is not the current collision-enabled "
            "0.8 x 1.2 x 0.45 m table")
    table_visual_policy = scene.get("coffee_table_visual_policy")
    if table_visual_policy != {
        "render_prim_path": "/World/Assets/CoffeeTable",
        "physical_prim_path": "/World/Assets/CoffeeTable",
        "retired_clone_path": TASK29_RETIRED_COFFEE_TABLE_VISUAL_PATH,
        "retired_clone_present": False,
        "render_visibility": "inherited",
    }:
        raise ValueError(
            "small Task 29 must render the unchanged physical CoffeeTable")
    paper_plate = scene.get("paper_plate")
    try:
        plate_translate = np.asarray(
            paper_plate["translate_xyz_m"], dtype=float)
        plate_min = np.asarray(
            paper_plate["world_aabb_min_xyz_m"], dtype=float)
        plate_max = np.asarray(
            paper_plate["world_aabb_max_xyz_m"], dtype=float)
        plate_size = np.asarray(
            paper_plate["world_size_xyz_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "small Task 29 PaperPlate descriptor is invalid") from exc
    if (
        not isinstance(paper_plate, dict)
        or paper_plate.get("role")
        != "visual_distractor_and_small_task30_support_asset"
        or paper_plate.get("prim_path") != TASK29_PAPER_PLATE_PATH
        or paper_plate.get("rigid_body_enabled") is not False
        or paper_plate.get("kinematic") is not False
        or paper_plate.get("collision_enabled") is not False
        or not np.allclose(
            plate_translate,
            np.asarray(TASK29_PAPER_PLATE_TRANSLATE_XYZ_M),
            atol=1e-8,
            rtol=0.0,
        )
        or not np.allclose(
            plate_size,
            np.asarray(TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M),
            atol=0.001,
            rtol=0.0,
        )
        or plate_min.shape != (3,)
        or plate_max.shape != (3,)
        or not math.isclose(
            float(plate_min[2]),
            TASK29_PAPER_PLATE_BOTTOM_M,
            abs_tol=0.001,
            rel_tol=0.0,
        )
        or not math.isclose(
            float(plate_max[2]),
            TASK29_PAPER_PLATE_TOP_M,
            abs_tol=0.001,
            rel_tol=0.0,
        )
    ):
        raise ValueError(
            "small Task 29 PaperPlate distractor/support is not canonical")
    _context = scene.get("non_task_render_context")
    # The initializer additionally reports which pinned context members it
    # excluded (their kinematic collider footprint intersects the cup
    # staging).  Validate the exclusion against the module constant instead
    # of rejecting the documented extra keys wholesale.
    if (
        not isinstance(_context, dict)
        or _context.get("prim_path")
        != "/World/Assets/Task56CurrentContext"
        or _context.get("active_for_accepted_physics_topology") is not True
        or _context.get("render_visibility") != "invisible"
        or tuple(_context.get("excluded_member_names", ()))
        != tuple(sorted(
            _task29_contract.TASK29_TASK56_CONTEXT_EXCLUDED_MEMBERS))
    ):
        raise ValueError(
            "small Task 29 inherited tabletop context is not render-hidden")
    contact_policy = scene.get("coffee_table_contact_policy")
    if contact_policy != {
        "robot_collision_enabled": True,
        "cup_collision_enabled": True,
        "collision_filter_group": None,
    }:
        raise ValueError(
            "small Task 29 robot/CoffeeTable collision policy is invalid")
    cups = scene.get("cups")
    if (
        not isinstance(cups, list)
        or len(cups) != 3
        or [cup.get("body_path") for cup in cups]
        != list(TASK29_FULL_CUP_ROOT_PATHS)
        or [cup.get("semantic_pose") for cup in cups]
        != [
            "upright_body_grasp",
            "upright_body_grasp",
            "lying_rim_grasp",
        ]
        or any(
            cup.get("dynamic") is not True
            or cup.get("ccd_enabled") is not True
            for cup in cups
        )
    ):
        raise ValueError(
            "small Task 29 full scene does not contain three canonical "
            "dynamic CCD cups")
    bins = scene.get("bins")
    if (
        not isinstance(bins, dict)
        or set(bins) != {"destination", "distractor"}
        or bins["destination"].get("body_path")
        != "/World/Assets/TrashBinTeal"
        or bins["distractor"].get("body_path")
        != "/World/Assets/TrashBinRed"
        or bins["destination"].get("collision_enabled") is not True
        or bins["distractor"].get("collision_enabled") is not True
        or float(scene.get(
            "bin_tablecloth_basket_clearance_m", {}
        ).get("destination", 0.0)) <= 0.0
        or float(scene.get(
            "bin_tablecloth_basket_clearance_m", {}
        ).get("distractor", 0.0)) <= 0.0
    ):
        raise ValueError(
            "small Task 29 full scene destination/distractor bins are invalid")
    return scene


def validate_trajectory_npz(path, expected_fps=FPS):
    """Fail closed before publishing an extracted trajectory to the shared runtime."""
    try:
        with np.load(path, allow_pickle=False) as trajectory:
            fps = np.asarray(trajectory["fps"])
            if fps.ndim != 0 or not math.isclose(
                    float(fps), expected_fps, abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"fps must be scalar {expected_fps:g}")
            arm = np.asarray(trajectory["arm"])
            if arm.ndim != 2 or arm.shape[0] < 2 or arm.shape[1] != 14:
                raise ValueError(f"arm has invalid shape {arm.shape}")
            frames = arm.shape[0]
            shapes = {
                "lgrip": (frames,),
                "rgrip": (frames,),
                "waist": (frames, 2),
                "head": (frames, 2),
                "base_xyyaw": (frames, 3),
            }
            arrays = [arm]
            for name, shape in shapes.items():
                value = np.asarray(trajectory[name])
                if value.shape != shape:
                    raise ValueError(
                        f"{name} has shape {value.shape}, expected {shape}")
                arrays.append(value)
            if not all(np.isfinite(value).all() for value in arrays):
                raise ValueError("trajectory contains non-finite values")
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid extracted trajectory: {exc}") from exc
    return frames


def copy_validated_trajectory(source, destination):
    """Copy one already-extracted action trajectory through the strict gate.

    Physics-only generation archives the final, post-filter trajectory.  A
    later render must consume those exact action bytes rather than re-solving
    the warp and hoping it produces the same file.  Validate both sides of the
    copy so a partial handoff can never reach the shared runtime path.
    """
    source = Path(source)
    destination = Path(destination)
    frames = validate_trajectory_npz(source)
    shutil.copyfile(source, destination)
    with destination.open("r+b") as stream:
        os.fsync(stream.fileno())
    copied_frames = validate_trajectory_npz(destination)
    if copied_frames != frames:
        raise ValueError(
            "trajectory frame count changed while copying: "
            f"{frames} -> {copied_frames}")
    return frames


def bind_playback_trajectory(episode_input, trajectory, *, source):
    """Add the generated action identity to rendered-frame provenance."""
    payload = json.loads(json.dumps(episode_input))
    trajectory = Path(trajectory)
    with np.load(trajectory, allow_pickle=False) as action:
        fps = float(np.asarray(action["fps"]).item())
    payload["playback_trajectory"] = {
        "source": str(source),
        "path": str(trajectory.resolve()),
        "bytes": trajectory.stat().st_size,
        "frames": validate_trajectory_npz(trajectory),
        "fps": fps,
    }
    return payload


def bag_place_hold_hand_from_trajectory(path):
    """Resolve the exactly one initially object-holding bag-place hand."""
    try:
        with np.load(path, allow_pickle=False) as trajectory:
            left = np.asarray(trajectory["lgrip"], dtype=float)
            right = np.asarray(trajectory["rgrip"], dtype=float)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            f"cannot resolve bag-place hold hand from trajectory: {exc}"
        ) from exc
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or left.size < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("bag-place grip trajectories are invalid")
    initial = {"l": float(left[0]), "r": float(right[0])}
    closed = [hand for hand, value in initial.items() if value >= 0.5]
    if len(closed) != 1:
        raise ValueError(
            "bag-place frame zero must identify exactly one closed gripper: "
            f"left={initial['l']:.3f}, right={initial['r']:.3f}"
        )
    hand = closed[0]
    selected = left if hand == "l" else right
    if float(np.min(selected)) >= 0.5:
        raise ValueError(
            f"bag-place selected {hand} gripper has no recorded release")
    return hand


# Compatibility name for historical tests and diagnostics.
task73_hold_hand_from_trajectory = bag_place_hold_hand_from_trajectory


def validate_place_init_result(result, coffee_table_calibration=None):
    """Bind a place initializer's output to the canonical task calibration.

    A structurally valid init JSON is not enough: without this check a stale or
    custom init can label another task's target path/zone with the current task
    id, and the renderer will faithfully publish that mismatched evidence.
    """
    if not isinstance(result, dict) or result.get("family") != "place":
        raise ValueError("place init result must declare family='place'")
    task = str(result.get("task", ""))
    config = REGISTRY.get(task)
    if not config or config.get("family") != "place":
        raise ValueError(f"unknown canonical place task {task!r}")
    expected_target = track_prim(task)
    if result.get("target") != config.get("prim"):
        raise ValueError(
            f"place init target mismatch for task {task}: "
            f"expected {config.get('prim')!r}, got {result.get('target')!r}")
    if result.get("track_prim") != expected_target:
        raise ValueError(
            f"place init track_prim mismatch for task {task}: "
            f"expected {expected_target!r}, got {result.get('track_prim')!r}")
    expected_destination = destination_for_task(
        task, coffee_table_calibration=coffee_table_calibration
    )
    try:
        serialized_destination = json.dumps(
            result.get("destination"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        serialized_expected_destination = json.dumps(
            expected_destination,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("place init returned an invalid destination") from exc
    if serialized_destination != serialized_expected_destination:
        raise ValueError(
            f"place init destination mismatch for task {task}: "
            f"expected {serialized_expected_destination}, "
            f"got {serialized_destination}")
    try:
        zone = np.asarray(result.get("zone"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("place init returned an invalid zone") from exc
    expected_zone = np.asarray(
        zone_for_task(
            task, coffee_table_calibration=coffee_table_calibration
        ),
        dtype=float,
    )
    if (zone.shape != (5,) or not np.isfinite(zone).all()
            or not np.allclose(zone, expected_zone, atol=1e-9, rtol=0)):
        raise ValueError(
            f"place init zone mismatch for task {task}: "
            f"expected {expected_zone.tolist()}, got {zone.tolist()}")
    return task, expected_target, [float(value) for value in zone]


def validate_place_grasp_controls(result):
    """Return renderer env for canonical, initializer-owned grasp controls.

    These values change which physical cross-section is placed between the
    pads.  Treating JSON strings, booleans or non-finite numbers as a numeric
    calibration would silently select a different grasp, so the handoff is
    deliberately stricter than a normal ``float(...)`` conversion.
    """
    if not isinstance(result, dict):
        raise ValueError("place init result must be a dictionary")
    environment = {}

    up_axis = result.get("up_axis")
    if up_axis is not None:
        if type(up_axis) is not int or up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be exactly one of integer 0, 1, 2")
        environment["UP_AXIS"] = str(up_axis)

    for source_key, environment_key, positive in (
        ("seat_long", "SEAT_LONG", False),
        ("seat_mid", "SEAT_MID", False),
        ("grip_width", "GRIP_WIDTH", True),
        ("release_retreat", "RELEASE_RETREAT", True),
    ):
        value = result.get(source_key)
        if value is None:
            continue
        if isinstance(value, bool) or type(value) not in (int, float):
            qualifier = "finite positive number" if positive else "finite number"
            raise ValueError(f"{source_key} must be a {qualifier}")
        numeric = float(value)
        if not math.isfinite(numeric) or (positive and numeric <= 0.0):
            qualifier = "finite positive number" if positive else "finite number"
            raise ValueError(f"{source_key} must be a {qualifier}")
        environment[environment_key] = str(numeric)

    plane_rotation = result.get("grasp_long_plane_rotation_deg")
    if plane_rotation is not None:
        if (
            isinstance(plane_rotation, bool)
            or type(plane_rotation) not in (int, float)
        ):
            raise ValueError(
                "grasp_long_plane_rotation_deg must be a finite number"
            )
        numeric = float(plane_rotation)
        if not math.isfinite(numeric) or not -90.0 <= numeric <= 90.0:
            raise ValueError(
                "grasp_long_plane_rotation_deg must be within [-90, 90]"
            )
        environment["GRASP_LONG_PLANE_ROTATION_DEG"] = str(numeric)

    approach_offset = result.get("approach_offset_xyz")
    if approach_offset is not None:
        if (
            not isinstance(approach_offset, (list, tuple))
            or len(approach_offset) != 3
            or any(
                isinstance(value, bool)
                or type(value) not in (int, float)
                for value in approach_offset
            )
        ):
            raise ValueError(
                "approach_offset_xyz must contain three finite numbers"
            )
        vector = np.asarray(approach_offset, dtype=float)
        if (
            vector.shape != (3,)
            or not np.isfinite(vector).all()
            or float(np.linalg.norm(vector))
            > BAG_PLACE_MAX_APPROACH_OFFSET_M
        ):
            raise ValueError(
                "approach_offset_xyz must be a finite 3-vector with "
                "magnitude at most "
                f"{BAG_PLACE_MAX_APPROACH_OFFSET_M:.3f} m"
            )
        environment["PLACE_APPROACH_OFFSET_XYZ"] = ",".join(
            str(float(value)) for value in vector
        )
    return environment


def validate_container_mount_records(records, expected=None):
    """Prove the running container sees the expected two host trees.

    The checkout intentionally has sibling ``genie_sim_demo`` (container main)
    and ``genie_sim`` (canonical robot assets) directories.  Accepting merely
    any mount at the expected container path can identify one tree while
    Isaac renders another.
    """
    if expected is None:
        expected = {
            CONTAINER_MAIN: Path(DEMO).parent,
            CONTAINER_ASSETS: ROBOT_ASSET_ROOT.parents[1],
        }
    if not isinstance(records, list):
        raise ValueError("Docker mount metadata must be a list")
    by_destination = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("Destination"), str):
            destination = record["Destination"]
            if destination in by_destination:
                raise ValueError(
                    f"container mount {destination} is declared more than once")
            by_destination[destination] = record
    for destination, wanted_source in expected.items():
        record = by_destination.get(destination)
        if record is None:
            raise ValueError(f"container mount {destination} is missing")
        source = record.get("Source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"container mount {destination} has no host source")
        try:
            actual = Path(source).resolve(strict=True)
            wanted = Path(wanted_source).resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"container mount {destination} source cannot be resolved: {exc}") from exc
        if actual != wanted:
            raise ValueError(
                f"container mount {destination} uses {actual}, expected {wanted}")
        # The runtime writes trajectories, frames and sidecars below main.
        # Robot USD layers are input-only and may intentionally be mounted RO.
        if destination == CONTAINER_MAIN and record.get("RW") is not True:
            raise ValueError(f"container mount {destination} is not read-write")


def verify_container_runtime_mounts():
    """Read Docker's mount table and validate the canonical demo/assets split."""
    result = run(
        ["docker", "inspect", "-f", "{{json .Mounts}}", CONTAINER_NAME],
        timeout=20,
    )
    if result.returncode != 0:
        raise SystemExit(
            "cannot inspect Isaac container mounts: "
            + (result.stdout + result.stderr)[-400:])
    try:
        records = json.loads(result.stdout)
        validate_container_mount_records(records)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid Isaac container mounts: {exc}") from exc


def stop_stale_container_playback():
    """Fence an orphaned container renderer before touching shared host state.

    Killing a host ``docker exec`` client does not guarantee that its container
    process exits.  A prior parent may have been SIGKILLed after losing the host
    lock while Isaac kept mutating the shared scene/frames.  Every new episode
    therefore fences the two narrowly-scoped managed renderer processes before
    trajectory extraction or scene initialization, and once more immediately
    before launching its own renderer.
    """
    command = [
        "docker", "exec", CONTAINER_NAME, "bash", "-lc",
        "pkill -TERM -f "
        "'[/]geniesim/main/demo_sim/usdz/(playback_video3|align_grasp).py' "
        "2>/dev/null || true; sleep 2; "
        "pkill -KILL -f "
        "'[/]geniesim/main/demo_sim/usdz/(playback_video3|align_grasp).py' "
        "2>/dev/null || true",
    ]
    result = run(command, timeout=30)
    if result.returncode != 0:
        raise SystemExit(
            "cannot fence stale managed Isaac playback: "
            + (result.stdout + result.stderr)[-400:])


def clear_container_runtime_outputs():
    """Remove fixed stale markers as the container user owning video3/.

    Isaac writes this bind-mounted directory as the container UID, so the host
    runner can read sidecars but cannot unlink them from a 0755 directory.
    Use fixed argv-only targets instead of weakening permissions on the tree.
    """
    runtime_dir = "/geniesim/main/demo_sim/cam_preview/video3"
    command = [
        "docker", "exec", CONTAINER_NAME, "rm", "-f",
        *[
            f"{runtime_dir}/{name}"
            for name in (
                "distractor_audit.json",
                "pick_contact_trace.npz",
                "place_metrics.npz",
                "place_metrics.json",
                "task29_full_metrics.npz",
                "task29_full_metrics.json",
                "video3.txt",
                "asset_policy.json",
            )
        ],
    ]
    result = run(command, timeout=30)
    if result.returncode != 0:
        raise SystemExit(
            "cannot clear stale container runtime outputs: "
            + (result.stdout + result.stderr)[-400:])


def fsync_directories(*directories):
    """Durably publish rename/unlink state before a completion marker is written."""
    seen = set()
    for directory in directories:
        path = Path(directory)
        try:
            identity = (path.stat().st_dev, path.stat().st_ino)
        except OSError:
            continue
        if identity in seen:
            continue
        seen.add(identity)
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def file_size(path):
    """Return lightweight generated-artifact metadata for manifests."""
    return Path(path).stat().st_size


def sync_file(source, dest, label):
    if not source.is_file():
        raise SystemExit(f"missing canonical {label}: {source}")
    if dest.is_file() and filecmp.cmp(source, dest, shallow=False):
        if dest.stat().st_mode & 0o777 != 0o644:
            dest.chmod(0o644)
            with dest.open("rb") as stream:
                os.fsync(stream.fileno())
            fsync_directories(dest.parent)
        print(f"runtime: {label} already current", flush=True)
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, tmp)
        tmp.chmod(0o644)
        if not filecmp.cmp(source, tmp, shallow=False):
            raise SystemExit(f"{label} temporary copy differs from source")
        with tmp.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, dest)
        fsync_directories(dest.parent)
    finally:
        tmp.unlink(missing_ok=True)
    if not filecmp.cmp(source, dest, shallow=False):
        raise SystemExit(f"{label} deployed copy differs from source")
    print(f"runtime: synced canonical {label}", flush=True)
    return str(dest)


def sync_runtime(task_id=None):
    """Make reviewed repo copies authoritative for files consumed by the Docker render."""
    synced = (
        (RUNTIME_SOURCE, RUNTIME_DEST, "playback_video3.py"),
        (PD_RUNTIME_SOURCE, PD_RUNTIME_DEST, "playback_pd.py"),
        (
            ROBOT_COLLISION_PROFILE_SOURCE,
            ROBOT_COLLISION_PROFILE_DEST,
            "robot_collision_profile.py",
        ),
        *((Path(REPO) / "real2sim/teleop" / name,
           Path(DEMO) / "usdz" / name, name)
          for name in ("robot_gripper_authority.py", "shelf_support_geometry.py", "shelf_geometry.py", "physical_distractors.py", "contact_step_aggregation.py", "coffee_table_partition.py", "episode_distractor_policy_v1.json")),
        (GEOMETRY_SOURCE, GEOMETRY_DEST, "place_collision_geometry.py"),
        (ALIGN_SOURCE, ALIGN_DEST, "align_grasp.py"),
        (PAD_GEOMETRY_SOURCE, PAD_GEOMETRY_DEST, "pad_geometry.py"),
        (
            PICK_PROVENANCE_SOURCE,
            PICK_PROVENANCE_DEST,
            "pick_provenance.py",
        ),
        (
            TASK53_CALIBRATION_SOURCE,
            TASK53_CALIBRATION_DEST,
            TASK53_CONTACT_SCENE_CALIBRATION_FILENAME,
        ),
        (
            CAMERA_CALIBRATION_SOURCE,
            CAMERA_CALIBRATION_DEST,
            "camera_pose_calibration.py",
        ),
        (
            ROBOT_POSITIONS_SOURCE,
            ROBOT_POSITIONS_DEST,
            "robot_positions.py",
        ),
        (
            GROUP_SHIFT_SOURCE,
            GROUP_SHIFT_DEST,
            "group_shift.py",
        ),
        (
            BAG_PLACE_CONTRACT_SOURCE,
            BAG_PLACE_CONTRACT_DEST,
            "bag_place_contract.py",
        ),
        (
            TASK6_DRAWER_CONTRACT_SOURCE,
            TASK6_DRAWER_CONTRACT_DEST,
            "task6_drawer_sequence_contract.py",
        ),
        (
            TASK6_HAT_CALIBRATION_SOURCE,
            TASK6_HAT_CALIBRATION_DEST,
            "calibrate_task6_hat_30cm_action.py",
        ),
        (
            TASK29_CONTRACT_SOURCE,
            TASK29_CONTRACT_DEST,
            "task29_full_reproduce.py",
        ),
        (
            TASK30_CONTRACT_SOURCE,
            TASK30_CONTRACT_DEST,
            "task30_throw_reproduce.py",
        ),
        (
            TASK69_CONTRACT_SOURCE,
            TASK69_CONTRACT_DEST,
            "task69_clear_tablecloth_reproduce.py",
        ),
        (
            DUAL_BIN_LAYOUT_SOURCE,
            DUAL_BIN_LAYOUT_DEST,
            "dual_trash_bin_layout.py",
        ),
        (
            TABLECLOTH_BASKET_CONTRACT_SOURCE,
            TABLECLOTH_BASKET_CONTRACT_DEST,
            "tablecloth_basket_contract.py",
        ),
        # playback_video3 imports shelf_asset_policy inside the container, so
        # deploy the policy's two small Python dependencies beside it.
        (SHELF_POLICY_SOURCE, SHELF_POLICY_DEST, "shelf_asset_policy.py"),
        (
            GIT_PROVENANCE_SOURCE,
            GIT_PROVENANCE_DEST,
            "git_provenance.py",
        ),
        (Path(REPO) / "real2sim" / "teleop" / "machine_paths.py",
         Path(DEMO) / "usdz" / "machine_paths.py", "machine_paths.py"),
        (
            SHELF_ENFORCER_SOURCE,
            SHELF_ENFORCER_DEST,
            "enforce_shelf_asset_policy.py",
        ),
        (
            SHELF_RELEASE_RECEIPT_SOURCE,
            SHELF_RELEASE_RECEIPT_DEST,
            "shelf_asset_release_receipt.json",
        ),
    ) + KITCHEN_RANDOM_RUNTIME_FILES
    if str(task_id) == "30":
        synced += TASK30_PAPER_ASSET_RUNTIME_FILES
    for source, destination, label in synced:
        sync_file(source, destination, label)
    texture_files = task_texture_runtime_files(task_id)
    for source, destination, label in texture_files:
        sync_file(source, destination, label)
    runtime_revision = runtime_source_revision(task_id)
    for source, destination, label in synced:
        if not filecmp.cmp(source, destination, shallow=False):
            raise SystemExit(f"{label} changed during runtime deployment")
    return runtime_revision


def enforce_runtime_shelf_policy():
    """Deactivate every document-external/unverified shelf child in Isaac USD."""
    child_env = os.environ.copy()
    child_env["SCENE_USDA"] = str(SCENE_PATH)
    result = run(
        [
            str(MESH311_PYTHON),
            str(SHELF_ENFORCER_SOURCE),
        ],
        timeout=180,
        cwd=REPO,
        env=child_env,
    )
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "shelf policy enforcer returned no valid JSON:\n"
            + (result.stdout + result.stderr)[-1000:]
        ) from exc
    if result.returncode != 0 or not isinstance(report, dict) or (
            report.get("ok") is not True):
        raise SystemExit(f"shelf policy enforcer failed: {report}")
    return report


def require_recovered_scene_audit(
    initializer_result, *, episode_task, episode_uid,
):
    """Measure an optional recovered benchmark plan before physics starts."""
    raw_plan = os.environ.get("REAL2SIM_RECOVERED_PLAN", "").strip()
    if not raw_plan:
        return None
    plan = Path(raw_plan)
    if not plan.is_absolute() or not plan.is_file():
        raise SystemExit(
            "REAL2SIM_RECOVERED_PLAN must name an existing absolute plan"
        )
    audit = plan.with_name("randomization_audit.json")
    receipt = plan.with_name("initializer_receipt.json")
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    receipt_payload = {
        "schema": "real2sim.recovered_initializer_receipt.v1",
        "benchmark_subtask": str(plan_payload.get("benchmark_subtask", "")),
        "rollout_index": int(plan_payload.get("rollout_index", -1)),
        "initializer_plan_id": str(plan_payload.get("plan_id", "")),
        "source_episode_uid": str(episode_uid),
        "initializer_result": {
            "ok": initializer_result.get("ok") is True,
            "small_task_id": str(episode_task),
            "family": str(initializer_result.get("family", "")),
            "track_prim": str(initializer_result.get("track_prim", "")),
            "base_xy_m": initializer_result.get("base"),
            "yaw_deg": initializer_result.get("yaw"),
            "hand": str(initializer_result.get("hand", "")),
        },
    }
    temporary = receipt.with_name(
        f".{receipt.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(receipt_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt)
    command = [
        str(MESH311_PYTHON),
        str(
            Path(REPO) / "real2sim" / "production"
            / "recovered_scene_materialization.py"
        ),
        "--plan", str(plan), "--receipt", str(receipt),
        "--audit", str(audit),
    ]
    result = run(command, timeout=300, cwd=REPO)
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "recovered scene audit returned no valid JSON:\n"
            + (result.stdout + result.stderr)[-2000:]
        ) from exc
    if result.returncode != 0 or report.get("ok") is not True:
        raise SystemExit(
            "recovered scene materialization audit failed: "
            + json.dumps(report, ensure_ascii=False)
        )
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from real2sim.production.recovered_task_contract import (
        require_materialized_rollout,
    )
    authorization = require_materialized_rollout(
        str(plan_payload.get("benchmark_subtask", "")),
        initializer_plan=plan,
        randomization_audit=audit,
    )
    if authorization.get("decision") != "run_strict_rollout":
        raise SystemExit(
            "recovered scene audit did not authorize strict rollout: "
            + "; ".join(map(str, authorization.get("blockers", ())))
        )
    print("recovered-scene-audit:", report, flush=True)
    return report


def apply_episode_scene_randomization(al, *, task_id, episode_uid, epdir):
    """Apply the current data-grounded clutter policy to the initialized scene."""
    if not randomization_enabled(task_id):
        return al
    hand_audit = infer_grasp_hand_from_hdf5(
        Path(epdir) / "record" / "raw_joints.h5"
    )
    resolved_hand, hand_resolution = reconcile_hand(
        al.get("hand", "?"), hand_audit
    )
    target_prim_path = runtime_target_prim(task_id, al)
    protected = set(contact_prims(task_id, episode_uid))
    if al.get("task53_support_mode"):
        protected.add("/World/Assets/CushionOnBook")
    randomized = run_randomization_subprocess(
        scene_path=SCENE_PATH,
        task_id=task_id,
        episode_uid=episode_uid,
        # Generation layer (datagen): a gen_seed-derived override makes the
        # clutter identity a function of the generation seed instead of the
        # episode UUID, so N generated episodes from one seed recording get N
        # distinct, reproducible layouts. Unset = the historical
        # episode-UUID-driven identity, byte-identical behaviour.
        seed_override=os.environ.get("SCENE_RANDOMIZATION_SEED"),
        target_prim_path=target_prim_path,
        hand=resolved_hand,
        place_zone=al.get("zone"),
        protected_prim_paths=protected,
        protected_objects=context_object_descriptors(
            al.get("current_secondary_context")
        ),
    )
    updated = dict(al)
    updated["hand"] = resolved_hand
    updated["hand_audit"] = hand_audit
    updated["hand_resolution"] = hand_resolution
    updated["scene_randomization"] = randomized
    updated["distract_prims"] = randomized["applied_prim_paths"]
    return updated


def randomized_distractor_audit_prims(al, task_id):
    """Track distractors only when this run actually randomized them.

    Initializers also list canonical neighbouring objects in
    ``distract_prims``.  A trajectory-core run with scene randomization
    disabled has no applied plan, so treating those neighbours as randomized
    clutter creates an audit/plan contradiction and can consume a stale plan
    from an earlier episode.
    """
    if not randomization_enabled(task_id):
        return []
    return sorted(str(path) for path in (al.get("distract_prims") or ()))


def kitchen_tableware_randomization_enabled(environ=None):
    """Follow overall appearance opt-in unless explicitly overridden."""
    effective = os.environ if environ is None else environ
    explicit = effective.get(KITCHEN_TABLEWARE_ENABLE_ENV)
    if explicit is not None:
        return explicit == "1"
    return appearance_randomization_enabled(effective)


def apply_episode_kitchen_tableware_randomization(
    al, *, task_id, episode_uid
):
    """Add one physical, collision-audited bowl/chopstick layout per scene."""
    if not kitchen_tableware_randomization_enabled():
        return al
    if str(task_id) in BAG_PLACE_TASK_ID_SET:
        return al
    randomized = run_tableware_subprocess(
        scene_path=SCENE_PATH,
        task_id=task_id,
        episode_uid=episode_uid,
        seed_override=os.environ.get("SCENE_RANDOMIZATION_SEED"),
        asset_root=KITCHEN_TABLEWARE_ASSET_ROOT,
    )
    updated = dict(al)
    updated["kitchen_tableware_randomization"] = {
        "policy_version": randomized.get("policy_version"),
        "seed_u64": randomized.get("seed_u64"),
        "support_id": randomized.get("support_id"),
        "support_prim_path": randomized.get("support_prim_path"),
        "templates": [
            cluster.get("template")
            for cluster in randomized.get("clusters", ())
        ],
        "object_count": randomized.get("collision_audit", {}).get(
            "object_count"
        ),
        "collision_free": randomized.get("collision_audit", {}).get(
            "collision_free"
        ),
        "cups_present": randomized.get("cups_present"),
        "physics_effect": randomized.get("physics_effect"),
    }
    return updated


def apply_episode_appearance_randomization(al, *, task_id, episode_uid):
    """Randomize lights + non-semantic material tints (opt-in, identity off).

    Runs AFTER scene randomization on the same working scene; authors only
    shader inputs and light attributes (never transforms), so it composes
    with clutter, group shift, and the session/baseline contract.  Gated on
    REAL2SIM_APPEARANCE_RANDOMIZATION=1: unset, the render is bit-identical
    to the historical pipeline.
    """
    if not appearance_randomization_enabled():
        return al
    if str(task_id) in BAG_PLACE_TASK_ID_SET:
        # Bag-place renders run on their own scene file; the v1 appearance
        # policy authors only the canonical room scene.
        return al
    randomized = run_appearance_subprocess(
        scene_path=SCENE_PATH,
        task_id=task_id,
        episode_uid=episode_uid,
        # Same gen_seed-derived override lane as the clutter identity, so one
        # generation seed pins layout AND look together.
        seed_override=os.environ.get("SCENE_RANDOMIZATION_SEED"),
    )
    updated = dict(al)
    kitchen_plan = randomized.get("kitchen_materials", {})
    updated["appearance_randomization"] = {
        key: randomized.get(key)
        for key in ("policy_version", "seed_u64", "episode_token")
    }
    updated["appearance_randomization"]["kitchen_materials"] = {
        "central_table_option_id": kitchen_plan.get(
            "central_table", {}).get("option_id"),
        "stove_rule_id": kitchen_plan.get("stove", {}).get("rule_id"),
        "countertop_material_id": kitchen_plan.get(
            "stove", {}).get("countertop_material_id"),
        "lower_material_id": kitchen_plan.get(
            "stove", {}).get("lower_material_id"),
        "upper_material_id": kitchen_plan.get(
            "stove", {}).get("upper_material_id"),
    }
    return updated


def _content_inventory(paths, root):
    """Describe one bounded episode input set by explicit paths and sizes."""
    total_bytes = 0
    files = sorted(Path(path) for path in paths)
    entries = []
    for path in files:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"episode input escapes root {root}: {path}") from exc
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size})
        total_bytes += size
    return {
        "bytes": total_bytes,
        "files": len(files),
        "entries": entries,
    }


def episode_input_signature(episode_dir):
    """Bind proprioception and each of the three primary REAL cameras."""
    root = Path(episode_dir)
    raw = root / "record" / "raw_joints.h5"
    if not raw.is_file():
        raise ValueError(f"missing episode raw_joints.h5: {raw}")
    cameras = {}
    for camera in REAL_CAMS:
        camera_root = root / "camera" / camera
        if not camera_root.is_dir():
            raise ValueError(
                f"missing primary REAL camera directory: {camera_root}")
        files = [
            path for path in camera_root.rglob("*")
            if path.is_file()
        ]
        if not files:
            raise ValueError(
                f"primary REAL camera has no files: {camera_root}")
        cameras[camera] = _content_inventory(files, root)
    payload = {
        "schema": EPISODE_INPUT_SCHEMA,
        "episode_uid": root.name,
        "raw_joints": _content_inventory([raw], root),
        "real_cameras": cameras,
    }
    return payload


def write_frame_policy_manifest(
    *,
    episode_uid,
    task_id,
    decision,
    runtime_revision,
    frame_count,
    episode_input,
):
    """Bind shared rendered frames to SIM runtime and exact REAL inputs."""
    valid_real_input = (
        isinstance(episode_input, dict)
        and episode_input.get("schema") == EPISODE_INPUT_SCHEMA
        and episode_input.get("episode_uid") == str(episode_uid)
    )
    trajectory = (
        episode_input.get("playback_trajectory", {})
        if isinstance(episode_input, dict) else {}
    )
    valid_recordless_input = (
        isinstance(episode_input, dict)
        and episode_input.get("schema") == RECORDLESS_INPUT_SCHEMA
        and episode_input.get("episode_uid") == str(episode_uid)
        and episode_input.get("task_id") == str(task_id)
        and episode_input.get("real_episode_archive_required") is False
        and isinstance(trajectory, dict)
        and int(trajectory.get("frames", -1)) == int(frame_count)
        and math.isclose(
            float(trajectory.get("fps", float("nan"))),
            FPS,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and int(trajectory.get("bytes", 0)) > 0
        and bool(str(trajectory.get("source", "")).strip())
    )
    if not (valid_real_input or valid_recordless_input):
        raise ValueError("invalid episode input signature")
    payload = {
        "schema": FRAME_PROVENANCE_SCHEMA,
        "episode_uid": str(episode_uid),
        "task_id": str(task_id),
        "asset_id": decision.asset_id,
        "asset_status": decision.status,
        "asset_revision": decision.asset_revision,
        "runtime_source_revision": runtime_revision,
        "scene_randomization": randomization_identity(
            str(task_id), str(episode_uid)
        ),
        "frame_count": int(frame_count),
        "episode_input": episode_input,
    }
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    encoded = base64.b64encode(serialized).decode("ascii")
    # video3/ is owned by the container UID on the bind mount.  Publish through
    # that owner with a fixed script and an injection-safe base64 env value.
    result = run(
        [
            "docker",
            "exec",
            "--env",
            f"REAL2SIM_FRAME_MANIFEST_B64={encoded}",
            CONTAINER_NAME,
            "bash",
            "-lc",
            (
                "set -e; "
                "d=/geniesim/main/demo_sim/cam_preview/video3; "
                "t=\"$d/.asset_policy.$$.tmp\"; "
                "trap 'rm -f \"$t\"' EXIT; "
                "umask 022; "
                "printf '%s' \"$REAL2SIM_FRAME_MANIFEST_B64\" "
                "| base64 -d > \"$t\"; "
                "sync -f \"$t\"; "
                "mv -f \"$t\" \"$d/asset_policy.json\"; "
                "sync -f \"$d\"; "
                "trap - EXIT"
            ),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        raise SystemExit(
            "cannot publish frame asset-policy manifest as container owner: "
            + (result.stdout + result.stderr)[-500:]
        )
    try:
        observed = json.loads(FRAME_POLICY_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"published frame asset-policy manifest is unreadable: {exc}"
        ) from exc
    if observed != payload:
        raise SystemExit("published frame asset-policy manifest changed in transit")
    return payload


def validate_frame_policy_manifest(
    *,
    episode_uid,
    task_id,
    decision,
    runtime_revision,
    episode_input=None,
):
    """Validate provenance before ``--skip-render`` may reuse shared frames."""
    path = FRAME_POLICY_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"missing or invalid frame asset-policy manifest {path}: {exc}"
        ) from exc
    if episode_input is None:
        episode_input = episode_input_signature(
            resolve_episode_dir(episode_uid))
    expected = {
        "schema": FRAME_PROVENANCE_SCHEMA,
        "episode_uid": str(episode_uid),
        "task_id": str(task_id),
        "asset_id": decision.asset_id,
        "asset_status": decision.status,
        "asset_revision": decision.asset_revision,
        "runtime_source_revision": runtime_revision,
        "scene_randomization": randomization_identity(
            str(task_id), str(episode_uid)
        ),
        "frame_count": completed_frame_count(),
        "episode_input": episode_input,
    }
    if payload != expected:
        raise ValueError(
            "shared frames do not match the requested episode/runtime/asset "
            f"policy: expected={expected}, observed={payload}"
        )
    return payload


def output_asset_manifest_path(out):
    output = Path(out)
    return output.with_name(output.stem + ".asset_policy.json")


def publish_output_asset_manifest(out, frame_manifest):
    """Bind a composed MP4 to the verified shared-frame asset provenance."""
    output = Path(out)
    payload = {
        "schema": OUTPUT_PROVENANCE_SCHEMA,
        "frame_manifest": frame_manifest,
        "output_path": str(output.resolve()),
        "output_size": output.stat().st_size,
    }
    destination = output_asset_manifest_path(output)
    temporary = destination.with_name(
        destination.name + f".{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, sort_keys=True, separators=(",", ":")
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directories(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def publish_sim_only_manifest(
    out,
    *,
    frame_manifest,
    episode_uid,
    task_id,
    run_id,
    frame_count,
):
    """Publish an explicit provenance boundary for a SIM-only review video.

    A benchmark task must remain runnable when its optional mother/reference
    episode archive is absent.  This sidecar prevents a SIM-only diagnostic
    from being mistaken for a REAL-vs-SIM comparison while binding the video
    to the exact live rollout and validated trajectory identity carried by the
    normal frame manifest.
    """
    output = Path(out)
    payload = {
        "schema": SIM_ONLY_PROVENANCE_SCHEMA,
        "episode_uid": str(episode_uid),
        "task_id": str(task_id),
        "run_id": str(run_id),
        "frame_count": int(frame_count),
        "fps": FPS,
        "cameras": list(SIM_DIRS),
        "contains_real_frames": False,
        "physical_evidence_same_run": True,
        "frame_manifest": frame_manifest,
        "output_path": str(output.resolve()),
        "output_size": output.stat().st_size,
    }
    destination = output.with_name(
        output.stem + ".sim_only_video.json"
    )
    temporary = destination.with_name(
        destination.name + f".{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directories(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def validate_output_asset_manifest(
    out,
    *,
    episode_uid,
    task_id,
    decision,
    runtime_revision,
    episode_input=None,
):
    """Require exact output bytes and current runtime/asset identity for reuse."""
    output = Path(out)
    manifest_path = output_asset_manifest_path(output)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing/invalid output asset manifest: {exc}") from exc
    if not output.is_file():
        raise ValueError(f"comparison output is missing: {output}")
    if episode_input is None:
        episode_input = episode_input_signature(
            resolve_episode_dir(episode_uid))
    expected_frame = {
        "schema": FRAME_PROVENANCE_SCHEMA,
        "episode_uid": str(episode_uid),
        "task_id": str(task_id),
        "asset_id": decision.asset_id,
        "asset_status": decision.status,
        "asset_revision": decision.asset_revision,
        "runtime_source_revision": runtime_revision,
        "scene_randomization": randomization_identity(
            str(task_id), str(episode_uid)
        ),
        "episode_input": episode_input,
    }
    frame = payload.get("frame_manifest")
    if not isinstance(frame, dict):
        raise ValueError("output asset manifest has no frame manifest")
    for key, value in expected_frame.items():
        if frame.get(key) != value:
            raise ValueError(
                f"output asset manifest {key} mismatch: "
                f"expected={value!r}, observed={frame.get(key)!r}"
            )
    expected = {
        "schema": OUTPUT_PROVENANCE_SCHEMA,
        "frame_manifest": frame,
        "output_path": str(output.resolve()),
        "output_size": output.stat().st_size,
    }
    if payload != expected:
        raise ValueError("output bytes do not match the asset manifest")
    return payload


def validate_output_episode_input(out, episode_uid):
    """Validate output bytes and exact raw/three-camera REAL provenance."""
    output = Path(out)
    path = output_asset_manifest_path(output)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"missing/invalid output provenance manifest: {exc}") from exc
    if payload.get("schema") != OUTPUT_PROVENANCE_SCHEMA:
        raise ValueError("output provenance schema is stale")
    frame = payload.get("frame_manifest")
    if (
        not isinstance(frame, dict)
        or frame.get("schema") != FRAME_PROVENANCE_SCHEMA
        or frame.get("episode_uid") != str(episode_uid)
    ):
        raise ValueError("output frame provenance identity mismatch")
    expected_input = episode_input_signature(
        resolve_episode_dir(episode_uid))
    if frame.get("episode_input") != expected_input:
        raise ValueError("output REAL episode inputs changed")
    if not output.is_file():
        raise ValueError(f"comparison output is missing: {output}")
    if payload.get("output_size") != output.stat().st_size:
        raise ValueError("output size does not match provenance manifest")
    if Path(str(payload.get("output_path", ""))).resolve() != output.resolve():
        raise ValueError("output path does not match provenance manifest")
    return payload


def runtime_source_revision(task_id=None):
    """Git revision and explicit files that construct the runtime scene."""
    sources = [
        (RUNTIME_SOURCE, "playback_video3.py"),
        (PD_RUNTIME_SOURCE, "playback_pd.py"),
        (ROBOT_COLLISION_PROFILE_SOURCE, "robot_collision_profile.py"),
        *((Path(REPO) / "real2sim/teleop" / name, name)
          for name in ("robot_gripper_authority.py", "shelf_support_geometry.py", "shelf_geometry.py", "physical_distractors.py", "contact_step_aggregation.py", "coffee_table_partition.py", "episode_distractor_policy_v1.json")),
        (GEOMETRY_SOURCE, "place_collision_geometry.py"),
        (ALIGN_SOURCE, "align_grasp.py"),
        (PAD_GEOMETRY_SOURCE, "pad_geometry.py"),
        (PICK_PROVENANCE_SOURCE, "pick_provenance.py"),
        (
            TASK53_CALIBRATION_SOURCE,
            TASK53_CONTACT_SCENE_CALIBRATION_FILENAME,
        ),
        (CAMERA_CALIBRATION_SOURCE, "camera_pose_calibration.py"),
        (
            TASK6_DRAWER_CONTRACT_SOURCE,
            "task6_drawer_sequence_contract.py",
        ),
        (
            TASK69_CONTRACT_SOURCE,
            "task69_clear_tablecloth_reproduce.py",
        ),
    ]
    sources.extend(
        (source, label)
        for source, _destination, label
        in task_texture_runtime_files(task_id)
    )
    sources.extend(ROBOT_RUNTIME_LAYERS)
    sources.extend(SCENE_RUNTIME_LAYERS)
    sources.append((SHELF_RELEASE_RECEIPT_SOURCE, "shelf_release_receipt"))
    if str(task_id) in BAG_PLACE_TASK_ID_SET:
        sources.extend(BAG_PLACE_RUNTIME_LAYERS)
    if str(task_id) == "30":
        sources.extend(
            (source, label)
            for source, _destination, label
            in TASK30_PAPER_ASSET_RUNTIME_FILES
        )
    sources.extend(
        (source, label)
        for source, _destination, label in KITCHEN_RANDOM_RUNTIME_FILES
    )
    sources.extend(
        (source, f"host/{source.name}") for source in MANAGED_HOST_SOURCES)
    tracked = []
    inventory = []
    for source, label in sources:
        if not source.is_file():
            raise ValueError(f"missing canonical runtime source: {source}")
        resolved = source.resolve()
        inventory.append({
            "label": label,
            "path": str(resolved),
            "bytes": source.stat().st_size,
        })
        try:
            relative = resolved.relative_to(Path(REPO).resolve())
        except ValueError:
            continue
        tracked_check = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=REPO,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked_check.returncode == 0:
            tracked.append(relative)
    return {
        "schema": "real2sim.runtime_source.v1",
        "source_revision": source_revision(tracked),
        "files": inventory,
    }


def require_runtime_source_unchanged(expected_revision, task_id, stage):
    """Require the same Git/runtime file identity during one render."""
    observed = runtime_source_revision(task_id)
    if observed != expected_revision:
        raise SystemExit(
            "canonical runtime changed during comparison "
            f"({stage}): expected={expected_revision}, observed={observed}"
        )
    return observed


def has_instance_assignment(text):
    """The comparison runtime is intentionally canonical, not INSTANCE-aware."""
    try:
        tokens = shlex.split(text or "")
    except ValueError:
        return True
    return any(token.partition("=")[0] == "INSTANCE" for token in tokens)


def copy_place_sidecars(
    out,
    *,
    expected_run_id=None,
    expected_task_id=None,
    expected_episode_uid=None,
    expected_target_body_path=None,
    expected_destination_kind=None,
    expected_destination_body_path=None,
    expected_destination_bottom_z=None,
    expected_zone=None,
):
    """Validate one NPZ/JSON pair, then publish each destination atomically."""
    sources = {
        name: Path(DEMO) / "cam_preview" / "video3" / name
        for name in ("place_metrics.npz", "place_metrics.json")
    }
    if not all(path.is_file() for path in sources.values()):
        return []
    try:
        metadata = validate_place_sidecar_pair(
            sources["place_metrics.npz"],
            sources["place_metrics.json"],
            expected_run_id=expected_run_id,
            expected_task_id=expected_task_id,
            expected_episode_uid=expected_episode_uid,
            expected_target_body_path=expected_target_body_path,
            expected_destination_kind=expected_destination_kind,
            expected_destination_body_path=expected_destination_body_path,
            expected_destination_bottom_z=expected_destination_bottom_z,
            expected_zone=expected_zone,
        )
    except Exception as exc:
        raise SystemExit(f"invalid place sidecar identity: {type(exc).__name__}: {exc}")
    npz_run_id = metadata["identity"]["run_id"]

    copied = []
    pending = []
    try:
        for name, src in sources.items():
            dst = out.with_name(out.stem + "." + name)
            tmp = dst.with_name(
                dst.name + f".{npz_run_id}.{uuid.uuid4().hex}.tmp")
            shutil.copy2(src, tmp)
            with tmp.open("r+b") as stream:
                os.fsync(stream.fileno())
            pending.append((tmp, dst))
        for tmp, dst in pending:
            os.replace(tmp, dst)
            print(f"wrote {dst}", flush=True)
            copied.append(dst)
    finally:
        if copied:
            fsync_directories(out.parent)
        for tmp, _dst in pending:
            tmp.unlink(missing_ok=True)
    try:
        validate_place_sidecar_pair(
            out.with_name(out.stem + ".place_metrics.npz"),
            out.with_name(out.stem + ".place_metrics.json"),
            expected_run_id=expected_run_id,
            expected_task_id=expected_task_id,
            expected_episode_uid=expected_episode_uid,
            expected_target_body_path=expected_target_body_path,
            expected_destination_kind=expected_destination_kind,
            expected_destination_body_path=expected_destination_body_path,
            expected_destination_bottom_z=expected_destination_bottom_z,
            expected_zone=expected_zone,
        )
    except Exception as exc:
        raise SystemExit(
            f"published place sidecar pair is invalid: "
            f"{type(exc).__name__}: {exc}") from exc
    return copied


TASK29_FULL_METRICS_SCHEMA = "real2sim.small_task29_full_metrics.v6"


def validate_task29_sidecar_pair(
    npz_path,
    json_path,
    *,
    expected_episode_uid,
):
    """Validate full scope, current initial state and static collision policy."""

    try:
        payload = json.loads(
            Path(json_path).read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as metrics:
            npz_schema = str(metrics["schema"])
            npz_task = str(metrics["task_id"])
            npz_uid = str(metrics["episode_uid"])
            npz_success = bool(metrics["success"])
            rendered_frames = int(metrics["rendered_frames"])
            inside_destination = np.asarray(
                metrics["final_inside_destination"], dtype=bool)
            inside_distractor = np.asarray(
                metrics["final_inside_distractor"], dtype=bool)
            settled = np.asarray(metrics["final_settled"], dtype=bool)
            bin_offset_x = float(metrics["bin_offset_x_m"])
            bin_offset_y = float(metrics["bin_offset_y_m"])
            distractor_robot_side_offset_x = float(
                metrics["distractor_robot_side_offset_x_m"])
            table_offset_xy = np.asarray(
                metrics["coffee_table_task_offset_xy_m"], dtype=float)
            plate_min = np.asarray(
                metrics["paper_plate_world_aabb_min_xyz"], dtype=float)
            plate_max = np.asarray(
                metrics["paper_plate_world_aabb_max_xyz"], dtype=float)
            grip_command_target = np.asarray(
                metrics["grip_command_target_rad"], dtype=float)
            grip_actual_joint_position = np.asarray(
                metrics["grip_actual_joint_position_rad"], dtype=float)
            target_contact_force_vectors = np.asarray(
                metrics["target_contact_force_vectors"], dtype=float)
            target_contact_counts = np.asarray(
                metrics["target_contact_counts"], dtype=np.int64)
            physical_surface_stop_success = bool(
                metrics["physical_surface_stop_success"])
            target_static_friction = float(
                metrics["target_static_friction"])
            target_dynamic_friction = float(
                metrics["target_dynamic_friction"])
            target_combine_mode = str(metrics["target_combine_mode"])
            rim_static_friction = float(
                metrics["rim_static_friction"])
            rim_dynamic_friction = float(
                metrics["rim_dynamic_friction"])
            rim_combine_mode = str(metrics["rim_combine_mode"])
            initial_translate = np.asarray(
                metrics["initial_root_translate_xyz"], dtype=float)
            initial_rotate = np.asarray(
                metrics["initial_rotate_xyz"], dtype=float)
            table_min = np.asarray(
                metrics["coffee_table_world_aabb_min_xyz"], dtype=float)
            table_max = np.asarray(
                metrics["coffee_table_world_aabb_max_xyz"], dtype=float)
            table_render_prim = str(
                metrics["coffee_table_render_prim_path"])
            retired_table_present = bool(
                metrics["retired_coffee_table_visual_present"])
            robot_base = np.asarray(metrics["robot_base_xy"], dtype=float)
            robot_orient = np.asarray(
                metrics["robot_orient_wxyz"], dtype=float)
            robot_table_collision = bool(
                metrics["robot_coffee_table_collision_enabled"])
            all_gripper_collision = bool(
                metrics["all_gripper_collision_proxies_enabled"])
            static_cup_lod = bool(metrics["cup_collision_lod_static"])
            toggle_count = int(
                metrics["in_episode_collision_toggle_count"])
            event_frames = np.asarray(
                metrics["contact_lod_event_frames"], dtype=np.int64)
            event_enabled = np.asarray(
                metrics["contact_lod_event_enabled"], dtype=bool)
            settled_transition_frames = np.asarray(
                metrics["settled_transition_frames"], dtype=np.int64)
            settled_transition_cup_ordinals = np.asarray(
                metrics["settled_transition_cup_ordinals"],
                dtype=np.int64,
            )
            pad_below_table = float(
                metrics["pad_anchor_below_table_top_m"])
            no_gross_table_intrusion = bool(
                metrics["no_gross_pad_anchor_table_intrusion"])
        payload_bin_offset_x = float(payload["bin_offset_x_m"])
        payload_bin_offset_y = float(payload["bin_offset_y_m"])
        payload_distractor_robot_side_offset_x = float(
            payload["distractor_robot_side_offset_x_m"])
        payload_target_material = payload[
            "contact_material_policy"]["all_cups"]
        payload_rim_material = payload[
            "contact_material_policy"]["cup2_override"]
    except Exception as exc:
        raise ValueError(
            "invalid small Task 29 sidecar pair: "
            f"{type(exc).__name__}: {exc}") from exc
    expected_initial_translate = np.asarray([
        TASK29_FULL_INITIAL_POSES[path]["translate_xyz"]
        for path in TASK29_FULL_CUP_ROOT_PATHS
    ], dtype=float)
    expected_initial_rotate = np.asarray([
        TASK29_FULL_INITIAL_POSES[path]["rotate_xyz"]
        for path in TASK29_FULL_CUP_ROOT_PATHS
    ], dtype=float)
    expected_table_min = np.asarray((
        TASK29_COFFEE_TABLE_CENTER_XY_M[0]
        - COFFEE_TABLE_CURRENT_SIZE_XYZ_M[0] / 2.0,
        TASK29_COFFEE_TABLE_CENTER_XY_M[1]
        - COFFEE_TABLE_CURRENT_SIZE_XYZ_M[1] / 2.0,
        COFFEE_TABLE_CURRENT_BOTTOM_M,
    ), dtype=float)
    expected_table_max = expected_table_min + np.asarray(
        COFFEE_TABLE_CURRENT_SIZE_XYZ_M, dtype=float)
    expected_station = robot_position("table_pick")
    expected_teal_center = np.asarray(
        translate_xy(
            (
                -1.98 + TASK29_BIN_OFFSET_X_M,
                -1.06 + TASK29_BIN_OFFSET_Y_M,
            ),
            COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
        ),
        dtype=float,
    )
    expected_red_center = np.asarray(
        translate_xy(
            (
                -1.98
                + TASK29_BIN_OFFSET_X_M
                + TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
                -1.06 + TASK29_BIN_OFFSET_Y_M,
            ),
            COFFEE_TABLE_LAYOUT_SHIFT_XY_M,
        ),
        dtype=float,
    )
    expected_plate_min = np.asarray((
        TASK29_PAPER_PLATE_TRANSLATE_XYZ_M[0]
        - TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M[0] / 2.0,
        TASK29_PAPER_PLATE_TRANSLATE_XYZ_M[1]
        - TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M[1] / 2.0,
        TASK29_PAPER_PLATE_BOTTOM_M,
    ), dtype=float)
    expected_plate_max = expected_plate_min + np.asarray(
        TASK29_PAPER_PLATE_EXPECTED_SIZE_XYZ_M, dtype=float)
    grip_policy = payload.get("gripper_surface_stop_policy", {})
    phase_evidence = grip_policy.get("phase_evidence", [])
    grip_phase_commands_valid = (
        grip_command_target.shape == (471, 2)
        and grip_actual_joint_position.shape == (471, 2)
        and target_contact_force_vectors.ndim == 5
        and target_contact_force_vectors.shape[1:] == (2, 2, 3, 3)
        and target_contact_counts.shape
        == target_contact_force_vectors.shape[:-1]
        and np.isfinite(grip_command_target).all()
        and np.isfinite(grip_actual_joint_position).all()
        and np.isfinite(target_contact_force_vectors).all()
        and np.all(target_contact_counts >= 0)
        and physical_surface_stop_success
        and isinstance(phase_evidence, list)
        and len(phase_evidence) == len(TASK29_GRIP_PHASES)
        and all(
            isinstance(evidence, dict)
            and evidence.get("pass") is True
            for evidence in phase_evidence
        )
    )
    if grip_phase_commands_valid:
        for phase in TASK29_GRIP_PHASES:
            hand_index = 0 if phase["hand"] == "l" else 1
            phase_values = grip_command_target[
                phase["start_frame"]:phase["end_frame"] + 1,
                hand_index,
            ]
            phase_target = float(phase["physical_target_rad"])
            if phase["name"].endswith("_rim"):
                ramp = TASK29_RIM_GRIP_CLOSE_RAMP_FRAMES
                phase_valid = bool(
                    len(phase_values) >= ramp
                    and phase_target < phase_values[0]
                    <= TASK29_GRIP_OPEN_LIMIT_RAD
                    and np.all(np.diff(phase_values[:ramp]) < 0.0)
                    and np.allclose(
                        phase_values[ramp - 1:],
                        phase_target,
                        atol=1e-9,
                        rtol=0.0,
                    )
                )
            else:
                phase_valid = bool(np.allclose(
                    phase_values,
                    phase_target,
                    atol=1e-9,
                    rtol=0.0,
                ))
            if not phase_valid:
                grip_phase_commands_valid = False
                break
    expected_static_policy = {
        "cup2_rim_enabled_from_frame_zero": True,
        "bag_mouth_liner_enabled": True,
        "all_gripper_collision_proxies_enabled": True,
        "in_episode_collision_toggle_count": 0,
        "cup_pose_velocity_writes": 0,
        "settled_cup_transitions": [
            dict(transition)
            for transition in TASK29_SETTLED_CUP_TRANSITIONS
        ],
    }
    expected_target_material = {
        "static_friction": TASK29_TARGET_STATIC_FRICTION,
        "dynamic_friction": TASK29_TARGET_DYNAMIC_FRICTION,
        "combine_mode": TASK29_TARGET_COMBINE_MODE,
        "active_from_frame_zero": True,
    }
    expected_rim_material = {
        "static_friction": TASK29_RIM_STATIC_FRICTION,
        "dynamic_friction": TASK29_RIM_DYNAMIC_FRICTION,
        "combine_mode": TASK29_RIM_COMBINE_MODE,
    }
    initial_state = payload.get("initial_state")
    if (
        payload.get("schema") != TASK29_FULL_METRICS_SCHEMA
        or npz_schema != TASK29_FULL_METRICS_SCHEMA
        or payload.get("task_id") != "29"
        or npz_task != "29"
        or payload.get("episode_uid") != expected_episode_uid
        or npz_uid != expected_episode_uid
        or expected_episode_uid != TASK29_FULL_EPISODE_UID
        or payload.get("rendered_frames") != 471
        or rendered_frames != 471
        or payload.get("success") is not True
        or not npz_success
        or inside_destination.shape != (3,)
        or not np.all(inside_destination)
        or inside_distractor.shape != (3,)
        or np.any(inside_distractor)
        or settled.shape != (3,)
        or not np.all(settled)
        or not math.isclose(
            bin_offset_x,
            TASK29_BIN_OFFSET_X_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            bin_offset_y,
            TASK29_BIN_OFFSET_Y_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            distractor_robot_side_offset_x,
            TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or table_offset_xy.shape != (2,)
        or not np.allclose(
            table_offset_xy,
            np.asarray(TASK29_COFFEE_TABLE_OFFSET_XY_M),
            atol=1e-12,
            rtol=0.0,
        )
        or plate_min.shape != (3,)
        or plate_max.shape != (3,)
        or not np.allclose(
            plate_min, expected_plate_min, atol=0.001, rtol=0.0)
        or not np.allclose(
            plate_max, expected_plate_max, atol=0.001, rtol=0.0)
        or not grip_phase_commands_valid
        or not math.isclose(
            target_static_friction,
            TASK29_TARGET_STATIC_FRICTION,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            target_dynamic_friction,
            TASK29_TARGET_DYNAMIC_FRICTION,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or target_combine_mode != TASK29_TARGET_COMBINE_MODE
        or not math.isclose(
            rim_static_friction,
            TASK29_RIM_STATIC_FRICTION,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            rim_dynamic_friction,
            TASK29_RIM_DYNAMIC_FRICTION,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or rim_combine_mode != TASK29_RIM_COMBINE_MODE
        or initial_translate.shape != (3, 3)
        or not np.allclose(
            initial_translate,
            expected_initial_translate,
            atol=1e-8,
            rtol=0.0,
        )
        or initial_rotate.shape != (3, 3)
        or not np.allclose(
            initial_rotate,
            expected_initial_rotate,
            atol=1e-7,
            rtol=0.0,
        )
        or table_min.shape != (3,)
        or table_max.shape != (3,)
        or not np.allclose(
            table_min, expected_table_min, atol=0.002, rtol=0.0)
        or not np.allclose(
            table_max, expected_table_max, atol=0.002, rtol=0.0)
        or table_render_prim != "/World/Assets/CoffeeTable"
        or retired_table_present
        or robot_base.shape != (2,)
        or not np.allclose(
            robot_base,
            np.asarray(expected_station.xy_m, dtype=float),
            atol=1e-8,
            rtol=0.0,
        )
        or robot_orient.shape != (4,)
        or not np.allclose(
            robot_orient,
            np.asarray(expected_station.quat_wxyz, dtype=float),
            atol=1e-7,
            rtol=0.0,
        )
        or not robot_table_collision
        or not all_gripper_collision
        or not static_cup_lod
        or toggle_count != 0
        or event_frames.shape != (0,)
        or event_enabled.shape != (0,)
        or not np.array_equal(
            settled_transition_frames,
            np.asarray(
                [
                    transition["frame"]
                    for transition in TASK29_SETTLED_CUP_TRANSITIONS
                ],
                dtype=np.int64,
            ),
        )
        or not np.array_equal(
            settled_transition_cup_ordinals,
            np.asarray(
                [
                    transition["cup_ordinal"]
                    for transition in TASK29_SETTLED_CUP_TRANSITIONS
                ],
                dtype=np.int64,
            ),
        )
        or not math.isfinite(pad_below_table)
        or pad_below_table < 0.0
        or pad_below_table > 0.005
        or not no_gross_table_intrusion
        or payload.get("static_collision_policy")
        != expected_static_policy
        or not math.isclose(
            payload_bin_offset_x,
            TASK29_BIN_OFFSET_X_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            payload_bin_offset_y,
            TASK29_BIN_OFFSET_Y_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            payload_distractor_robot_side_offset_x,
            TASK29_DISTRACTOR_ROBOT_SIDE_OFFSET_X_M,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or grip_policy.get("mode")
        != "pure_physical_target_contact_surface_stop"
        or grip_policy.get("physical_closed_target_rad_by_grasp")
        != TASK29_GRIP_PHYSICAL_CLOSED_TARGET_RAD
        or not math.isclose(
            grip_policy.get("physical_close_max_force", float("nan")),
            TASK29_GRIP_CLOSE_MAX_FORCE,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            grip_policy.get(
                "rim_physical_close_max_force", float("nan")),
            TASK29_RIM_GRIP_CLOSE_MAX_FORCE,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or grip_policy.get("rim_close_ramp_frames")
        != TASK29_RIM_GRIP_CLOSE_RAMP_FRAMES
        or not np.allclose(
            np.asarray(
                grip_policy.get("rim_grasp_world_offset_xyz_m"),
                dtype=float,
            ),
            np.asarray(
                TASK29_RIM_GRASP_WORLD_OFFSET_XYZ_M,
                dtype=float,
            ),
            atol=1e-12,
            rtol=0.0,
        )
        or grip_policy.get("visual_link_offsets_m") != {}
        or grip_policy.get("visual_transform_authored") is not False
        or grip_policy.get("collision_geometry_mode")
        != "stock_visible_collision_coincident"
        or grip_policy.get("collision_transform_authored") is not False
        or grip_policy.get("phases")
        != [dict(phase) for phase in TASK29_GRIP_PHASES]
        or grip_policy.get("success") is not True
        or payload_target_material != expected_target_material
        or any(
            payload_rim_material.get(key) != value
            for key, value in expected_rim_material.items()
        )
        or payload.get("in_episode_collision_events") != []
        or payload.get("no_gross_pad_anchor_table_intrusion") is not True
        or not isinstance(initial_state, dict)
        or not np.allclose(
            np.asarray(
                initial_state.get("cup_root_translate_xyz_m"),
                dtype=float,
            ),
            expected_initial_translate,
            atol=1e-8,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(
                initial_state.get("coffee_table_world_aabb_min_xyz_m"),
                dtype=float,
            ),
            expected_table_min,
            atol=0.002,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(
                initial_state.get("coffee_table_world_aabb_max_xyz_m"),
                dtype=float,
            ),
            expected_table_max,
            atol=0.002,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(
                initial_state.get("coffee_table_task_offset_xy_m"),
                dtype=float,
            ),
            np.asarray(TASK29_COFFEE_TABLE_OFFSET_XY_M),
            atol=1e-12,
            rtol=0.0,
        )
        or initial_state.get("coffee_table_render_policy") != {
            "render_prim_path": "/World/Assets/CoffeeTable",
            "physical_prim_path": "/World/Assets/CoffeeTable",
            "retired_clone_path": TASK29_RETIRED_COFFEE_TABLE_VISUAL_PATH,
            "retired_clone_present": False,
            "render_visibility": "inherited",
        }
        or initial_state.get("paper_plate", {}).get(
            "prim_path") != TASK29_PAPER_PLATE_PATH
        or initial_state.get("paper_plate", {}).get(
            "rigid_body_enabled") is not False
        or initial_state.get("paper_plate", {}).get(
            "kinematic") is not False
        or initial_state.get("paper_plate", {}).get(
            "collision_enabled") is not False
        or initial_state.get("non_task_render_context") != {
            "prim_path": "/World/Assets/Task56CurrentContext",
            "active_for_accepted_physics_topology": True,
            "render_visibility": "invisible",
        }
        or not np.allclose(
            np.asarray(
                initial_state.get("paper_plate", {}).get(
                    "world_aabb_min_xyz_m"),
                dtype=float,
            ),
            expected_plate_min,
            atol=0.001,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(
                initial_state.get("paper_plate", {}).get(
                    "world_aabb_max_xyz_m"),
                dtype=float,
            ),
            expected_plate_max,
            atol=0.001,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(
                payload.get("destination", {}).get("center_xy_m"),
                dtype=float,
            ),
            expected_teal_center,
            atol=1e-8,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(
                payload.get("distractor", {}).get("center_xy_m"),
                dtype=float,
            ),
            expected_red_center,
            atol=1e-8,
            rtol=0.0,
        )
        or payload.get("distractor", {}).get(
            "collision_enabled") is not True
        or payload.get("distractor", {}).get(
            "role") != "physical_adjacent_robot_side_distractor"
    ):
        raise ValueError(
            "small Task 29 sidecar does not prove the current three-cup "
            "initial state, plate/bin layout, surface-stop gripper policy, "
            "static contact policy, no gross table intrusion and three "
            "settled teal-bin entries")
    return payload


def copy_task29_sidecars(out, *, expected_episode_uid):
    """Validate and atomically publish the full small Task 29 evidence pair."""
    sources = {
        name: Path(DEMO) / "cam_preview" / "video3" / name
        for name in (
            "task29_full_metrics.npz",
            "task29_full_metrics.json",
        )
    }
    if not all(path.is_file() for path in sources.values()):
        return []
    try:
        validate_task29_sidecar_pair(
            sources["task29_full_metrics.npz"],
            sources["task29_full_metrics.json"],
            expected_episode_uid=expected_episode_uid,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    copied = []
    pending = []
    try:
        for name, src in sources.items():
            dst = out.with_name(out.stem + "." + name)
            tmp = dst.with_name(
                dst.name + f".{uuid.uuid4().hex}.tmp")
            shutil.copy2(src, tmp)
            with tmp.open("r+b") as stream:
                os.fsync(stream.fileno())
            pending.append((tmp, dst))
        for tmp, dst in pending:
            os.replace(tmp, dst)
            copied.append(dst)
            print(f"wrote {dst}", flush=True)
    finally:
        if copied:
            fsync_directories(out.parent)
        for tmp, _dst in pending:
            tmp.unlink(missing_ok=True)
    try:
        validate_task29_sidecar_pair(
            out.with_name(out.stem + ".task29_full_metrics.npz"),
            out.with_name(out.stem + ".task29_full_metrics.json"),
            expected_episode_uid=expected_episode_uid,
        )
    except ValueError as exc:
        raise SystemExit(
            f"published small Task 29 sidecar pair is invalid: {exc}"
        ) from exc
    return copied


def validate_pick_contact_trace(
    path,
    *,
    expected_task_id,
    expected_episode_uid,
    expected_runtime_revision,
    expected_target_body_path,
    expected_frame_count,
    expected_furniture_center_xy_m=None,
):
    """Require a complete, finite trace tied to this exact pick render."""
    try:
        with np.load(path, allow_pickle=False) as trace:
            schema = str(trace["schema"].item())
            task_id = str(trace["task_id"].item())
            episode_uid = str(trace["episode_uid"].item())
            runtime_revision = (
                json.loads(str(trace["runtime_source_revision"].item()))
                if "runtime_source_revision" in trace.files
                else None
            )
            target_body_path = str(trace["target_body_path"].item())
            frame_count = int(trace["frame_count"].item())
            fps = float(trace["fps"].item())
            fabric_obj_xyz = np.asarray(
                trace["fabric_obj_xyz"], dtype=float)
            usd_obj_xyz = np.asarray(trace["usd_obj_xyz"], dtype=float)
            pad_xyz = np.asarray(trace["pad_xyz"], dtype=float)
            grip_joint_position = np.asarray(
                trace["grip_joint_position"], dtype=float)
            pad_force_vectors = np.asarray(
                trace["pad_force_vectors"], dtype=float)
            pad_contacts = np.asarray(trace["pad_contacts"])
            layout_errors = (
                current_layout_trace_errors(
                    expected_task_id,
                    trace,
                    expected_furniture_center_xy_m=(
                        expected_furniture_center_xy_m),
                )
                if str(expected_task_id) in CURRENT_LAYOUT_SUCCESS_TASKS
                else []
            )
            shelf_surface_stop = None
            shelf_grasp_align = None
            if str(expected_task_id) in SHELF_SURFACE_STOP_TASK_IDS:
                shelf_surface_stop = {
                    "enabled": bool(
                        trace["shelf_surface_stop_enabled"].item()),
                    "latch_frame": int(
                        trace["shelf_surface_stop_latch_frame"].item()),
                    "latch_gap_m": float(
                        trace["shelf_surface_stop_latch_gap_m"].item()),
                    "latch_force_n": np.asarray(
                        trace["shelf_surface_stop_latch_force_n"],
                        dtype=float,
                    ),
                    "hold_target_rad": float(
                        trace[
                            "shelf_surface_stop_hold_target_rad"].item()),
                    "preload_rad": float(
                        trace["shelf_surface_stop_preload_rad"].item()),
                    "force_threshold_n": float(
                        trace[
                            "shelf_surface_stop_force_threshold_n"].item()),
                    "force_hold_n": float(
                        trace[
                            "shelf_surface_stop_force_hold_n"].item()),
                    "max_follow_rad": float(
                        trace[
                            "shelf_surface_stop_max_follow_rad"].item()),
                    "follow_adjustment_frames": int(
                        trace[
                            "shelf_surface_stop_follow_adjustment_frames"
                        ].item()),
                }
                shelf_grasp_align = {
                    "enabled": bool(
                        trace["shelf_grasp_align_enabled"].item()),
                    "recorded_waist_trajectory_modified": bool(
                        trace[
                            "shelf_recorded_waist_trajectory_modified"
                        ].item()),
                    "lift_offset_m": float(
                        trace["shelf_lift_offset_m"].item()),
                    "requested_xyz_m": np.asarray(
                        trace["shelf_grasp_align_requested_xyz_m"],
                        dtype=float,
                    ),
                    "predicted_xyz_m": np.asarray(
                        trace["shelf_grasp_align_predicted_xyz_m"],
                        dtype=float,
                    ),
                    "requested_rotation_vector_rad": np.asarray(
                        trace[
                            "shelf_grasp_align_requested_rotation_vector_rad"
                        ],
                        dtype=float,
                    ),
                    "predicted_rotation_vector_rad": np.asarray(
                        trace[
                            "shelf_grasp_align_predicted_rotation_vector_rad"
                        ],
                        dtype=float,
                    ),
                    "joint_delta_rad": np.asarray(
                        trace["shelf_grasp_align_joint_delta_rad"],
                        dtype=float,
                    ),
                    "target_center_z_m": float(
                        trace[
                            "shelf_grasp_align_target_center_z_m"].item()),
                    "pad_center_z_m": float(
                        trace[
                            "shelf_grasp_align_pad_center_z_m"].item()),
                    "plan_frame": int(
                        trace["shelf_grasp_align_plan_frame"].item()),
                    "full_frame": int(
                        trace["shelf_grasp_align_full_frame"].item()),
                }
    except Exception as exc:
        raise ValueError(
            f"unreadable pick contact trace: {type(exc).__name__}: {exc}"
        ) from exc
    expected_task_id = str(expected_task_id)
    if schema != "real2sim.pick_contact_trace.v1":
        raise ValueError(f"unexpected pick trace schema {schema!r}")
    identity = {
        "task_id": (task_id, expected_task_id),
        "episode_uid": (episode_uid, str(expected_episode_uid)),
        "target_body_path": (
            target_body_path, str(expected_target_body_path)),
        "frame_count": (frame_count, int(expected_frame_count)),
    }
    if expected_runtime_revision is not None:
        identity["runtime_source_revision"] = (
            runtime_revision, expected_runtime_revision
        )
    mismatches = [
        f"{name}: observed={observed!r}, expected={expected!r}"
        for name, (observed, expected) in identity.items()
        if observed != expected
    ]
    if mismatches:
        raise ValueError(
            "pick contact trace identity mismatch: " + "; ".join(mismatches))
    if layout_errors:
        raise ValueError(
            "pick contact trace current-layout mismatch: "
            + "; ".join(layout_errors))
    if not math.isclose(fps, FPS, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"pick contact trace fps is {fps}, expected {FPS}")
    expected_shapes = {
        "fabric_obj_xyz": (
            fabric_obj_xyz, (frame_count, 3)),
        "usd_obj_xyz": (usd_obj_xyz, (frame_count, 3)),
        "pad_xyz": (pad_xyz, (frame_count, 2, 3)),
        "grip_joint_position": (
            grip_joint_position, (frame_count,)),
        "pad_force_vectors": (
            pad_force_vectors, (frame_count, 2, 3)),
        "pad_contacts": (pad_contacts, (frame_count, 2)),
    }
    for name, (array, shape) in expected_shapes.items():
        if array.shape != shape:
            raise ValueError(
                f"pick contact trace {name} shape {array.shape}, "
                f"expected {shape}")
        if name != "pad_contacts" and not np.isfinite(array).all():
            raise ValueError(
                f"pick contact trace {name} contains non-finite values")
    if pad_contacts.dtype.kind != "b":
        raise ValueError("pick contact trace pad_contacts must be boolean")
    if shelf_surface_stop is not None:
        latch_frame = shelf_surface_stop["latch_frame"]
        latch_force_n = shelf_surface_stop["latch_force_n"]
        if not shelf_surface_stop["enabled"]:
            raise ValueError(
                "shelf pick contact trace has surface-stop disabled")
        if latch_force_n.shape != (2,):
            raise ValueError(
                "shelf pick contact trace latch forces must have shape (2,)")
        if (
            latch_frame < -1
            or latch_frame >= frame_count
            or not math.isfinite(shelf_surface_stop["preload_rad"])
            or shelf_surface_stop["preload_rad"] <= 0.0
            or not math.isfinite(shelf_surface_stop["force_threshold_n"])
            or shelf_surface_stop["force_threshold_n"] <= 0.0
            or not math.isfinite(shelf_surface_stop["force_hold_n"])
            or (
                shelf_surface_stop["force_hold_n"]
                < shelf_surface_stop["force_threshold_n"]
            )
            or not math.isfinite(shelf_surface_stop["max_follow_rad"])
            or not 0.0 < shelf_surface_stop["max_follow_rad"] <= 0.4
            or shelf_surface_stop["follow_adjustment_frames"] < 0
            or (
                shelf_surface_stop["follow_adjustment_frames"]
                > frame_count
            )
        ):
            raise ValueError(
                "shelf pick contact trace has invalid surface-stop controls")
        if latch_frame >= 0:
            if (
                not math.isfinite(shelf_surface_stop["latch_gap_m"])
                or shelf_surface_stop["latch_gap_m"] <= 0.0
                or not np.isfinite(latch_force_n).all()
                or np.any(
                    latch_force_n
                    < shelf_surface_stop["force_threshold_n"])
                or not math.isfinite(
                    shelf_surface_stop["hold_target_rad"])
            ):
                raise ValueError(
                    "shelf pick contact trace has invalid latch evidence")
        elif (
            math.isfinite(shelf_surface_stop["latch_gap_m"])
            or np.isfinite(latch_force_n).any()
            or math.isfinite(shelf_surface_stop["hold_target_rad"])
        ):
            raise ValueError(
                "unlatched shelf pick trace contains latch measurements")
        shelf_grasp_align_translation_limit_m = (
            0.145 if str(task_id) == "38" else 0.12
        )
        if shelf_grasp_align["enabled"] and (
            shelf_grasp_align["requested_xyz_m"].shape != (3,)
            or shelf_grasp_align["predicted_xyz_m"].shape != (3,)
            or shelf_grasp_align["joint_delta_rad"].shape != (7,)
            or (
                shelf_grasp_align[
                    "requested_rotation_vector_rad"].shape
                != (3,)
            )
            or (
                shelf_grasp_align[
                    "predicted_rotation_vector_rad"].shape
                != (3,)
            )
            or not np.isfinite(
                shelf_grasp_align["requested_xyz_m"]).all()
            or not np.isfinite(
                shelf_grasp_align["predicted_xyz_m"]).all()
            or not np.isfinite(
                shelf_grasp_align["joint_delta_rad"]).all()
            or not np.isfinite(
                shelf_grasp_align[
                    "requested_rotation_vector_rad"]).all()
            or not np.isfinite(
                shelf_grasp_align[
                    "predicted_rotation_vector_rad"]).all()
            or not math.isfinite(
                shelf_grasp_align["target_center_z_m"])
            or not math.isfinite(
                shelf_grasp_align["pad_center_z_m"])
            or shelf_grasp_align["plan_frame"] < 0
            or not (
                shelf_grasp_align["plan_frame"]
                < shelf_grasp_align["full_frame"]
                < frame_count
            )
            or np.linalg.norm(
                shelf_grasp_align["requested_xyz_m"]
            ) > shelf_grasp_align_translation_limit_m + 1e-9
            or np.linalg.norm(
                shelf_grasp_align["predicted_xyz_m"]
                - shelf_grasp_align["requested_xyz_m"]
            ) > 0.005000001
            or np.linalg.norm(
                shelf_grasp_align[
                    "predicted_rotation_vector_rad"]
                - shelf_grasp_align[
                    "requested_rotation_vector_rad"]
            ) > 0.050000001
        ):
            raise ValueError(
                "shelf pick contact trace has invalid grasp alignment")
        if not shelf_grasp_align["enabled"] and (
            shelf_grasp_align["requested_xyz_m"].shape != (3,)
            or shelf_grasp_align["predicted_xyz_m"].shape != (3,)
            or shelf_grasp_align["joint_delta_rad"].shape != (7,)
            or not np.allclose(
                shelf_grasp_align["requested_xyz_m"], 0.0)
            or not np.allclose(
                shelf_grasp_align["predicted_xyz_m"], 0.0)
            or not np.allclose(
                shelf_grasp_align["joint_delta_rad"], 0.0)
            or shelf_grasp_align["plan_frame"] != -1
            or shelf_grasp_align["full_frame"] != -1
        ):
            raise ValueError(
                "recorded-action shelf trace contains a grasp alignment")
        if (
            shelf_grasp_align["recorded_waist_trajectory_modified"]
            or not math.isclose(
                shelf_grasp_align["lift_offset_m"],
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "recorded-action shelf trace changed the waist trajectory")
    return {
        "schema": schema,
        "task_id": task_id,
        "episode_uid": episode_uid,
        "runtime_source_revision": runtime_revision,
        "target_body_path": target_body_path,
        "frame_count": frame_count,
        "fps": fps,
        "shelf_surface_stop": shelf_surface_stop,
        "shelf_grasp_align": shelf_grasp_align,
    }


def copy_pick_contact_trace(
    out,
    *,
    expected_task_id,
    expected_episode_uid,
    expected_runtime_revision,
    expected_target_body_path,
    expected_frame_count,
    expected_furniture_center_xy_m=None,
):
    """Validate and atomically publish the trace beside its comparison MP4."""
    source = (
        Path(DEMO) / "cam_preview" / "video3" / "pick_contact_trace.npz"
    )
    if not source.is_file():
        return None
    expected = {
        "expected_task_id": expected_task_id,
        "expected_episode_uid": expected_episode_uid,
        "expected_runtime_revision": expected_runtime_revision,
        "expected_target_body_path": expected_target_body_path,
        "expected_frame_count": expected_frame_count,
        "expected_furniture_center_xy_m": (
            expected_furniture_center_xy_m),
    }
    try:
        validate_pick_contact_trace(source, **expected)
    except Exception as exc:
        raise SystemExit(
            "invalid pick contact trace identity: "
            f"{type(exc).__name__}: {exc}") from exc
    destination = out.with_name(out.stem + ".pick_contact_trace.npz")
    temporary = destination.with_name(
        destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directories(out.parent)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        validate_pick_contact_trace(destination, **expected)
    except Exception as exc:
        raise SystemExit(
            "published pick contact trace is invalid: "
            f"{type(exc).__name__}: {exc}") from exc
    print(f"wrote {destination}", flush=True)
    return destination


def copy_distractor_audit(
        out, *, expected_task_id, expected_episode_uid, expected_prims):
    """Validate and atomically bind randomized-prop motion to one output."""
    expected = sorted(str(path) for path in expected_prims)
    if not expected:
        return None
    plan_source = Path(
        str(SCENE_PATH) + ".episode_randomization.json")
    try:
        plan = json.loads(plan_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"missing or invalid episode randomization plan: {exc}") from exc
    if (
        plan.get("schema")
        != "real2sim.episode_scene_randomization.applied.v1"
        or str(plan.get("runtime_task_id", plan.get("task_id"))) != str(expected_task_id)
        or plan.get("enabled") is not True
        or sorted(str(path) for path in (
            plan.get("interactive_prim_paths") or ())) != expected
    ):
        raise SystemExit(
            "episode randomization plan does not match the runtime "
            "distractor inventory")
    source = (
        Path(DEMO) / "cam_preview" / "video3" / "distractor_audit.json"
    )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"missing or invalid distractor no-touch audit: {exc}") from exc
    errors = distractor_audit_errors(
        payload,
        task_id=expected_task_id,
        episode_uid=expected_episode_uid,
        expected_prims=expected,
    )
    if errors:
        raise SystemExit(
            "distractor no-touch audit failed: " + "; ".join(errors))

    plan_destination = out.with_name(
        out.stem + ".episode_randomization.json")
    destination = out.with_name(out.stem + ".distractor_audit.json")
    plan_temporary = plan_destination.with_name(
        plan_destination.name + f".{uuid.uuid4().hex}.tmp")
    temporary = destination.with_name(
        destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(plan_source, plan_temporary)
        shutil.copy2(source, temporary)
        with plan_temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(plan_temporary, plan_destination)
        os.replace(temporary, destination)
        fsync_directories(out.parent)
    finally:
        plan_temporary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    try:
        copied = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"published distractor audit is unreadable: {exc}") from exc
    copied_errors = distractor_audit_errors(
        copied,
        task_id=expected_task_id,
        episode_uid=expected_episode_uid,
        expected_prims=expected,
    )
    if copied_errors:
        raise SystemExit(
            "published distractor no-touch audit failed: "
            + "; ".join(copied_errors))
    copied_plan = json.loads(
        plan_destination.read_text(encoding="utf-8"))
    if copied_plan != plan:
        raise SystemExit(
            "published episode randomization plan changed in transit")
    print(f"wrote {destination}", flush=True)
    return destination


def archive_output_sidecars(out, reason):
    """Move derived sidecars away when their video provenance cannot be proven."""
    existing = [
        out.with_name(out.stem + "." + name)
        for name in (
            "place_metrics.npz",
            "place_metrics.json",
            "task29_full_metrics.npz",
            "task29_full_metrics.json",
            "pick_contact_trace.npz",
            "episode_randomization.json",
            "distractor_audit.json",
            "asset_policy.json",
            "sim_only_video.json",
        )
        if out.with_name(out.stem + "." + name).is_file()
    ]
    if not existing:
        return []
    root = out.parent / "_superseded"
    root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(
        prefix=f"{out.stem}_{reason}_", dir=root))
    archived = []
    for path in existing:
        target = destination / path.name
        os.replace(path, target)
        archived.append(target)
        print(f"archived unverifiable sidecar {path} -> {target}", flush=True)
    fsync_directories(out.parent, root, destination)
    return archived


def validated_sim_frames():
    """Require one contiguous image from every camera for every trajectory frame."""
    expected = completed_frame_count()
    frames = {}
    for camera in SIM_DIRS:
        directory = Path(DEMO) / "cam_preview" / "video3" / camera
        paths = sorted(directory.glob("f_*.jpg"))
        wanted = [f"f_{index:04d}.jpg" for index in range(expected)]
        names = [path.name for path in paths]
        if names != wanted:
            missing = sorted(set(wanted) - set(names))
            extra = sorted(set(names) - set(wanted))
            raise SystemExit(
                f"{camera} frame sequence is not contiguous: "
                f"missing={missing[:5]} extra={extra[:5]}")
        frames[camera] = [str(path) for path in paths]
    return frames, expected


def completed_frame_count():
    """Read the post-app-close completion marker and return its exact frame count."""
    marker = Path(DEMO) / "cam_preview" / "video3" / "video3.txt"
    try:
        match = re.fullmatch(
            r"frames ([0-9]+) of ([0-9]+)\s*",
            marker.read_text(errors="strict"),
        )
    except (OSError, UnicodeError):
        match = None
    if not match:
        raise SystemExit("missing or invalid completed video3 frame manifest")
    written, expected = map(int, match.groups())
    if expected < 1 or written != expected:
        raise SystemExit(
            f"incomplete video3 frame manifest: wrote {written} of {expected}")
    return expected


def metrics_only_output_kind(init_result, episode_task):
    """Return the physical evidence kind supported without camera output."""
    family = str((init_result or {}).get("family", ""))
    task = str((init_result or {}).get("task", episode_task))
    if family == "throw" and task == "29":
        return "task29"
    if REGISTRY.get(str(episode_task), {}).get("family") in PICK_CONTACT_FAMILIES:
        return "pick"
    if family == "place":
        return "place"
    raise ValueError(
        "--metrics-only requires a successful place, pick/snack/tier2, "
        "or current full small Task 29 init")


def requires_contact_trace(init_result, episode_task):
    """Return whether playback must publish target/pad telemetry.

    The historical filename says ``pick_contact_trace``, but the artifact is
    also the canonical source for per-frame pad positions in an in-hand-start
    place episode.  Such episodes have no pickup phase; they still need the
    runtime trace from frame zero so state export and structural validation do
    not have to invent or zero-fill pad geometry.
    """
    family = str((init_result or {}).get("family", ""))
    registry_family = REGISTRY.get(str(episode_task), {}).get("family")
    return family == "place" or registry_family in PICK_CONTACT_FAMILIES


def require_decoded_frame(image, *, source, camera, frame):
    """Never publish a comparison containing a silent grey missing-frame tile."""
    if (image is None or not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[0] < 1 or image.shape[1] < 1):
        raise RuntimeError(
            f"{source} camera {camera} frame {frame} is missing or undecodable")
    if image.shape[2] < 3:
        raise RuntimeError(
            f"{source} camera {camera} frame {frame} has fewer than 3 channels")
    return image


def next_decoded_frame(iterator, *, source, camera, frame):
    try:
        image = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(
            f"{source} camera {camera} ended before frame {frame}") from exc
    return require_decoded_frame(
        image, source=source, camera=camera, frame=frame)


def fit_cell(img):
    """Resize into a CELL_W x CELL_H box preserving aspect, pad with dark grey."""
    return fit_box(img, CELL_W, CELL_H)


def fit_box(img, width, height):
    """Resize into an explicit box preserving aspect, padded with dark grey."""
    cell = np.full((height, width, 3), 24, np.uint8)
    h, w = img.shape[:2]
    s = min(width / w, height / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh))
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = r
    return cell


def band(text, width):
    b = np.full((BAND, width, 3), 8, np.uint8)
    cv2.putText(b, text, (10, BAND - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return b


def task_owned_arm_initial_environment(task_id, requested, full_state_capture):
    """Forward only action-frame initializers already implemented by playback.

    This selects no perturbation and changes no action. Playback must still
    converge to the commanded first frame and record its measured readback.
    Other initializer modes use their task-owned producer entrypoints.
    """
    requested = requested.strip()
    if not requested:
        return {}
    if str(task_id) not in {"38", "405", "55", "60"}:
        raise ValueError(
            "task-owned arm action-frame initializer is not implemented "
            f"for runtime {task_id}")
    if full_state_capture != "1":
        raise ValueError(
            "task-owned arm initialization requires FULL_STATE_CAPTURE=1")
    try:
        values = np.asarray([float(value) for value in requested.split(",")])
    except ValueError as exc:
        raise ValueError(
            "TASK_OWNED_ARM_INITIAL_DELTA_RAD must contain 14 finite "
            "comma-separated numbers") from exc
    if (values.shape != (14,) or not np.isfinite(values).all()
            or np.linalg.norm(values) <= 1e-12
            or np.max(np.abs(values)) > math.radians(4.0)):
        raise ValueError(
            "TASK_OWNED_ARM_INITIAL_DELTA_RAD must be a finite nonzero "
            "14-vector within the 4-degree per-joint bound")
    return {"TASK_OWNED_ARM_INITIAL_DELTA_RAD": requested}


def main():
    try:
        lock_stream = acquire_or_inherit_lock(PLACE_LOCK)
    except BlockingIOError:
        raise SystemExit(
            "another managed place render owns the shared Isaac lock")
    # Pick/throw batch runners historically used a separate lock even though
    # they mutate the same scene_room.usda and playback_traj.npz.  Hold both
    # locks so a comparison can never mix one episode's real frames with a
    # concurrently replaced sim trajectory.
    scene_lock_stream = SCENE_TRANSACTION_LOCK.open("a+", encoding="ascii")
    fcntl.flock(scene_lock_stream, fcntl.LOCK_EX)
    ap = argparse.ArgumentParser()
    ap.add_argument("uuid")
    ap.add_argument("--skip-render", action="store_true", help="reuse existing video3 frames")
    ap.add_argument("--metrics-only", action="store_true",
                    help="run identical physics and write strict physical "
                    "sidecars, without camera IO or comparison encoding")
    ap.add_argument("--sim-only", action="store_true",
                    help="run live physics and encode the three SIM cameras "
                    "without requiring optional real-episode camera data")
    ap.add_argument("--skip-refine", action="store_true", help="keep the scene as it is")
    ap.add_argument("--init-cmd", default="", help="argv command that places the scene for this "
                    "episode (family init, e.g. 'PICK_TASK=53 conda run -n base python "
                    "real2sim/teleop/init_pick.py'); default = book refine_episode")
    ap.add_argument("--traj-filter", default="", help="generation-layer hook "
                    "(datagen): argv command run on the freshly extracted "
                    "trajectory npz (path appended as last arg) between "
                    "extraction and publish; the rewritten file must pass the "
                    "same schema validation as the recorded one")
    ap.add_argument("--trajectory-input", default="", help="validated, "
                    "already-extracted playback_traj.npz from a prior "
                    "physics-only handoff; mutually exclusive with "
                    "--traj-filter")
    ap.add_argument("--out", default="", help="output mp4 path (default dataset/sim_episodes_repro/<tag>_compare3.mp4)")
    a = ap.parse_args()
    try:
        house_layout_request = house_layout_canary_hook.load_request()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ap.error(f"invalid House-layout canary request: {exc}")
    house_layout_action = None
    house_layout_expected_furniture_center_xy_m = None
    if a.metrics_only and a.skip_render:
        ap.error("--metrics-only cannot be combined with --skip-render")
    if a.sim_only and (a.metrics_only or a.skip_render):
        ap.error("--sim-only cannot be combined with --metrics-only/--skip-render")
    if a.sim_only and not a.trajectory_input:
        ap.error("--sim-only requires --trajectory-input")
    if a.trajectory_input and a.traj_filter:
        ap.error("--trajectory-input cannot be combined with --traj-filter")
    if a.trajectory_input and a.skip_render:
        ap.error("--trajectory-input cannot be combined with --skip-render")
    if os.environ.get("INSTANCE", ""):
        ap.error(
            "INSTANCE is unsupported here: init and playback must use the same canonical "
            "scene/trajectory/output paths")
    if has_instance_assignment(a.init_cmd) or has_instance_assignment(
            os.environ.get("SIM_ENV", "")):
        ap.error("INSTANCE assignment is unsupported in --init-cmd/SIM_ENV")
    try:
        sim_env_values = parse_env_assignments(os.environ.get("SIM_ENV", ""))
    except ValueError as exc:
        ap.error(str(exc))
    try:
        coffee_table_calibration = (
            coffee_table_calibration_from_environment()
        )
    except ValueError as exc:
        ap.error(f"invalid CoffeeTable size calibration: {exc}")
    reserved = sorted(set(sim_env_values) & RESERVED_SIM_ENV)
    if reserved:
        ap.error(
            "SIM_ENV cannot override managed runtime identity: "
            + ", ".join(reserved))
    unsupported = sorted(set(sim_env_values) - ALLOWED_SIM_ENV)
    if unsupported:
        ap.error(
            "SIM_ENV contains unsupported runtime knobs: "
            + ", ".join(unsupported))
    if a.out and Path(a.out).suffix.lower() != ".mp4":
        ap.error("--out must use an .mp4 suffix")
    recordless_replay = False
    try:
        epdir = resolve_episode_dir(a.uuid)
    except ValueError as exc:
        explicit_task = os.environ.get("REAL2SIM_TASK_ID", "").strip()
        try:
            canonical_uid = str(uuid.UUID(a.uuid))
        except ValueError:
            canonical_uid = ""
        if not (
            (a.metrics_only or a.sim_only)
            and a.trajectory_input
            and canonical_uid
            and explicit_task
            and os.environ.get("EPISODE_UID", canonical_uid)
            == canonical_uid
        ):
            raise SystemExit(str(exc)) from exc
        epdir = Path(REC) / canonical_uid
        recordless_replay = True
    uid = epdir.name
    tag = uid[:8]
    task53_episode = None
    try:
        episode_task = (
            os.environ["REAL2SIM_TASK_ID"].strip()
            if recordless_replay
            else resolve_episode_task_id(epdir)
        )
        if recordless_replay and episode_task not in (
            set(REGISTRY) | set(LEGACY_RUNTIME_TARGET_PRIMS)
        ):
            raise ValueError(
                "recordless replay task is absent from the runtime registry")
        if (
            house_layout_request is not None
            and episode_task in CURRENT_LAYOUT_SUCCESS_TASKS
        ):
            furniture = current_layout_success_contract(
                episode_task)["furniture"]
            house_layout_expected_furniture_center_xy_m = (
                house_layout_canary_hook.transform_frame_points(
                    furniture["center_xy_m"], house_layout_request,
                ).tolist()
            )
        if episode_task == "53":
            task53_episode = task53_episode_calibration(
                task53_contact_scene_calibration_identity(
                    TASK53_CALIBRATION_SOURCE),
                uid,
            )
            controls = task53_episode["contact_controls"]
            task53_defaults = {
                "GRIP_PAST": repr(controls["grip_past_rad"]),
                "TASK53_STATIC_FRICTION": repr(
                    controls["static_friction"]),
                "TASK53_DYNAMIC_FRICTION": repr(
                    controls["dynamic_friction"]),
                "TASK53_TORSIONAL_PATCH_RADIUS": repr(
                    controls["torsional_patch_radius_m"]),
                "TASK53_MIN_TORSIONAL_PATCH_RADIUS": repr(
                    controls["min_torsional_patch_radius_m"]),
                "TASK53_REAL_VISUALS": "1",
            }
            for key, value in task53_defaults.items():
                sim_env_values.setdefault(key, value)
        for key, value in EPISODE_SIM_ENV_DEFAULTS.get(
                (episode_task, uid), {}).items():
            sim_env_values.setdefault(key, value)
        for key, value in TASK_SIM_ENV_DEFAULTS.get(
                episode_task, {}).items():
            sim_env_values.setdefault(key, value)
        asset_decision, asset_revisions = (
            verify_episode_asset_policy(episode_task)
        )
    except ValueError as exc:
        raise SystemExit(
            f"episode asset-policy verification failed: {exc}"
        ) from exc
    if a.skip_refine and asset_decision.applicable:
        raise SystemExit(
            "--skip-refine is disabled for retail tasks because the target "
            "model and document-only scene filter must be reverified"
        )
    print(json.dumps({
        "asset_policy": "verified",
        "task": episode_task,
        "episode_identity_source": (
            "explicit_validated_trajectory_binding"
            if recordless_replay else "real_episode_metadata"
        ),
        "asset_id": asset_decision.asset_id,
        "asset_status": asset_decision.status,
        "asset_revision": asset_decision.asset_revision,
        "scene_asset_revisions": asset_revisions,
    }, sort_keys=True), flush=True)
    run_id = None
    frame_policy_manifest = None
    if a.metrics_only:
        episode_input = None
    elif a.sim_only:
        episode_input = {
            "schema": RECORDLESS_INPUT_SCHEMA,
            "episode_uid": uid,
            "task_id": episode_task,
            "real_episode_archive_required": False,
        }
    else:
        episode_input = episode_input_signature(epdir)
    al = {}
    zone = None
    place_destination = None
    bag_place_hold_hand = None
    metrics_output_kind = None

    if a.skip_render:
        try:
            frame_policy_manifest = validate_frame_policy_manifest(
                episode_uid=uid,
                task_id=episode_task,
                decision=asset_decision,
                runtime_revision=runtime_source_revision(episode_task),
                episode_input=episode_input,
            )
        except ValueError as exc:
            raise SystemExit(
                f"--skip-render asset provenance rejected: {exc}"
            ) from exc

    if not a.skip_render:
        verify_container_runtime_mounts()
        # This must precede every mutation of shared scene/trajectory/output
        # state.  An orphaned container process does not retain our host flock.
        stop_stale_container_playback()
        runtime_revision = sync_runtime(episode_task)
        runtime_revision_json = json.dumps(
            runtime_revision, sort_keys=True, separators=(",", ":"))
        run_id = uuid.uuid4().hex
        # A failed/non-place render must not inherit another episode's physical evidence.
        clear_container_runtime_outputs()
        trajectory = Path(DEMO) / "playback_traj.npz"
        trajectory_tmp = trajectory.with_name(
            f"playback_traj.{run_id}.tmp.npz")
        try:
            if a.trajectory_input:
                trajectory_source = Path(a.trajectory_input)
                if not trajectory_source.is_absolute():
                    trajectory_source = Path(REPO) / trajectory_source
                copy_validated_trajectory(
                    trajectory_source, trajectory_tmp)
            else:
                r = run(
                    [
                        sys.executable,
                        "real2sim/teleop/extract_episode.py",
                        str(epdir / "record" / "raw_joints.h5"),
                        "--fps",
                        str(FPS),
                        "--out",
                        str(trajectory_tmp),
                    ],
                    timeout=600,
                    cwd=REPO,
                )
                if r.returncode != 0:
                    raise SystemExit(
                        "extract failed:\n"
                        + r.stdout[-500:] + r.stderr[-500:])
            validate_trajectory_npz(trajectory_tmp)
            if a.traj_filter:
                # Generation-layer seam (datagen G2/G3): rewrite the seed
                # trajectory before anything downstream reads it. The
                # grasp-anchor alignment and object placement both derive
                # from this file, so a retargeted trajectory carries its
                # perturbed object pose with it — no separate wiring. The
                # filtered file must pass the exact validator the recorded
                # one passed; a filter error aborts the render.
                fr = run(
                    shlex.split(a.traj_filter) + [str(trajectory_tmp)],
                    timeout=600, cwd=REPO)
                if fr.returncode != 0:
                    raise SystemExit(
                        "traj filter failed:\n"
                        + fr.stdout[-500:] + fr.stderr[-500:])
                validate_trajectory_npz(trajectory_tmp)
            with trajectory_tmp.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(trajectory_tmp, trajectory)
            fsync_directories(trajectory.parent)
            if house_layout_request is not None:
                house_layout_action = (
                    house_layout_canary_hook.transform_playback_action(
                        trajectory, house_layout_request
                    )
                )
                validate_trajectory_npz(trajectory)
            if episode_input is not None and (
                    a.traj_filter or a.trajectory_input):
                episode_input = bind_playback_trajectory(
                    episode_input,
                    trajectory,
                    source=(
                        str(a.trajectory_input)
                        if a.trajectory_input
                        else "generated_by_traj_filter"
                    ),
                )
            if episode_task in BAG_PLACE_TASK_ID_SET:
                bag_place_hold_hand = bag_place_hold_hand_from_trajectory(
                    trajectory)
        except Exception as exc:
            raise SystemExit(f"extract produced invalid trajectory: {exc}") from exc
        finally:
            trajectory_tmp.unlink(missing_ok=True)
        if not a.skip_refine:
            default_init_cmd = (
                EPISODE_INIT_COMMAND_DEFAULTS.get((episode_task, uid))
                or TASK_INIT_COMMAND_DEFAULTS.get(episode_task)
            )
            if episode_task in BAG_PLACE_TASK_ID_SET:
                default_init_cmd = (
                    f"TASK={episode_task} {shlex.quote(sys.executable)} "
                    "real2sim/teleop/init_task.py"
                )
            if not default_init_cmd and episode_task in REGISTRY:
                # init_task.py calls itself "THE single uniform init entry
                # point for every real2sim task" and dispatches by family, but
                # only four tasks and the bag-place set ever reached it: every
                # other registered task fell through to refine_episode.py,
                # which is not an initializer at all. That is why the reproduce
                # queue could only ever run tasks 8/37/40/41 — the other 17
                # shelf tasks would have rendered an uninitialised scene.
                # Registry membership is the same gate init_task itself
                # applies, so this cannot invent a task that has no family.
                default_init_cmd = (
                    f"TASK={episode_task} EPISODE_UID={{episode_uid}} "
                    f"{shlex.quote(sys.executable)} "
                    "real2sim/teleop/init_task.py"
                )
            if default_init_cmd:
                default_init_cmd = default_init_cmd.format(
                    episode_uid=uid,
                    task53_bite_z=(
                        repr(task53_episode[
                            "contact_controls"]["bite_z_m"])
                        if task53_episode is not None else ""
                    ),
                )
            init_cmd = (
                a.init_cmd
                or default_init_cmd
                or f"python real2sim/teleop/refine_episode.py {uid}"
            )
            # Print the command that actually ran. Without this line a job can
            # pass --init-cmd, watch the render produce the default result, and
            # have no way to tell whether the override was ignored or simply
            # made no difference -- which is exactly what happened when three
            # IH_YAW values for task 28 all returned the same object yaw.
            print(f"[init-cmd] {'override' if a.init_cmd else 'default'}: "
                  f"{init_cmd}", flush=True)
            try:
                r = run_command_spec(
                    init_cmd,
                    timeout=1300,
                    cwd=REPO,
                    extra_env={
                        "EPISODE_UID": uid,
                        "REAL2SIM_TASK_ID": episode_task,
                        # ``init_task.py`` also supports standalone callers,
                        # where it owns scene randomization by default.  This
                        # orchestrator owns the same mutation immediately
                        # below so it can validate and archive the applied
                        # plan.  Disable the nested pass here: one episode
                        # must receive exactly one deterministic layout.
                        "REAL2SIM_INIT_SCENE_RANDOMIZATION": "0",
                    },
                )
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                raise SystemExit(
                    f"init command failed: {type(exc).__name__}: {exc}") from exc
            try:
                al = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception as exc:
                raise SystemExit(
                    f"init produced invalid JSON ({type(exc).__name__}): "
                    + (r.stdout + r.stderr)[-400:])
            if (not isinstance(al, dict) or r.returncode != 0
                    or al.get("ok") is not True):
                raise SystemExit(f"init failed: {al}")
            try:
                validate_retail_init_result(
                    al, episode_task, asset_decision
                )
                validate_task29_full_init_result(
                    al,
                    episode_task=episode_task,
                    episode_uid=uid,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"init violated current episode contract: {exc}"
                ) from exc
            if episode_task in BAG_PLACE_TASK_ID_SET:
                if bag_place_hold_hand not in ("l", "r"):
                    raise SystemExit(
                        "bag-place trajectory did not resolve a physical "
                        "hold hand")
                al = dict(al)
                al["hand"] = bag_place_hold_hand
            try:
                al = apply_episode_scene_randomization(
                    al,
                    task_id=episode_task,
                    episode_uid=uid,
                    epdir=epdir,
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as exc:
                raise SystemExit(
                    "episode scene randomization failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                al = apply_episode_kitchen_tableware_randomization(
                    al,
                    task_id=episode_task,
                    episode_uid=uid,
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as exc:
                raise SystemExit(
                    "episode kitchen tableware randomization failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                al = apply_episode_appearance_randomization(
                    al,
                    task_id=episode_task,
                    episode_uid=uid,
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as exc:
                raise SystemExit(
                    "episode appearance randomization failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if house_layout_request is not None:
                try:
                    house_layout_scene = (
                        house_layout_canary_hook.apply_working_scene(
                            SCENE_PATH, house_layout_request
                        )
                    )
                    house_layout_evidence = (
                        house_layout_canary_hook.write_evidence(
                            house_layout_request,
                            action=house_layout_action,
                            scene=house_layout_scene,
                        )
                    )
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    raise SystemExit(
                        "House layout canary application failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                print(
                    "[house-layout-canary] "
                    + json.dumps(
                        house_layout_evidence,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            print("init:", al, flush=True)
        if episode_task not in BAG_PLACE_TASK_ID_SET:
            enforce_runtime_shelf_policy()
        try:
            post_decision, post_revisions = (
                verify_episode_asset_policy(episode_task)
            )
        except ValueError as exc:
            raise SystemExit(
                f"post-init asset-policy verification failed: {exc}"
            ) from exc
        if (
            post_decision != asset_decision
            or post_revisions != asset_revisions
        ):
            raise SystemExit(
                "retail asset identity changed during scene initialization"
            )
        # Recovered benchmark plans are audited only now: the exclusive
        # session is open, the initializer and all frame-zero randomizers have
        # finished, and simulator physics has not started.  A pending plan or
        # a failed measured audit can therefore never reach playback.
        require_recovered_scene_audit(
            al, episode_task=episode_task, episode_uid=uid,
        )
        # Physical in-hand setup: for place the object starts held. playback_video3 seats it between
        # the calibrated collision pads, closes by drive + collision + friction, and releases only
        # when the recorded grip opens. Pick/snack/throw are unchanged.
        hold_env = {}
        if a.metrics_only:
            try:
                metrics_output_kind = metrics_only_output_kind(
                    al, episode_task)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        if al.get("family") == "place":
            try:
                place_task, place_target, zone = validate_place_init_result(
                    al,
                    coffee_table_calibration=coffee_table_calibration,
                )
                if house_layout_request is not None:
                    zone = house_layout_canary_hook.transform_frame_zone(
                        zone, house_layout_request).tolist()
                place_destination = destination_for_task(
                    place_task,
                    coffee_table_calibration=coffee_table_calibration,
                )
                hold_env.update(validate_place_grasp_controls(al))
            except ValueError as exc:
                raise SystemExit(f"invalid canonical place init result: {exc}") from exc
            hh = (al.get("hand") or "").lower()
            hold_env.update({
                "HOLD_MODE": "inhand",
                "HOLD_PRIM": place_target,
                "PLACE_TASK_ID": place_task,
                "PLACE_DESTINATION_KIND": place_destination["kind"],
                "PLACE_DESTINATION_BODY_PATH": place_destination["prim_path"],
                "PLACE_DESTINATION_BOTTOM_Z": str(
                    place_destination["bottom_z"]),
                "EPISODE_UID": uid,
                "PLACE_RUN_ID": run_id,
                "RUNTIME_SOURCE_REVISION": runtime_revision_json,
            })
            hold_env["PLACE_ZONE"] = ",".join(str(value) for value in zone)
            # Articulated-drawer episodes may carry a bounded FK-derived
            # per-episode station correction; forward it so playback records
            # it in place_metrics and the strict layout gate can accept the
            # shifted base (check_success task28_station_correction_m).
            _drawer_info = al.get("drawer") or {}
            _station_correction = _drawer_info.get("station_correction_m")
            if (
                place_task == "28"
                and isinstance(_station_correction, (list, tuple))
                and len(_station_correction) == 2
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    and abs(v) <= 0.12
                    for v in _station_correction)
            ):
                hold_env["T28_STATION_CORRECTION"] = ",".join(
                    str(float(v)) for v in _station_correction)
            if hh in ("l", "r"):
                hold_env["HOLD_HAND"] = hh
            if al.get("surface_z") is not None:
                surface_z = float(al["surface_z"])
                if not math.isfinite(surface_z):
                    raise SystemExit("place init returned non-finite surface_z")
                hold_env["HOLD_SURFACE_Z"] = str(surface_z)
        runtime_identity_env = {
            "REAL2SIM_TASK_ID": episode_task,
            "RUNTIME_SOURCE_REVISION": runtime_revision_json,
        }
        _distractor_track = randomized_distractor_audit_prims(
            al, episode_task)
        if _distractor_track:
            # The randomized clutter is other tasks' interactive props; an
            # instruction-following replay must not touch them.  Playback
            # samples these prims post-settle and at the end and archives a
            # displacement audit next to the video.
            runtime_identity_env["DISTRACTOR_TRACK_PRIMS"] = json.dumps(
                _distractor_track, separators=(",", ":"))
        _group_shift_raw = os.environ.get("REAL2SIM_GROUP_SHIFT_XY_M")
        if _group_shift_raw:
            # The container runtime imports robot_positions/group_shift from
            # usdz/ and must resolve the SAME per-episode shift as the host.
            runtime_identity_env["REAL2SIM_GROUP_SHIFT_XY_M"] = (
                _group_shift_raw)
        contact_trace_required = requires_contact_trace(al, episode_task)
        if contact_trace_required:
            # Pick and in-hand-start place comparisons need the same raw
            # target/pad telemetry.  For place this begins at frame zero; it
            # does not imply that the recording contains a pickup phase.
            # Sensor creation is managed here and never caller-overridable.
            runtime_identity_env["PICK_CONTACT_INSTRUMENTATION"] = "1"
            runtime_identity_env["EPISODE_UID"] = uid
        if asset_decision.asset_id is not None:
            runtime_identity_env["ASSET_ID"] = asset_decision.asset_id
        if asset_decision.asset_revision is not None:
            runtime_identity_env["ASSET_REVISION"] = (
                asset_decision.asset_revision
            )
        try:
            _registry_track_prim = runtime_target_prim(
                episode_task, al)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        runtime_identity_env["REAL2SIM_TARGET_PRIM"] = (
            _registry_track_prim)
        require_runtime_source_unchanged(
            runtime_revision, episode_task, "before simulator playback")
        stop_stale_container_playback()
        runtime_env = dict(runtime_identity_env)
        runtime_env.update(hold_env)
        runtime_env.update(sim_env_values)
        # Full-state capture is a production control, not a tunable simulator
        # knob.  Keep it out of SIM_ENV (where a generated action could
        # silently change the evidence contract), but explicitly forward the
        # reviewed host-side opt-in to the isolated docker playback.  Merely
        # exporting an environment variable on the host is otherwise not
        # enough: ``docker exec`` only receives the allowlisted values built
        # here.
        _full_state_capture = os.environ.get(
            "FULL_STATE_CAPTURE", "0").strip()
        if _full_state_capture not in {"0", "1"}:
            raise SystemExit(
                "FULL_STATE_CAPTURE must be exactly 0 or 1")
        if _full_state_capture == "1":
            _full_state_scope = os.environ.get(
                "FULL_STATE_SCOPE", "target_robot").strip()
            if _full_state_scope not in {
                "all_active", "target_robot", "tracked_usd"
            }:
                raise SystemExit(
                    "FULL_STATE_SCOPE must be all_active, target_robot, "
                    "or tracked_usd")
            runtime_env["FULL_STATE_CAPTURE"] = "1"
            runtime_env["FULL_STATE_SCOPE"] = _full_state_scope
        if "TASK38_INIT_ROBOT_STATE" in os.environ:
            initial_mode = os.environ["TASK38_INIT_ROBOT_STATE"]
            if episode_task != "38" or initial_mode not in {"0", "1"}:
                raise SystemExit("TASK38_INIT_ROBOT_STATE requires task 38 and 0 or 1")
            runtime_env["TASK38_INIT_ROBOT_STATE"] = initial_mode
        _task_owned_arm_initial_delta = os.environ.get(
            "TASK_OWNED_ARM_INITIAL_DELTA_RAD", "").strip()
        try:
            runtime_env.update(task_owned_arm_initial_environment(
                episode_task, _task_owned_arm_initial_delta,
                _full_state_capture))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        initialized_hand = str(al.get("hand", "")).strip().lower()
        if initialized_hand in ("l", "r"):
            runtime_env["ACTIVE_HAND"] = initialized_hand
        if episode_task in BAG_PLACE_TASK_ID_SET:
            runtime_env.update(
                task73_grasp_calibration_environment(uid))
            runtime_env.setdefault("POST_STEPS", "360")
            runtime_env["SCENE_USDA"] = (
                "/geniesim/main/demo_sim/scene_room.usda"
            )
            runtime_env["POLICY_SCENE_USDA"] = (
                "/geniesim/main/demo_sim/scene_room.usda"
            )
        if a.metrics_only:
            runtime_env["METRICS_ONLY"] = "1"
        display_env = " ".join(
            f"{key}={value}" for key, value in sorted(runtime_env.items()))
        print(f"rendering sim 3-cam playback ... ({display_env or 'default'})", flush=True)
        docker_command = ["timeout", "2400", "docker", "exec"]
        for key, value in sorted({
                "CUDA_VISIBLE_DEVICES": _MACHINE_RENDER_GPU,
                **runtime_env}.items()):
            docker_command.extend(["--env", f"{key}={value}"])
        playback_command = (
            "omni_python /geniesim/main/demo_sim/usdz/playback_video3.py"
        )
        if episode_task in BAG_PLACE_TASK_ID_SET:
            playback_command += (
                " --/persistent/physics/enableDeformableBeta=true"
            )
        docker_command.extend([
            CONTAINER_NAME,
            "bash",
            "-ic",
            playback_command,
        ])
        r = run(docker_command, timeout=2500)
        for _l in (r.stdout or "").splitlines():      # surface physical grasp diagnostics
            if any(
                marker in _l
                for marker in (
                    "[hold]",
                    "[grasp]",
                    "[v3 settle]",
                    "[place]",
                    "[task56-contact]",
                    "[task53-provenance]",
                    "[task53-visual]",
                    "[task53-friction]",
                    "[task53-torsion]",
                    "[task59-torsion]",
                    "[pick-target-settle]",
                    "[pick-target]",
                    "[pick-trace]",
                    "[cushion-contact]",
                    "[glasses-case-visual]",
                    "[task73]",
                )
            ):
                print("  " + _l.strip(), flush=True)
        done_line = any(
            re.fullmatch(r"done [0-9]+ frames -> .+", line.strip())
            for line in r.stdout.splitlines()
        )
        if r.returncode != 0 or not done_line:
            # A Python-level failure inside the renderer makes libomni.graph's
            # atexit handler segfault on the way out, and breakpad then writes
            # ~10 KB of stack frames to stderr AFTER the traceback. Tailing raw
            # stderr therefore reports the crash dump and evicts the exception
            # that caused it — task 41 failed a real physical gate for four
            # runs while its logs showed nothing but symbol names. Keep the
            # head of the dump (signal, address, faulting module) so a genuine
            # mid-run crash is still diagnosable.
            raise SystemExit(
                "sim render failed:\n"
                + (r.stdout or "")[-4000:]
                + _without_crash_dump(r.stderr or "")[-8000:]
            )
        require_runtime_source_unchanged(
            runtime_revision, episode_task, "after simulator playback")
        if not a.metrics_only:
            completed = completed_frame_count()
            frame_policy_manifest = write_frame_policy_manifest(
                episode_uid=uid,
                task_id=episode_task,
                decision=asset_decision,
                runtime_revision=runtime_revision,
                frame_count=completed,
                episode_input=episode_input,
            )

    out = Path(a.out) if a.out else Path(REPO) / "dataset" / "sim_episodes_repro" / f"{tag}_compare3.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    expected_sidecar = {
        "expected_run_id": run_id,
        "expected_task_id": al.get("task"),
        "expected_episode_uid": uid,
        "expected_target_body_path": al.get("track_prim"),
        "expected_destination_kind": (
            place_destination or {}).get("kind"),
        "expected_destination_body_path": (
            place_destination or {}).get("prim_path"),
        "expected_destination_bottom_z": (
            place_destination or {}).get("bottom_z"),
        "expected_zone": zone,
    }
    if a.metrics_only:
        if metrics_output_kind == "task29":
            copied = copy_task29_sidecars(
                out, expected_episode_uid=uid)
            if len(copied) != 2:
                raise SystemExit(
                    "metrics-only run produced "
                    f"{len(copied)}/2 small Task 29 sidecars")
        elif metrics_output_kind == "place":
            copied = copy_place_sidecars(out, **expected_sidecar)
            if len(copied) != 2:
                raise SystemExit(
                    f"metrics-only run produced "
                    f"{len(copied)}/2 place sidecars")
            copied_trace = copy_pick_contact_trace(
                out,
                expected_task_id=episode_task,
                expected_episode_uid=uid,
                expected_runtime_revision=runtime_revision,
                expected_target_body_path=_registry_track_prim,
                expected_frame_count=completed_frame_count(),
                expected_furniture_center_xy_m=(
                    house_layout_expected_furniture_center_xy_m),
            )
            if copied_trace is None:
                raise SystemExit(
                    "metrics-only place run produced no contact trace")
        elif metrics_output_kind == "pick":
            copied = copy_pick_contact_trace(
                out,
                expected_task_id=episode_task,
                expected_episode_uid=uid,
                expected_runtime_revision=runtime_revision,
                expected_target_body_path=_registry_track_prim,
                expected_frame_count=completed_frame_count(),
                expected_furniture_center_xy_m=(
                    house_layout_expected_furniture_center_xy_m),
            )
            if copied is None:
                raise SystemExit(
                    "metrics-only pick run produced no contact trace")
        else:
            raise SystemExit(
                "metrics-only run has no resolved physical output kind")
        copy_distractor_audit(
            out,
            expected_task_id=episode_task,
            expected_episode_uid=uid,
            expected_prims=_distractor_track,
        )
        return
    if a.skip_render:
        # Reused shared frames have no run manifest tying them to a sidecar.
        # Keep this mode visual-only and recoverably archive any prior evidence
        # attached to the output instead of silently relabeling it.
        archive_output_sidecars(out, "skip_render_visual_only")
    else:
        archive_output_sidecars(out, "replaced_by_new_run")

    sim_frames, T = validated_sim_frames()

    if a.sim_only:
        width = 1192

        def frames():
            for t in range(T):
                sims = [
                    require_decoded_frame(
                        cv2.imread(sim_frames[camera][t]),
                        source="sim",
                        camera=camera,
                        frame=t,
                    )
                    for camera in SIM_DIRS
                ]
                head = fit_box(sims[0], 768, 480)
                hands = np.vstack([
                    fit_box(sims[1], 424, 240),
                    fit_box(sims[2], 424, 240),
                ])
                hdr = band(
                    f"SIM ONLY  {tag}   head | left/right hand cameras   "
                    f"t={t/FPS:5.2f}s  frame {t}/{T}",
                    width,
                )
                yield np.vstack([hdr, np.hstack([head, hands])])
    else:
        # the exact grid extract_episode used, in raw stream timestamps
        rec = convert_episode(str(epdir / "record" / "raw_joints.h5"))
        ts = rec["ts"].astype(np.float64)
        dur = (ts[-1] - ts[0]) / 1e9
        n = max(2, int(dur * FPS))
        grid_ns = ts[0] + np.linspace(0, dur, n) * 1e9
        if len(grid_ns) != T:
            raise SystemExit(
                "real/sim trajectory length mismatch: "
                f"real grid={len(grid_ns)}, sim={T}")

        real_iters = [
            resample_to_grid(camera_frames(epdir, cam, None), grid_ns)
            for cam in REAL_CAMS
        ]
        width = CELL_W * 3

        def frames():
            for t in range(len(grid_ns)):
                reals = [
                    next_decoded_frame(
                        iterator, source="real", camera=camera, frame=t)
                    for iterator, camera in zip(real_iters, REAL_CAMS)
                ]
                row_r = np.hstack([fit_cell(im) for im in reals])
                sims = [
                    require_decoded_frame(
                        cv2.imread(sim_frames[camera][t]),
                        source="sim",
                        camera=camera,
                        frame=t,
                    )
                    for camera in SIM_DIRS
                ]
                row_s = np.hstack([fit_cell(im) for im in sims])
                hdr = band(
                    f"REAL  {tag}   head_color | hand_left | hand_right"
                    f"   t={t/FPS:5.2f}s  frame {t}/{len(grid_ns)}",
                    width,
                )
                mid = band(
                    "SIM   scene_room playback (Head/Left/Right_Camera)",
                    width,
                )
                yield np.vstack([hdr, row_r, mid, row_s])

    video_tmp = out.with_name(
        f".{out.stem}.{run_id or uuid.uuid4().hex}.tmp.mp4")
    try:
        nfr = write_mp4(frames(), video_tmp, FPS, None)
        if nfr != T:
            raise SystemExit(
                f"comparison video truncated: wrote {nfr} of {T} frames")
        with video_tmp.open("r+b") as stream:
            os.fsync(stream.fileno())
        if not a.skip_render:
            require_runtime_source_unchanged(
                runtime_revision, episode_task, "before comparison publication")
        os.replace(video_tmp, out)
        fsync_directories(out.parent)
    finally:
        video_tmp.unlink(missing_ok=True)
    print(f"wrote {out}  ({nfr} frames, {nfr/FPS:.1f}s)")
    if frame_policy_manifest is None:
        raise SystemExit("comparison has no verified frame asset provenance")
    if not a.skip_render:
        require_runtime_source_unchanged(
            runtime_revision, episode_task, "before provenance publication")
    publish_output_asset_manifest(out, frame_policy_manifest)
    if a.sim_only:
        publish_sim_only_manifest(
            out,
            frame_manifest=frame_policy_manifest,
            episode_uid=uid,
            task_id=episode_task,
            run_id=run_id,
            frame_count=T,
        )
    if not a.skip_render and al.get("family") == "place":
        copied = copy_place_sidecars(out, **expected_sidecar)
        if len(copied) != 2:
            raise SystemExit(
                f"place render produced {len(copied)}/2 validated sidecars")
    elif (
        not a.skip_render
        and al.get("family") == "throw"
        and al.get("task") == "29"
    ):
        copied = copy_task29_sidecars(
            out, expected_episode_uid=uid)
        if len(copied) != 2:
            raise SystemExit(
                "small Task 29 render produced "
                f"{len(copied)}/2 validated sidecars")
    if not a.skip_render and contact_trace_required:
        copied_trace = copy_pick_contact_trace(
            out,
            expected_task_id=episode_task,
            expected_episode_uid=uid,
            expected_runtime_revision=runtime_revision,
            expected_target_body_path=_registry_track_prim,
            expected_frame_count=T,
            expected_furniture_center_xy_m=(
                house_layout_expected_furniture_center_xy_m),
        )
        if copied_trace is None:
            raise SystemExit(
                "pick/place render produced no contact trace")
    elif a.skip_render:
        print("skip-render is visual-only; physical sidecars were not published", flush=True)
    if not a.skip_render:
        copy_distractor_audit(
            out,
            expected_task_id=episode_task,
            expected_episode_uid=uid,
            expected_prims=_distractor_track,
        )


if __name__ == "__main__":
    main()
