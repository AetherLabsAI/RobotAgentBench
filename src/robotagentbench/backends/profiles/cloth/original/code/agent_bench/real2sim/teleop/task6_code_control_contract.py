"""Code-control runtime details owned by Task-6 benchmark tasks.

Each task can optionally initialize from a real pick episode.  Its legacy
recording label is provenance, not a second identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_positions import (
    COFFEE_TABLE_CURRENT_CENTER_XY_M,
    COFFEE_TABLE_CURRENT_SIZE_XYZ_M,
    COFFEE_TABLE_CURRENT_TOP_M,
    robot_position,
)
from coat_rack_contract import (
    AABB_MAX_XYZ_M as COAT_RACK_AABB_MAX_XYZ_M,
    AABB_MIN_XYZ_M as COAT_RACK_AABB_MIN_XYZ_M,
    HANG_BRANCH_JUNCTION_XYZ_M,
    HANG_BRANCH_TIP_CENTER_XYZ_M,
    PRIM_PATH as COAT_RACK_PATH,
)


SCHEMA = "real2sim.task6_code_control_contract.v1"
FPS = 20.0
TARGET_COUNT = 50


@dataclass(frozen=True)
class PlaceMode:
    benchmark_subtask: str
    runtime_task_id: str
    label: str
    real_data_reference_id: str
    source_episode_uid: str
    target_path: str
    hand: str
    destination_kind: str
    destination_path: str
    destination_zone: tuple[float, float, float, float, float]
    destination_bottom_z: float
    robot_station: str
    furniture_path: str
    furniture_aabb_min_xyz_m: tuple[float, float, float]
    furniture_aabb_max_xyz_m: tuple[float, float, float]
    destination_aabb_min_xyz_m: tuple[float, float, float] | None = None
    destination_aabb_max_xyz_m: tuple[float, float, float] | None = None
    source_window_before_close_frames: int | None = None
    source_window_after_close_frames: int | None = None
    grasp_up_axis: int | None = None
    grip_width_m: float | None = None
    grip_release_advance_frames: int = 0
    max_bilateral_contact_dropout_frames: int = 0
    attach_rms_tolerance_m: float = 0.025
    post_release_direct_retreat_delay_frames: int = 0
    post_release_direct_retreat_frames: int | None = None
    source_relation: str = "reverse_of_pick_from_same_destination"


_coffee_table_zone = (
    float(COFFEE_TABLE_CURRENT_CENTER_XY_M[0]),
    float(COFFEE_TABLE_CURRENT_CENTER_XY_M[1]),
    float(COFFEE_TABLE_CURRENT_SIZE_XYZ_M[0]) / 2.0,
    float(COFFEE_TABLE_CURRENT_SIZE_XYZ_M[1]) / 2.0,
    float(COFFEE_TABLE_CURRENT_TOP_M),
)


MODES = {
    "6.5": PlaceMode(
        benchmark_subtask="6.5",
        runtime_task_id="benchmark_tidy_sofa_6_5",
        label=(
            "benchmark Task 6.5 (green Sales Order bill to canonical "
            "CoffeeTable drawer)"
        ),
        real_data_reference_id="28",
        source_episode_uid="05e907e9-643c-4ca4-aae5-7fc30baddfaa",
        target_path="/World/Assets/Note",
        hand="l",
        destination_kind="bin",
        destination_path="/World/Assets/CoffeeTable/black_box_1",
        destination_zone=(
            -1.6853036226080885,
            -0.5176448646303313,
            0.120,
            0.080,
            0.3373112690122057,
        ),
        destination_bottom_z=0.203662484,
        # The task-owned producer uses the already validated two-hand drawer
        # choreography at this station. Furniture remains at the canonical
        # global pose; source recording labels do not affect task identity.
        robot_station="coffee_table_drawer_bill_two_hand",
        furniture_path="/World/Assets/CoffeeTable",
        furniture_aabb_min_xyz_m=(
            -2.349999995986884,
            -0.7210435941542119,
            -6.705522756180538e-09,
        ),
        furniture_aabb_max_xyz_m=(
            -1.550000005801256,
            0.4789564280510332,
            0.45,
        ),
        destination_aabb_min_xyz_m=(
            -1.8405820086964546,
            -0.6297119681218109,
            0.19999322958861535,
        ),
        destination_aabb_max_xyz_m=(
            -1.5300252365197225,
            -0.40557776113885174,
            0.3373112690122057,
        ),
        grip_width_m=0.011,
        max_bilateral_contact_dropout_frames=10,
        attach_rms_tolerance_m=0.030,
        source_relation=(
            "small_Task_28_target_identity_with_reviewed_benchmark_6_15_"
            "two_hand_drawer_choreography_reference"
        ),
    ),
    "6.7": PlaceMode(
        benchmark_subtask="6.7",
        runtime_task_id="benchmark_tidy_sofa_6_7",
        label=(
            "benchmark tidy_sofa 6.7 (hat onto canonical coat-rack branch; "
            "direct real-reference action)"
        ),
        real_data_reference_id="70",
        source_episode_uid="69d6f246-1d00-451e-b52a-dad5ed311275",
        target_path="/World/Assets/BaseballCap",
        hand="l",
        # The generic place runtime only accepts surface/bin metadata.  The
        # Task-6 scorer replaces horizontal-surface semantics with strict
        # branch-overlap, release, separation and stable-hang gates.
        destination_kind="surface",
        destination_path=COAT_RACK_PATH,
        destination_zone=(
            (HANG_BRANCH_JUNCTION_XYZ_M[0]
             + HANG_BRANCH_TIP_CENTER_XYZ_M[0]) / 2.0,
            (HANG_BRANCH_JUNCTION_XYZ_M[1]
             + HANG_BRANCH_TIP_CENTER_XYZ_M[1]) / 2.0,
            0.20,
            0.20,
            HANG_BRANCH_JUNCTION_XYZ_M[2],
        ),
        destination_bottom_z=HANG_BRANCH_JUNCTION_XYZ_M[2],
        robot_station="coat_rack_hang",
        furniture_path=COAT_RACK_PATH,
        furniture_aabb_min_xyz_m=COAT_RACK_AABB_MIN_XYZ_M,
        furniture_aabb_max_xyz_m=COAT_RACK_AABB_MAX_XYZ_M,
        source_relation="direct_real_place_to_same_destination",
    ),
    "6.3": PlaceMode(
        benchmark_subtask="6.3",
        runtime_task_id="benchmark_tidy_sofa_6_3",
        label=(
            "benchmark tidy_sofa 6.3 (book to side table; "
            "code-control with optional real reference)"
        ),
        # Recording label 60 picks a wallet from this exact SmallCoffeeTable. Its
        # reverse supplies destination/station geometry; fresh physics must
        # still prove that the substituted Book can be clamped and released.
        real_data_reference_id="60",
        source_episode_uid="126623c7-9aaa-46c2-b40c-5e4947b198bc",
        target_path="/World/Assets/Book",
        hand="r",
        destination_kind="surface",
        destination_path="/World/Assets/SmallCoffeeTable",
        destination_zone=(
            -2.115033684951505,
            -1.9866492246218145,
            0.4835373656735427,
            0.6354666481038845,
            0.4572000012693597,
        ),
        destination_bottom_z=0.4572000012693597,
        robot_station="sidetable_pick",
        furniture_path="/World/Assets/SmallCoffeeTable",
        furniture_aabb_min_xyz_m=(
            -2.5985710506250475,
            -2.622115872725699,
            0.006988783010899189,
        ),
        furniture_aabb_max_xyz_m=(
            -1.6314963192779621,
            -1.35118257651793,
            0.4572000012693597,
        ),
        # The literal reverse of the source pre-grasp approach keeps touching
        # the Book for 22 frames after opening.  Keep its first 12 frames so
        # the upright Book is guided down to the tabletop, then withdraw to
        # the source-start posture over eight frames.  This only replaces the
        # tail of the source pre-grasp approach.
        post_release_direct_retreat_delay_frames=12,
        post_release_direct_retreat_frames=8,
    ),
    "6.9": PlaceMode(
        benchmark_subtask="6.9",
        runtime_task_id="benchmark_tidy_sofa_6_9",
        label=(
            "benchmark tidy_sofa 6.9 (remote to CoffeeTable; "
            "code-control with optional real reference)"
        ),
        real_data_reference_id="61",
        source_episode_uid="0048dd8e-f700-40e1-9192-8f6b85318a06",
        target_path="/World/Assets/Remote",
        hand="r",
        destination_kind="surface",
        destination_path="/World/Assets/CoffeeTable",
        destination_zone=_coffee_table_zone,
        destination_bottom_z=float(COFFEE_TABLE_CURRENT_TOP_M),
        robot_station="table_pick",
        furniture_path="/World/Assets/CoffeeTable",
        furniture_aabb_min_xyz_m=(
            -2.349999995986884,
            -0.7210435941542119,
            -6.705522756180538e-09,
        ),
        furniture_aabb_max_xyz_m=(
            -1.550000005801256,
            0.4789564280510332,
            0.45,
        ),
        # PCA axis 2 is the 19.6 mm thickness.  Keep it world-up so the pads
        # clamp the 47.55 mm body width and the Remote releases face-up.
        grasp_up_axis=2,
        # Select the reviewed PCA box's 47.55 mm body cross-section and route
        # it through the settled finite-pad solver.  A 0.05 mm inward margin
        # avoids exceeding the live envelope through float composition.
        grip_width_m=0.0475,
    ),
    "6.10": PlaceMode(
        benchmark_subtask="6.10",
        runtime_task_id="benchmark_tidy_sofa_6_10",
        label=(
            "benchmark tidy_sofa 6.10 (remote to TV cabinet; "
            "code-control with optional real reference)"
        ),
        real_data_reference_id="58",
        source_episode_uid="001648fd-8d51-47cc-ad71-2155d9152e89",
        target_path="/World/Assets/Remote",
        hand="l",
        destination_kind="surface",
        destination_path="/World/Assets/TvTable",
        # Current canonical TvTable AABB: 0.30 x 1.68 x 0.45 m.
        destination_zone=(
            -3.7097067168016433,
            -0.2985062975457392,
            0.15,
            0.84,
            0.4500000171283664,
        ),
        destination_bottom_z=0.4500000171283664,
        robot_station="tv_cabinet_pick",
        furniture_path="/World/Assets/TvTable",
        furniture_aabb_min_xyz_m=(
            -3.859706718931998,
            -1.1385062473688605,
            4.741336458971253e-12,
        ),
        furniture_aabb_max_xyz_m=(
            -3.5597067146712886,
            0.5414936522773821,
            0.4500000171283664,
        ),
        grasp_up_axis=2,
        grip_width_m=0.0475,
        # In the literal reverse, bilateral contact ends 55 frames before the
        # recorded grip command opens.  Advance only that release channel so
        # the command follows the physical end of the carry; all kinematics
        # remain the exact source inverse.
        grip_release_advance_frames=55,
    ),
    "6.15": PlaceMode(
        benchmark_subtask="6.15",
        runtime_task_id="benchmark_tidy_sofa_6_15",
        label=(
            "benchmark tidy_sofa 6.15 (wallet to CoffeeTable drawer; "
            "code-control with optional real reference)"
        ),
        real_data_reference_id="92",
        source_episode_uid="066de736-3628-48c7-aeba-598ff23344f6",
        target_path="/World/Assets/Wallet",
        hand="l",
        destination_kind="bin",
        destination_path="/World/Assets/CoffeeTable/black_box_1",
        destination_zone=(
            -1.6853036226080885,
            -0.5176448646303313,
            0.120,
            0.080,
            0.3373112690122057,
        ),
        destination_bottom_z=0.203662484,
        robot_station="coffee_table_drawer_wallet",
        furniture_path="/World/Assets/CoffeeTable",
        furniture_aabb_min_xyz_m=(
            -2.349999995986884,
            -0.7210435941542119,
            -6.705522756180538e-09,
        ),
        furniture_aabb_max_xyz_m=(
            -1.550000005801256,
            0.4789564280510332,
            0.45,
        ),
        # Start with only a 20 mm visible lip.  The right hand physically
        # pulls this dynamic drawer before the left hand places the wallet,
        # then pushes it back to the same almost-closed pose.
        destination_aabb_min_xyz_m=(
            -1.8405820086964546,
            -0.6297119681218109,
            0.19999322958861535,
        ),
        destination_aabb_max_xyz_m=(
            -1.5300252365197225,
            -0.40557776113885174,
            0.3373112690122057,
        ),
        # Keep the complete grasp-to-carry branch plus two seconds of the
        # original approach.  Reversing 77 seconds of pre-grasp search would
        # add no placement behavior and would needlessly dominate production.
        source_window_before_close_frames=40,
        source_window_after_close_frames=240,
        # Route the 40 mm filled Wallet thickness through the settled
        # finite-pad solver; the generic four-step gap sample over-closes the
        # linkage after it reaches steady state.
        grip_width_m=0.0395,
        # The 165 x 130 mm filled Wallet briefly loads the drawer lip while
        # entering.  The accepted canary has 95.8% opposed bilateral evidence
        # across the carry, with a longest raw-contact dropout of seven frames
        # and 27.5 mm object-tip RMS.  Keep bounded room for that compliant
        # contact event without treating it as a second grasp.
        max_bilateral_contact_dropout_frames=10,
        attach_rms_tolerance_m=0.030,
    ),
}


def mode(benchmark_subtask: str) -> PlaceMode:
    try:
        return MODES[str(benchmark_subtask)]
    except KeyError as exc:
        raise ValueError(
            "Task-6 code-control mode must be one of "
            f"{sorted(MODES)}, got {benchmark_subtask!r}"
        ) from exc


def station_pose(benchmark_subtask: str) -> dict[str, object]:
    selected = mode(benchmark_subtask)
    pose = robot_position(selected.robot_station)
    return {
        "station": pose.key,
        "base_xy_m": tuple(float(value) for value in pose.xy_m),
        "yaw_deg": float(pose.yaw_deg),
        "quat_wxyz": tuple(float(value) for value in pose.quat_wxyz),
    }


__all__ = [
    "FPS",
    "MODES",
    "PlaceMode",
    "SCHEMA",
    "TARGET_COUNT",
    "mode",
    "station_pose",
]
