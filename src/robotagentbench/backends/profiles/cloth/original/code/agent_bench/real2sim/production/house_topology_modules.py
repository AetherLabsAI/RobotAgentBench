"""Canonical interaction-module geometry for generated House layouts.

This is scene/layout structure, not a benchmark task registry.  The final v12
catalog and its generator share these definitions so no superseded topology
candidate catalog is required at runtime.
"""

from __future__ import annotations

from typing import Any

MODULES: dict[str, dict[str, Any]] = {
    "coffee_table_cluster": {
        "label": "coffee table operation zone",
        "canonical_frame": "coffee_table",
        "canonical_pivot_xy_m": [-1.9500000009, -0.1210435831],
        "fixture_size_xy_m": [0.889, 1.513041],
        "footprint_bounds_xy_m": [-0.489, -0.913041, 0.4, 0.6],
        "footprint_source": "canonical_usd_composed_world_bbox_v1",
        "station_offset_m": 1.55,
        "allowed_space_kinds": ["living"],
        "source_prim_roots": [
            "/World/Assets/CoffeeTable",
            "/World/Assets/TableclothBasket",
            "/World/Assets/TrashBinTeal",
            "/World/Assets/TrashBinRed",
        ],
        "runtime_readiness": "current_cluster_overlay_available",
    },
    "sofa_cluster": {
        "label": "sofa operation zone",
        "canonical_frame": "sofa",
        "canonical_pivot_xy_m": [0.05, -0.871043533],
        "fixture_size_xy_m": [0.800357, 2.70],
        "footprint_bounds_xy_m": [-0.400357, -1.35, 0.4, 1.35],
        "footprint_source": "canonical_usd_composed_world_bbox_v1",
        "station_offset_m": 2.00,
        "allowed_space_kinds": ["living"],
        "source_prim_roots": [
            "/World/Assets/Sofa", "/World/Assets/SeatOverlay",
            "/World/Assets/SeatRiser",
        ],
        "runtime_readiness": "current_cluster_overlay_available",
    },
    "side_table_cluster": {
        "label": "side table operation zone",
        "canonical_frame": "side_table",
        "canonical_pivot_xy_m": [-2.11505, -1.98665],
        "fixture_size_xy_m": [1.428572, 1.270934],
        "footprint_bounds_xy_m": [-0.483522, -0.635466, 0.94505, 0.635468],
        "footprint_source": "canonical_usd_composed_world_bbox_v1",
        "station_offset_m": 1.05,
        "allowed_space_kinds": ["living"],
        "source_prim_roots": [
            "/World/Assets/SmallCoffeeTable", "/World/Assets/NightStand",
        ],
        "runtime_readiness": "current_cluster_overlay_available",
    },
    "tv_cluster": {
        "label": "television cabinet zone",
        "canonical_frame": "tv_cabinet",
        "canonical_pivot_xy_m": [-3.7097067168, -0.2985062975],
        "fixture_size_xy_m": [0.300002, 1.68],
        "footprint_bounds_xy_m": [-0.150001, -0.84, 0.150001, 0.84],
        "footprint_source": "canonical_usd_composed_world_bbox_v1",
        "station_offset_m": 1.45,
        "allowed_space_kinds": ["living"],
        "source_prim_roots": ["/World/Assets/TvTable", "/World/Assets/Tv"],
        "runtime_readiness": "current_cluster_overlay_available",
    },
    "snack_shelf_cluster": {
        "label": "snack shelf operation zone",
        "canonical_frame": "snack_shelf",
        "canonical_pivot_xy_m": [-2.369021151, 2.2529566045],
        # The baseline keeps shelf inventory inactive, so the bare shelf bbox
        # is too narrow for a live task session.  This envelope activates all
        # 26 Sheet/session-declared ShelfItems and Tier2Items in an anonymous
        # layer and unions them with the shelf and collision overlays.
        "fixture_size_xy_m": [1.043982, 0.500002],
        "footprint_bounds_xy_m": [-0.543979, -0.250001, 0.500003, 0.250001],
        "footprint_source": "canonical_usd_all_session_shelf_inventory_bbox_v1",
        "station_offset_m": 0.95,
        "allowed_space_kinds": ["storage"],
        "source_prim_roots": [
            "/World/Assets/ShelvingUnit", "/World/Assets/Tier2Overlay",
            "/World/Assets/Tier3Overlay", "/World/Assets/ShelfItems",
            "/World/Assets/Tier2Items",
        ],
        "runtime_readiness": (
            "all_session_inventory_envelope_audited_overlay_replay_pending"
        ),
    },
    "tote_cluster": {
        "label": "packing tote zone",
        "canonical_frame": "tote",
        "canonical_pivot_xy_m": [2.785, -0.135],
        # A live Family-5 session activates /World/Task73.  Its measured
        # canonical XY bounds relative to this pivot are approximately
        # [-0.220385, -0.209057, 0.222148, 0.847922].  Union that task asset
        # with the 0.70 x 0.55 m generated support so wall placement cannot
        # leave the active tote outside the room.
        "fixture_size_xy_m": [0.70, 1.122924],
        "footprint_bounds_xy_m": [-0.35, -0.275001, 0.35, 0.847923],
        "footprint_source": "canonical_usd_active_task73_bbox_v1",
        "station_offset_m": 0.90,
        "allowed_space_kinds": ["dining"],
        "source_prim_roots": [
            "/World/Task73",
            "/World/HouseTopologyModules/TotePod",
        ],
        "asset_materialization": {
            "reference_path": (
                "arena_scene/v1_2/demo_sim/house_task_assets/task73/"
                "tote_bag/usd/tote_bag_manual_skin_v21_zup.usd"
            ),
            "stage_bundle_root": (
                "arena_scene/v1_2/demo_sim/house_task_assets/task73/tote_bag"
            ),
            "required_files": [
                "usd/tote_bag_manual_skin_v21_zup.usd",
                "textures/leather_basecolor.png",
                "textures/leather_roughness.png",
                "textures/leather_normal.png",
            ],
            "required_file_sizes_bytes": {
                "usd/tote_bag_manual_skin_v21_zup.usd": 808987,
                "textures/leather_basecolor.png": 10775558,
                "textures/leather_roughness.png": 25415,
                "textures/leather_normal.png": 19529485,
            },
            "reference_default_prim": "/World",
            "module_root_path": "/World/HouseTopologyModules/TotePod",
            "placement_prim_path": "/World/HouseTopologyModules/TotePod/Tote",
            "instance_prim_path": "/World/HouseTopologyModules/TotePod/Tote/Asset",
            # Match the accepted Task-73 tote's physical support plane.  A
            # 0.90 m top lets the deformable tote settle about 20 mm below its
            # canonical working height and makes otherwise accepted place
            # actions land below the strict bin-bottom gate.
            "asset_local_translate_m": [0.0, 0.0, 0.92],
            "asset_local_scale_xyz": [1.35, 1.50, 1.05],
            "asset_bbox_center_xy_m": [0.000653, 0.000031],
            "asset_local_yaw_deg": 0.0,
            "support_prim_path": "/World/HouseTopologyModules/TotePod/SupportCounter",
            "support_size_xyz_m": [0.70, 0.55, 0.92],
            "selected_asset": "task73_tote_bag_manual_skin_v21_zup",
            "selection_evidence": "Task 73 accepted tote revision v97-endon-seat080",
            "task_variant_policy": (
                "task-owned scene may activate /World/Task73 for physics replay"
            ),
            "render_materialization_physics": "disabled_until_same_action_replay",
        },
        "runtime_readiness": (
            "task73_active_envelope_audited_overlay_replay_pending"
        ),
    },
    "coat_rack_cluster": {
        "label": "coat rack operation zone",
        "canonical_frame": "coat_rack",
        "canonical_pivot_xy_m": [0.0, 2.02],
        "fixture_size_xy_m": [0.888588, 0.874991],
        "footprint_bounds_xy_m": [-0.50114, -0.393375, 0.387448, 0.481616],
        "footprint_source": "canonical_usd_composed_world_bbox_v1",
        "station_offset_m": 0.95,
        "allowed_space_kinds": ["entry"],
        "source_prim_roots": ["/World/Assets/CoatRack"],
        "runtime_readiness": "station_retarget_required",
    },
    "dining_table_cluster": {
        "label": "dining table operation zone",
        "canonical_frame": "dining_table",
        "canonical_pivot_xy_m": [2.70, -0.86],
        "fixture_size_xy_m": [0.90, 2.80],
        "footprint_bounds_xy_m": [-0.45, -1.40, 0.45, 1.40],
        "footprint_source": "canonical_usd_counter_top_and_base_bbox_v1",
        "station_offset_m": 2.05,
        "allowed_space_kinds": ["dining"],
        "source_prim_roots": [
            "/World/Background/Shell/CounterTop_17",
            "/World/Background/Shell/CounterBase_16",
            "/World/Assets/Task108Plate", "/World/Assets/Task108Placemat",
        ],
        "runtime_readiness": "must_detach_appliances_before_overlay",
    },
    "coffee_machine_cluster": {
        "label": "coffee machine counter pod",
        "canonical_frame": "coffee_machine",
        "canonical_pivot_xy_m": [2.914, -0.099],
        "fixture_size_xy_m": [0.90, 0.75],
        "footprint_bounds_xy_m": [-0.45, -0.375, 0.45, 0.375],
        "footprint_source": "generated_support_exact_dimensions_v1",
        "station_offset_m": 0.95,
        "allowed_space_kinds": ["kitchen"],
        "source_prim_roots": [
            "/World/Task75",
            "/World/HouseTopologyModules/CoffeeMachineSupport",
        ],
        "support_materialization": {
            "root_prim_path": "/World/HouseTopologyModules/CoffeeMachineSupport",
            "support_prim_path": "/World/HouseTopologyModules/CoffeeMachineSupport/Counter",
            "size_xyz_m": [0.90, 0.75, 0.90]
        },
        "runtime_readiness": "generated_support_counter_required",
    },
    "air_fryer_cluster": {
        "label": "air fryer counter pod",
        "canonical_frame": "air_fryer",
        "canonical_pivot_xy_m": [3.55, 0.55],
        "fixture_size_xy_m": [0.90, 0.75],
        "footprint_bounds_xy_m": [-0.45, -0.375, 0.45, 0.375],
        "footprint_source": "generated_support_exact_dimensions_v1",
        "station_offset_m": 0.95,
        "allowed_space_kinds": ["kitchen"],
        "source_prim_roots": ["/World/HouseTopologyModules/AirFryerPod"],
        "asset_materialization": {
            "reference_path": (
                "real2sim/production/assets/kitchen_task_appliances/"
                "air_fryer_chefman_task63/usd/"
                "air_fryer_chefman_task_compatible_photo_v8_repro_v1.usda"
            ),
            "required_files": [
                "air_fryer_chefman_task_compatible_photo_v8_repro_v1.usda",
                "air_fryer_chefman_photo_physics_v8_repro_v1.usda",
                "air_fryer_chefman_photo_v8.usda",
                "airfryer_touch_panel_v8.png",
            ],
            "required_file_sizes_bytes": {
                "air_fryer_chefman_task_compatible_photo_v8_repro_v1.usda": 236,
                "air_fryer_chefman_photo_physics_v8_repro_v1.usda": 149892,
                "air_fryer_chefman_photo_v8.usda": 470379,
                "airfryer_touch_panel_v8.png": 19040,
            },
            "reference_default_prim": "/AirFryer",
            "module_root_path": "/World/HouseTopologyModules/AirFryerPod",
            "placement_prim_path": (
                "/World/HouseTopologyModules/AirFryerPod/AirFryer"
            ),
            "instance_prim_path": (
                "/World/HouseTopologyModules/AirFryerPod/AirFryer/Asset"
            ),
            "asset_local_translate_m": [0.0, 0.0, 0.90],
            "asset_bbox_center_xy_m": [0.054, 0.0],
            "asset_local_yaw_deg": -90.0,
            "support_prim_path": (
                "/World/HouseTopologyModules/AirFryerPod/SupportCounter"
            ),
            "support_size_xyz_m": [0.90, 0.75, 0.90],
            "selected_asset": "chefman_photo_v8_task_compatible_repro",
            "selection_evidence": (
                "Task 66 current reproduction wrapper; Task 63 oldparams "
                "wrapper composes the same Chefman photo-v8 base"
            ),
            "task_variant_policy": (
                "task-owned scene may replace the default wrapper before "
                "same-action physics replay"
            ),
            "render_materialization_physics": "disabled_until_same_action_replay",
        },
        "runtime_readiness": (
            "chefman_photo_v8_visual_bound_physics_replay_pending"
        ),
    },
    "microwave_cluster": {
        "label": "microwave counter pod",
        "canonical_frame": "microwave",
        "canonical_pivot_xy_m": [2.3878, -0.643],
        "fixture_size_xy_m": [1.00, 0.80],
        "footprint_bounds_xy_m": [-0.50, -0.40, 0.50, 0.40],
        "footprint_source": "generated_support_exact_dimensions_v1",
        "station_offset_m": 1.00,
        "allowed_space_kinds": ["kitchen"],
        "source_prim_roots": [
            "/World/Assets/DiningTableMicrowave",
            "/World/Assets/MicrowaveTakePlate",
            "/World/Assets/MicrowaveFoodbox",
            "/World/Assets/MicrowaveBeefbox",
            "/World/Assets/MicrowaveSaladbox",
            "/World/HouseTopologyModules/MicrowaveSupport",
        ],
        "support_materialization": {
            "root_prim_path": "/World/HouseTopologyModules/MicrowaveSupport",
            "support_prim_path": "/World/HouseTopologyModules/MicrowaveSupport/Counter",
            "size_xyz_m": [1.00, 0.80, 0.90]
        },
        "runtime_readiness": "generated_support_counter_required",
    },
    "refrigerator_cluster": {
        "label": "refrigerator operation zone",
        "canonical_frame": "refrigerator",
        "canonical_pivot_xy_m": [3.173564261, 2.0315666665],
        "fixture_size_xy_m": [0.912608, 0.909218],
        "footprint_bounds_xy_m": [-0.456304, -0.454609, 0.456304, 0.454609],
        "footprint_source": "canonical_usd_composed_world_bbox_v1",
        "station_offset_m": 1.05,
        "allowed_space_kinds": ["kitchen"],
        "source_prim_roots": [
            "/World/Background/Shell/KitchenRefrigerator",
        ],
        "runtime_readiness": "articulation_and_payload_replay_required",
    },
    "kitchen_counter_cluster": {
        "label": "free preparation counter",
        "canonical_frame": "kitchen_counter",
        "canonical_pivot_xy_m": [4.678965, -0.2408],
        "fixture_size_xy_m": [1.80, 0.70],
        "footprint_bounds_xy_m": [-0.90, -0.35, 0.90, 0.35],
        "footprint_source": "generated_support_exact_dimensions_v1",
        "station_offset_m": 1.00,
        "allowed_space_kinds": ["kitchen"],
        "source_prim_roots": ["/World/HouseTopologyModules/PrepCounter"],
        "support_materialization": {
            "root_prim_path": "/World/HouseTopologyModules/PrepCounter",
            "support_prim_path": "/World/HouseTopologyModules/PrepCounter/Counter",
            "size_xyz_m": [1.80, 0.70, 0.90]
        },
        "runtime_readiness": "generated_counter_and_binding_migration_required",
    },
}
