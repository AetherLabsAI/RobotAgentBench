// Generated from openend/tasks/trash_transfer/paper_variants.json; do not hand-edit.
(function(root){const data={
  "schema_version": 1,
  "asset_id": "paper_ball",
  "asset_entry": "all_assets/paper_ball/asset.usda",
  "trash_category": "paper",
  "task_type": "trash_transfer",
  "maximum_score_per_instance": 8,
  "selection_owner": "unified_scene_initialization",
  "runtime_resize_allowed": false,
  "shared_asset_mutation_allowed": false,
  "source_asset_sha256": "82c6cb8011bd10b2e7f0c7a310c53cd1050f2f132c924063edc0a3d20b950636",
  "source_mesh_points_sha256": "83d8442a03c34422dcde49247fe986437183e9360dec27b5dc071663b3d58fe9",
  "base_effective_mesh_axis_scale": [
    1.5,
    1.5,
    1.5
  ],
  "variant_axis_frame": "mesh_axes_before_asset_root_rotation_not_world_xyz",
  "variants": {
    "large": {
      "label": "大纸团",
      "effective_mesh_axis_scale": [
        1.5,
        1.5,
        1.5
      ],
      "relative_mesh_axis_scale": [
        1,
        1,
        1
      ],
      "authored_pose_size_m": [
        0.058117708467195356,
        0.0855660897200112,
        0.15538805167546588
      ],
      "rotation_removed_size_m": [
        0.15538805164396763,
        0.0855660897200112,
        0.05811770809115526
      ],
      "source": "Task7 original 7.2_old_00..09/prepared/initial_scene.usda; effective episodeInitial matrix axis lengths",
      "source_instances_checked": 10
    },
    "small": {
      "label": "小纸团",
      "effective_mesh_axis_scale": [
        0.875,
        1.25,
        1.5
      ],
      "relative_mesh_axis_scale": [
        0.5833333333333334,
        0.8333333333333334,
        1
      ],
      "authored_pose_size_m": [
        0.058117708467195356,
        0.071305074766676,
        0.09064303014379159
      ],
      "rotation_removed_size_m": [
        0.09064303012564773,
        0.07130507476667597,
        0.05811770809115524
      ],
      "source": "Task7 original 7.2_new_00..39/scene.usda; effective task7Guide matrix axis lengths",
      "source_instances_checked": 40
    }
  },
  "limitations": [
    "Size variants only: no new orientation or physical parameter approval is implied.",
    "Rotation-removed dimensions are not world-aligned XY placement bounds.",
    "Apply relative scale in mesh axes before existing root rotation; do not multiply absolute scale onto the already scaled asset.",
    "Source replays were normalized to the small size; original video scenes are authoritative for large/small provenance.",
    "Existing generated rooms are not retroactively resized.",
    "User authorized shared preview and scene-generator wiring; no physical success claimed."
  ]
};
if(typeof module!=='undefined'&&module.exports)module.exports=data;
else root.PAPER_VARIANTS=data;
})(typeof window==='undefined'?globalThis:window);
