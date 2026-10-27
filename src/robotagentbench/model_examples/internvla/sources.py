"""Source identities checked for this adapter; no weights/data are bundled."""
UPSTREAM = {
    'repository':'https://github.com/InternRobotics/InternVLA-A-series',
    'commit':'e6fc904f9edbfb14532e97095fc2372202517f76',
    'api':'InternVLAA15Policy.predict_action_chunk',
    'license':'CC BY-NC-SA 4.0 (upstream dependency, not redistributed here)',
    'references':(
        'evaluation/RoboTwin/inference.py',
        'src/lerobot/transforms/core.py',
        'src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py',
        'src/lerobot/configs/policies.py',
        'src/lerobot/policies/pretrained.py',
        'tutorials/installation.md',
    ),
}

# These identify the real historical G1-serving evidence inspected read-only.
# They are NOT pins for arbitrary replacement checkpoints and do not establish
# new-model or new-worker execution. No historical evaluator paths are needed.
G1_REFERENCE = {
    'kind':'historical_model_ready_and_actual_model_serving_source_not_trajectory_replay',
    'control_schema_sha256':'256c212eb078ce43c218960859d1d3cd75cc7ce60b377e78a93f1a704624f03c',
    'serving_source_sha256':'a2e030601ee9531de6780a3a5220eb482ff523507acb440246590f56b9a5b3ea',
    'checkpoint_config_sha256':'80bcaeff1f2364ce196544dae6ab9bc717d098fc7b0a833859071b1853f0cbe9',
    'stats_sha256':'33f24fb5490e5c5f6808d18b7ba66a04549e48311c900cf7a8ede6534eb5726a',
    'state_dim':23,
    'supervised_action_dim':22,
    'chunk_size':50,
    'padded_state_action_dim':32,
    'new_real_model_execution':False,
    'historical_task_index_lookup_reused':False,
    'historical_base_projection_reused':False,
}
