"""Original source invocation recipe only; no simulator/scorer implementation.

Physical CLI values come directly from the pinned original trial, matching the
already-reviewed run_original_entry / run_source_owner recipe. The 65 frozen
rows supply its initial command input; the mandatory owner hook intercepts
before the diagnostic replay loop. They are not formal post-admission actions.
"""
from pathlib import Path

PHYSICAL_KEYS=('material_model','stretch','shear','bend','damping','velocity_damping','iterations',
 'cloth_friction','youngs_modulus','surface_thickness','surface_bend_modulus','clean_surface_bend_modulus',
 'elasticity_damping','bend_damping','surface_self_filter_distance','surface_max_linear_velocity',
 'surface_contact_offset','surface_rest_offset','surface_collision_multiplier','surface_pair_updates',
 'clean_rest_fold_deg','clean_initial_shape','clean_initial_offset','dirty_initial_offset')
SOURCE_FLAGS=('diagnostic_convex_table_collider_candidate','diagnostic_basket_box_collider_candidate','diagnostic_basket_box_replace_source')


def source_argv(owner,selected,trial,output,commands):
    roots={k:Path(v)for k,v in owner['directories'].items()}
    native=roots['original_root']/'code/tablecloth_workflow/native_teleop/lay_20260924'
    if trial['fps']!=20 or trial['physics_clock']['actual_hz']!=60:
        raise ValueError('Existing original 60Hz/20Hz source recipe required')
    argv=[str(native/selected['entry']),'--diagnostic-canonical','--headless-command-replay',
        '--reference',str(native/selected['reference']),
        '--scene',str(roots['asset_root']/'assets/alienware/arena_scene/v1_2/demo_sim/scene_room_baseline.usda'),
        '--robot',str(roots['asset_root']/'assets/alienware/arena_scene/v1_2/source/geniesim/assets/robot/G1_omnipicker/robot.usda'),
        '--output',str(Path(output)/'outputs/original'),'--commands',str(commands),'--max-frames','64']
    settings=trial['config']
    for name in PHYSICAL_KEYS:
        value=settings[name]
        if value is None:continue
        argv.append('--'+name.replace('_','-'))
        argv.extend(str(v)for v in value)if isinstance(value,(list,tuple))else argv.append(str(value))
    for name in SOURCE_FLAGS:
        if settings[name]:argv.append('--'+name.replace('_','-'))
    if settings.get('continue_diagnostic_geometry'):argv.append('--continue-diagnostic-geometry')
    return argv
