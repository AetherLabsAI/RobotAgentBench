"""Check recorded evidence; no image statistic is treated as task success/glare."""
import argparse
import json
from pathlib import Path
import subprocess
from .initial import task_output
from .cycle import FLASH_MULTIPLIER, MULTIPLIER, FLASH_S


def verify(output):
    out=task_output(output).resolve(strict=True)
    report=json.loads((out/'report.json').read_text())
    frames=json.loads((out/'frames.json').read_text())
    illumination=json.loads((out/'lighting.json').read_text())
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'stream=width,height,nb_frames','-show_entries','format=duration','-of','json',
        str(out/'lighting_preview.mp4')],text=True))
    flashes=[e for e in illumination['events'] if e['kind']=='light_changed' and e['state']['flash']]
    placements=illumination.get('direct_flash_placement_events',[])
    checks=dict(recorded=report['status']=='recorded', clean_shutdown=report.get('native_shutdown_complete') is True,
        cleanup_clean=not report['cleanup_errors'], initial_accepted=report['initial_acceptance']['accepted'] is True,
        layers_unchanged=report['source_layers_unchanged'] is True, idle_no_false_score=report['measured_idle_score']==0,
        original_camera=probe['streams'][0]['width']==640 and probe['streams'][0]['height']==480,
        complete_capture=len(frames)==241 and int(probe['streams'][0]['nb_frames'])==241,
        original_clock=all(abs(b['time_s']-a['time_s']-.05)<1e-6 for a,b in zip(frames,frames[1:])),
        flash_observed=bool(flashes),
        flash_range=all(FLASH_MULTIPLIER[0]<=e['state']['flash_multiplier']<=FLASH_MULTIPLIER[1] for e in flashes),
        ambient_range=all(MULTIPLIER[0]<=e['state']['multiplier']<=MULTIPLIER[1] for e in illumination['events']),
        flash_duration=all(FLASH_S[0]-1e-9<=e['flash_end_s']-e['scheduled_time_s']<=FLASH_S[1]+1e-9 for e in flashes),
        clear_position_recorded=len(placements)==len(flashes)>0,
        clear_before_obstruction=bool(placements) and all(p['original_camera_near_clip_m']<p['distance_from_camera_m']<
            p['first_obstruction_m'] and p['center_ray_unobstructed'] for p in placements))
    summary={}
    for mode in ('normal','ambient','side','camera_flash'):
        rows=[f for f in frames if f['state']['mode']==mode]
        if rows:
            summary[mode]=dict(frames=len(rows), mean_rgb_min=min(f['mean_rgb'] for f in rows),
                mean_rgb_max=max(f['mean_rgb'] for f in rows),
                bright_fraction_max=max(f['bright_fraction'] for f in rows),
                camera_world_position_m=rows[0].get('camera_world_position_m'),
                emitter_world_position_m=rows[0].get('emitter_world_position_m'))
    result=dict(integrity_passed=all(checks.values()), checks=checks, image_observations=summary,
                rendered_video=probe, preview_only=True, model_run=False,
                task_success_verified=False, visual_glare_verified=False,
                note='Glare needs visual inspection; intensity authoring alone is not evidence of visible glare.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    result=verify(a.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)
