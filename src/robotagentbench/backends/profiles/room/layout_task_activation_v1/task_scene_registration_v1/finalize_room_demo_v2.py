"""Encode, decode-verify and publish the completed native room preview."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--web',type=Path,required=True);a=ap.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
render=a.run/'render_v4';r=json.loads((a.run/'navigation_result.json').read_text());c=json.loads((a.run/'composition.json').read_text());s=json.loads((a.run/'settled_bounds_audit.json').read_text());m=json.loads((render/'render_manifest.json').read_text())
audit=json.loads((a.run/'corrected_placement_audit.json').read_text())
assert audit['passed']
assert r['status']=='native_action22_navigation_smoke_passed' and s['support_checks_passed']
assert m['status']=='frames_complete' and m['frame_count']==len(r['rows'])==871
assert all((render/'frames'/f'{i:05d}.png').is_file() for i in range(871))
assert all(sha(p)==h for p,h in c['asset_entry_hashes'].items())
assert sha(c['source'])==c['source_sha256']
video=render/'room_navigation.mp4';assert not video.exists()
subprocess.run(['ffmpeg','-v','error','-framerate','20','-i',str(render/'frames/%05d.png'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)]))['streams'][0]
assert int(probe['nb_read_frames'])==871 and probe['r_frame_rate']=='20/1'
subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
dest=a.web/a.run.name;dest.mkdir(exist_ok=False)
for name in ['overview.png','living.png','kitchen.png','shelf.png','coffee.png','room_navigation.mp4']:shutil.copy2(render/name,dest/name)
summary=dict(status='initialization_and_short_navigation_preview_complete',layout_id=1,seed=42,asset_instances=len(c['objects']),
 shelf='all_assets/snack_shelf',shelf_required_categories=14,unique_ordinary_objects=8,source_layout_and_assets_unchanged=True,
 control_hz=20,physics_hz=60,video=probe,video_sha256=sha(video),full_decode_passed=True,
 initial_pose_writes=r['initial_pose_writes'],rollout_pose_velocity_writes=r['rollout_pose_velocity_writes'],
 native_model=r['navigation_model'],translation_first_leg_m=r['translation_m'],net_displacement_m=r['total_displacement_m'],
 nonfloor_robot_penetrations_over_1mm=len(r['contacts']),table_sleep_to_wake_transitions=r['sleep_to_wake_transitions'],
 restored_L1_seat_support=c['restored_support'],settled_support_checks=s['support_checks_passed'],
 placement_repair=c['placement_repair'],corrected_placement_audit=audit,views=m,full_manipulation_acceptance=False,task_specific_frequency_switch=False,
 limitations=c['limitations']+[s['scope'],'Short reference aisle route, not coverage of all room regions or all 50 layouts'])
(dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for src,name in [(a.run/'composition.json','composition.json'),(a.run/'settled_bounds_audit.json','settled_bounds_audit.json')]:shutil.copy2(src,dest/name)
for name in ['placement_repair.json','corrected_placement_audit.json']:shutil.copy2(a.run/name,dest/name)
print(json.dumps({'status':summary['status'],'video':probe,'destination':str(dest),'corrected_placement_passed':audit['passed']},ensure_ascii=False),flush=True)

