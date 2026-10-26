"""Targeted camera-only render; immutable room and previous images retained."""
import fcntl,json,sys
from pathlib import Path
from render_stream import stream
from publish_render import publish
HERE=Path(__file__).resolve().parent
folder=HERE/'generated_rooms/room_237e574c5df3'

if __name__=='__main__':
    with (HERE/'generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def report(value):
            (folder/'fridge_camera_repair_progress.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
            print(json.dumps(value,ensure_ascii=False),flush=True)
        stream([sys.executable,'-B',str(HERE/'launch_render.py'),str(folder),
                '--output','renders_v4','--reuse-output','renders_v3',
                '--views','area_fridge','area_microwave'],folder/'render_fridge_camera_fix.log',report)
        publish(folder.name,'renders_v4')
