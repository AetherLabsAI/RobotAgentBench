"""Recover only the verified 18-view legacy failure; never rerun or fake missing views."""
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from render_quality import record_view

HERE=Path(__file__).resolve().parent


def recover():
    folder=HERE/'generated_rooms/room_237e574c5df3'
    source=folder/'renders_v2'
    output=folder/'renders_v3'
    error=json.loads((source/'error.json').read_text())
    if error['error']!='Invalid rendered image: area_fridge':
        raise ValueError('Unexpected legacy failure')
    record=json.loads((folder/'composition.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if record['scene_sha256']!=digest(folder/'room.usda'):
        raise ValueError('Scene changed')
    views=[];pending=None;total=None
    for line in (folder/'render.log').read_text().splitlines():
        if line.startswith('RENDERED '):pending=line.split()[1]
        if line.startswith('ROOM_PROGRESS '):
            event=json.loads(line[len('ROOM_PROGRESS '):])
            if 'total' in event:total=event['total']
            if pending and event.get('completed')==len(views)+1:
                views.append(dict(name=pending,label=event['message']))
                pending=None
    if len(views)!=18 or total!=20 or views[-1]['name']!='area_drawer':
        raise ValueError('Unexpected legacy progress; no publication')
    output.mkdir(exist_ok=False)
    images=[]
    for view in views:
        with Image.open(source/(view['name']+'.png')) as image:
            image.load()
            item,warning=record_view(np.asarray(image),view,output)
        if warning:raise ValueError('Retained image did not pass quality check: '+str(warning))
        images.append(item)
    receipt=dict(renderer='Isaac RTX RaytracedLighting',resolution=[1280,800],
        scene_sha256=record['scene_sha256'],images=images,static_scene_only=True,native_physics_run=False,
        status='recovered_partial',total_views=20,attempted_views=19,valid_images=18,
        warnings=[dict(name='area_fridge',label='冰箱操作区',reason='旧运行图片检查失败，未保留异常像素',status='failed'),
                  dict(name='area_microwave',label='微波炉操作区',reason='旧运行提前结束，此视角尚未渲染',status='not_attempted')],
        recovery=dict(source_directory='renders_v2',source_log_sha256=digest(folder/'render.log'),
                      note='Existing images revalidated only; no new rendering or physics. Camera metadata was not retained by old run.'))
    (output/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    shutil.copy2(folder/'render_gpu.json',folder/'render_gpu_renders_v3.json')
    shutil.copy2(folder/'composition.json',folder/'composition.before_warning_recovery.json')
    from publish_render import publish
    publish(folder.name,'renders_v3')


if __name__=='__main__':
    with (HERE/'generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        recover()
