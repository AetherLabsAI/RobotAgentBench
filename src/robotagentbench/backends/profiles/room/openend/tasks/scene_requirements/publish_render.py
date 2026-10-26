"""Publish an already completed immutable episode render after verifying receipts."""
import hashlib
import json
import sys
from generate_room import HERE, REMOTE, SCP, HOST, run, SERVER_MODE, fetch


def publish(episode,directory='renders_v2'):
    if directory not in ('renders_v2','renders_v3','renders_v4'):raise ValueError('Invalid render directory')
    if not episode.startswith('room_') or any(c not in 'room_0123456789abcdef' for c in episode):
        raise ValueError('Invalid episode id')
    folder=HERE/'generated_rooms'/episode
    folder.mkdir(parents=True,exist_ok=True)
    remote=REMOTE+'/generated_rooms/'+episode
    gpu_file='render_gpu.json' if directory=='renders_v2' else 'render_gpu_'+directory+'.json'
    for name in ('composition.json','runtime_request.json','room.usda',gpu_file):
        fetch(remote,name,folder)
    if not SERVER_MODE:run(SCP+['-r',HOST+':'+remote+'/'+directory,str(folder)+'/'])
    result=json.loads((folder/'composition.json').read_text())
    receipt=json.loads((folder/directory/'manifest.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(folder/'room.usda')!=receipt['scene_sha256'] or receipt['scene_sha256']!=result['scene_sha256']:
        raise ValueError('Rendered scene hash mismatch')
    if not receipt['images']:
        raise ValueError('No usable rendered images')
    for item in receipt['images']:
        import re
        if not re.fullmatch(r'(overview|living|kitchen|area_[a-z_]+)\.png',item['file']):
            raise ValueError('Unexpected render file')
        if digest(folder/directory/item['file'])!=item['sha256']:
            raise ValueError('Rendered image hash mismatch')
    result['render']=dict(receipt,directory=directory,gpu=json.loads((folder/gpu_file).read_text()))
    serialized=json.dumps(result,ensure_ascii=False,indent=2)
    (folder/'composition.json').write_text(serialized)
    if not SERVER_MODE:run(SCP+[str(folder/'composition.json'),HOST+':'+remote+'/composition.json'])
    tmp=HERE/'current_room.next.json';tmp.write_text(serialized);tmp.replace(HERE/'current_room.json')
    print(json.dumps(dict(episode=episode,images=[i['name'] for i in receipt['images']],gpu=result['render']['gpu']['index'])))

if __name__=='__main__':publish(*sys.argv[1:])
