"""Generate an immutable test scene without requesting GPU rendering."""
import json,sys,uuid
from pathlib import Path
from generate_room import HERE,REMOTE,SSH,SCP,HOST,run,choose_plan
from room_request import resolve

request=json.loads(Path(sys.argv[1]).read_text())
folder=HERE/'generated_rooms'/('binding_'+uuid.uuid4().hex[:12]);folder.mkdir()
(folder/'request.json').write_text(json.dumps(request))
data=choose_plan(resolve(request),folder)
(folder/'input.json').write_text(json.dumps(data))
remote=REMOTE+'/generated_rooms/'+folder.name
print(str(folder),flush=True)
run(SSH+['mkdir -p '+remote])
run(SCP+[str(folder/'input.json'),HOST+':'+remote+'/input.json'])
run(SSH+['bash /home/lingjun/robot_agent_arena/api-protocol/evaluation/tools/run_cpu_usd.sh '+REMOTE+'/build_selected_room.py '+remote+'/input.json'])
for name in ('composition.json','runtime_request.json','room.usda'):
    run(SCP+[HOST+':'+remote+'/'+name,str(folder/name)])
print(json.dumps(json.loads((folder/'composition.json').read_text())['registration'],ensure_ascii=False))
