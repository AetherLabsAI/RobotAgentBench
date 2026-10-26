import json,shutil
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
run=HERE/'other_layouts_v4';web=HERE.parent/'web/other_layouts_v4'
views=['overview','kitchen','coffee','shelf','sofa','table','small_table']
summaries=json.loads((run/'summary.json').read_text())
for row in summaries:
    folder=Path(row['scene']).parent
    assert json.loads((folder/'mesh_audit.json').read_text())['planar_mesh_check_passed']
    assert json.loads((folder/'renders/manifest.json').read_text())['static_scene_only']
    assert len(row['living_randomization']['changed_objects'])==9
    for name in views:
        with Image.open(folder/('detail_fix' if name=='small_table' else 'renders')/f'{name}.png') as im:
            im.load();assert im.size==(1280,800)
web.mkdir(exist_ok=False)
for row in summaries:
    folder=Path(row['scene']).parent;target=web/f'layout{row["layout_id"]:02d}';target.mkdir()
    for name in views:shutil.copy2(folder/('detail_fix' if name=='small_table' else 'renders')/f'{name}.png',target)
    shutil.copy2(folder/'mesh_audit.json',target)
shutil.copy2(run/'summary.json',web)
shutil.copy2(HERE/'other_layouts_failures_v1.json',web/'failures.json')
print('Published',len(summaries),'layouts,',len(views)*len(summaries),'static views; no native validation claimed')
