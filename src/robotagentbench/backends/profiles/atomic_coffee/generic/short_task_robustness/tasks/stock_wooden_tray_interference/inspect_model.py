"""Blender CPU inspection; no saved source edits, drivers or autorun scripts."""
import json
from pathlib import Path
import sys
import bpy

HERE = Path(__file__).resolve().parent
source = HERE / 'assets/model/FPS ARMS RIG 1.blend'
bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False,
                         use_scripts=False)
scene = bpy.context.scene
objects = []
for obj in scene.objects:
    row = dict(name=obj.name, type=obj.type,
               position=list(obj.location), scale=list(obj.scale),
               dimensions=list(obj.dimensions))
    if obj.type == 'ARMATURE':
        row['bones'] = [dict(name=b.name, parent=b.parent.name if b.parent else None,
                             head=list(b.head_local), tail=list(b.tail_local))
                        for b in obj.data.bones]
    if obj.type == 'MESH':
        row.update(vertices=len(obj.data.vertices), polygons=len(obj.data.polygons),
                   vertex_groups=[g.name for g in obj.vertex_groups],
                   modifiers=[dict(type=m.type, name=m.name) for m in obj.modifiers])
    objects.append(row)
out = HERE / 'assets/model_inspection.json'
out.write_text(json.dumps(dict(
    blender_version=bpy.app.version_string,
    unit_system=scene.unit_settings.system,
    scale_length=scene.unit_settings.scale_length,
    objects=objects,
    materials=[m.name for m in bpy.data.materials],
    images=[dict(name=i.name, path=i.filepath, packed=bool(i.packed_file))
            for i in bpy.data.images],
    actions=[a.name for a in bpy.data.actions],
    physical_validation=False), indent=2) + '\n')
print('MODEL_INSPECTION', out)
