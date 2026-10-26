"""CPU Blender asset-review closeups, NOT simulation frames or grasp proof.

Uses the real unscaled can asset, source-weighted human mesh and skin texture.
The cameras/lights belong only to this new static review scene.
"""
from pathlib import Path
import json
import argparse
import sys
import bpy
from mathutils import Matrix,Vector

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='static_grip_007')
parser.add_argument('--asset',default='sprite_c1')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT=HERE/args.output
ROOT=HERE.parents[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
asset=ROOT/'api-protocol/all_assets'/args.asset/'asset.usda'
bpy.ops.wm.usd_import(filepath=str(asset))
can_objects=[o for o in scene.objects if o.type=='MESH']
corners=[o.matrix_world@Vector(c) for o in can_objects for c in o.bound_box]
lo=Vector([min(p[i] for p in corners) for i in range(3)])
hi=Vector([max(p[i] for p in corners) for i in range(3)])
center=(lo+hi)/2
data=json.loads((OUT/'hand.json').read_text());fit=json.loads((OUT/'fit.json').read_text())
if abs((hi-lo).z-fit['height_m'])>1e-4:raise RuntimeError('Imported target dimensions differ; no automatic rescaling')
rotation=Matrix(((0,-1,0),(0,0,1),(-1,0,0)))
grip_center=Vector(fit['center_local'])
points=[rotation@(Vector(p)-grip_center)+center for p in data['frames'][0]['points']]
mesh=bpy.data.meshes.new('SourceWeightedHand')
mesh.from_pydata(points,[],data['faces']);mesh.update()
obj=bpy.data.objects.new('StaticHandPreview',mesh);scene.collection.objects.link(obj)
uv=mesh.uv_layers.new(name='UVMap')
for loop,tex in zip(uv.data,data['uv']):loop.uv=tex
for face in mesh.polygons:face.use_smooth=True
material=bpy.data.materials.new('OriginalSkin');material.use_nodes=True
nodes=material.node_tree.nodes
surface=nodes.get('Principled BSDF');surface.inputs['Roughness'].default_value=.65
texture=nodes.new('ShaderNodeTexImage')
texture.image=bpy.data.images.load(str(HERE/'assets/model/new_diff.png'))
material.node_tree.links.new(texture.outputs['Color'],surface.inputs['Base Color'])
obj.data.materials.append(material)
world=bpy.data.worlds.new('ReviewWorld');world.use_nodes=True
world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.8,.8,1)
world.node_tree.nodes['Background'].inputs[1].default_value=.4;scene.world=world
for name,pos,power,size in [('Key',(.1,-.3,.45),35,.35),('Fill',(-.25,.15,.3),20,.3)]:
    light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
    item=bpy.data.objects.new(name,light);scene.collection.objects.link(item);item.location=pos
    item.rotation_euler=(center-item.location).to_track_quat('-Z','Y').to_euler()
camera=bpy.data.cameras.new('StaticReviewOnly');camera.type='ORTHO';camera.ortho_scale=.29
cam=bpy.data.objects.new('StaticReviewOnly',camera);scene.collection.objects.link(cam);scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32
scene.render.threads_mode='FIXED';scene.render.threads=8
scene.render.resolution_x=640;scene.render.resolution_y=480;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
for name,position in [('front',(.12,-.40,.17)),('side',(-.33,.10,.17)),('top',(.015,-.015,.5))]:
    cam.location=Vector(position)+center
    cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(OUT/(name+'.png'));bpy.ops.render.render(write_still=True)
(OUT/'render_receipt.json').write_text(json.dumps(dict(kind='static_asset_review',renderer='Blender Cycles CPU',
    physics_run=False,can_asset=str(asset),can_dimensions_m=list(hi-lo),can_scale_changed=False,
    views=['front','side','top']),indent=2)+'\n')
