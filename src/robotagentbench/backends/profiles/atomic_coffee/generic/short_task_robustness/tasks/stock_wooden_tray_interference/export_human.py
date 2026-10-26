"""Offline Blender CPU bake of the licensed right arm; originals are read-only.

The 0.1 unit conversion was explicitly approved by the user. Finger poses
are a visual/scripted grasp, NOT a claim of a friction-only physical grasp.
The target is carried by the separately approved temporary native constraint.
"""
import hashlib
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'assets/model/FPS ARMS RIG 1.blend'
SCALE = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', choices=['R','L'], default='R')
    parser.add_argument('--cylindrical', action='store_true')
    parser.add_argument('--forearm-only', action='store_true')
    parser.add_argument('--rig-data', action='store_true', help='Export source skin weights for task-local static pose fitting')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    suffix = '.'+args.side
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE), load_ui=False, use_scripts=False)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    body = next(o for o in bpy.data.objects if o.type == 'MESH')
    arm.animation_data_clear()
    bpy.context.view_layer.update()
    # Wrist-aligned local frame; export the right side only. Full skin remains
    # continuous, while the physics representation has separate bone hulls.
    wrist = arm.pose.bones['hand'+suffix].matrix.copy()
    canonical = wrist.inverted() @ arm.matrix_world.inverted() @ body.matrix_world
    groups = {g.index: g.name for g in body.vertex_groups}
    selected = [v.index for v in body.data.vertices
                if sum(g.weight for g in v.groups if groups[g.group].endswith(suffix)) > 0.5]
    if args.forearm_only:
        elbow_y=(wrist.inverted() @ arm.pose.bones['forearm'+suffix].matrix).translation.y
        selected=[i for i in selected if (canonical @ body.data.vertices[i].co).y >= elbow_y]
    indices = {old: new for new, old in enumerate(selected)}
    faces = [p for p in body.data.polygons if all(i in indices for i in p.vertices)]
    uv = body.data.uv_layers.active
    bones = [b for b in arm.pose.bones if b.name.endswith(suffix)
             and '.control' not in b.name and any(
                 groups[g.group] == b.name and g.weight > 0
                 for i in selected for g in body.data.vertices[i].groups)]
    if args.forearm_only:
        bones=[b for b in bones if not b.name.startswith(('upper_arm','deltoid','clavicle'))]
    # Canonical metre transforms retain rotation, scale only translations.
    def transform(bone):
        m = wrist.inverted() @ bone.matrix
        m.translation *= SCALE
        return [list(row) for row in m]

    bone_names = {b.name for b in bones}
    memberships = {name: [] for name in bone_names}
    for old in selected:
        weights = [(g.weight, groups[g.group]) for g in body.data.vertices[old].groups
                   if groups[g.group] in bone_names]
        if weights:
            memberships[max(weights)[1]].append(indices[old])
        elif args.forearm_only:
            memberships['forearm'+suffix].append(indices[old])
    base_rotations = {b.name: b.rotation_euler.copy() for b in bones}
    rest = {b.name: transform(b) for b in bones}
    frames = []
    # CPU animation bake; these are geometric poses, not solver settings.
    # A quarter-turn closes each finger chain around a cylindrical object.
    # The native admission must reject penetrating/misaligned grasp poses.
    for sample in range(21):
        fraction = sample / 20
        for b in bones:
            b.rotation_mode = 'XYZ'
            b.rotation_euler = base_rotations[b.name].copy()
            if args.cylindrical and b.name.startswith('f_'):
                bend = {'.01.': 45, '.02.': 65, '.03.': 35}
                b.rotation_euler.x += fraction*math.radians(next(v for k,v in bend.items() if k in b.name))
            elif b.name.startswith('f_') and '.01.' in b.name:
                b.rotation_euler.x += fraction * math.pi / 3
            elif b.name == 'thumb.02'+suffix:
                b.rotation_euler.x += fraction * math.pi / 4
        bpy.context.view_layer.update()
        evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        if len(mesh.vertices) != len(body.data.vertices):
            raise RuntimeError('Unexpected topology-changing source modifier')
        points = [list((canonical @ mesh.vertices[i].co) * SCALE) for i in selected]
        frames.append(dict(closure=fraction, points=points,
                           bones={b.name: transform(b) for b in bones}))
        evaluated.to_mesh_clear()
    result = dict(schema='arena.human.cpu_bake.v1', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  unit_scale_m=SCALE, scale_basis='user_approved_2026-10-11',
                  source_license='CC0-1.0', texture='../model/new_diff.png',
                  faces=[[indices[i] for i in p.vertices] for p in faces],
                  uv=[list(uv.data[i].uv) for p in faces for i in p.loop_indices],
                  rest_bones=rest, collider_vertices=memberships, frames=frames,
                  physical_validation=False, grasp='approved_temporary_native_fixed_joint')
    if args.rig_data:
        result['parents']={b.name:b.parent.name if b.parent and b.parent.name in bone_names else None for b in bones}
        result['skin_weights']=[]
        for old in selected:
            weights={groups[g.group]:g.weight for g in body.data.vertices[old].groups if groups[g.group] in bone_names}
            if not weights:weights={'forearm'+suffix:1.}
            total=sum(weights.values())
            result['skin_weights'].append({k:v/total for k,v in weights.items()})
    out = HERE / 'assets/derived'
    out.mkdir(exist_ok=True)
    # Preserve the original bake. Right-named interface keys are canonical
    # actuator labels, not a claim about source handedness.
    if args.side=='L':
        for key in ('rest_bones','collider_vertices'):
            result[key]={k.replace('.L','.R'):v for k,v in result[key].items()}
        for frame in result['frames']:
            frame['bones']={k.replace('.L','.R'):v for k,v in frame['bones'].items()}
        if args.rig_data:
            result['parents']={k.replace('.L','.R'):v.replace('.L','.R') if v else None for k,v in result['parents'].items()}
            result['skin_weights']=[{k.replace('.L','.R'):v for k,v in w.items()} for w in result['skin_weights']]
    result['source_side']=args.side
    result['actor_extent']='hand_and_forearm' if args.forearm_only else 'full_source_arm'
    name = ('left_arm' if args.side=='L' else 'right_arm') + ('_grip' if args.cylindrical else '')+('_forearm' if args.forearm_only else '')+('_rig' if args.rig_data else '')+'.json'
    (out / name).write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print('EXPORTED', json.dumps(dict(vertices=len(selected), faces=len(faces), bones=len(bones),
                                     frames=len(frames), unit_scale_m=SCALE)))


if __name__ == '__main__':
    main()
