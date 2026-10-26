"""Blender BVH cross-finger triangle intersection diagnostic (not physics)."""
import argparse
import json
import sys
from pathlib import Path
from mathutils.bvhtree import BVHTree

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('outputs',nargs='+')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
for folder in args.outputs:
    out=HERE/folder
    data=json.loads((out/'hand.json').read_text())
    groups=data['collider_vertices'];points=data['frames'][0]['points']
    names=['index','middle','ring','pinky','thumb']
    trees={};counts={}
    for name in names:
        prefix='thumb.' if name=='thumb' else 'f_'+name+'.'
        indices=set(i for k,v in groups.items() if k.startswith(prefix) for i in v)
        # Excludes shared webbing; only triangles wholly belonging to a digit.
        faces=[f for f in data['faces'] if set(f).issubset(indices)]
        trees[name]=BVHTree.FromPolygons(points,faces,all_triangles=False,epsilon=0.)
        counts[name]=len(faces)
    pairs={}
    for i,a in enumerate(names):
        for b in names[i+1:]:pairs[a+' / '+b]=len(trees[a].overlap(trees[b]))
    report=dict(kind='static_cross_finger_triangle_bvh',face_counts=counts,
        intersection_pair_counts=pairs,excludes_shared_webbing=True,physical_validation=False)
    (out/'finger_intersections.json').write_text(json.dumps(report,indent=2)+'\n')
    print(folder,json.dumps(report),flush=True)
