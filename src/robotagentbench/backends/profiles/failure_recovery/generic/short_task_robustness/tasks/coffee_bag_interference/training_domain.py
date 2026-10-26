"""Exact original atomic 1-2 bag-origin XY randomization domain.

Read the unchanged producer's uniform bounds; do not expand the domain around
the current start, confuse root coordinates with footprint or silently shorten
the approved displacement range.
"""
import ast
import hashlib
import math
from itertools import product
from pathlib import Path


def producer_offset_bounds():
    from evaluation.tasks.coffee_machine import vendor_producer
    path=vendor_producer.HERE/'build_cycle_initial_reference.py'
    tree=ast.parse(path.read_text())
    choices=[node.value for node in ast.walk(tree) if isinstance(node,ast.keyword) and node.arg=='bag_offset_m']
    if len(choices)!=1:raise ValueError('Original bag offset sampler must be unique')
    offsets=choices[0]
    if not isinstance(offsets,ast.List) or len(offsets.elts)!=3:raise ValueError('Unexpected source bag offset shape')
    bounds=[]
    for node in offsets.elts[:2]:
        if (not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute)
                or node.func.attr!='uniform' or len(node.args)!=2):raise ValueError('Source must retain its uniform XY sampler')
        pair=[float(ast.literal_eval(v)) for v in node.args]
        if not all(math.isfinite(v) for v in pair) or pair[0]>=pair[1]:raise ValueError('Invalid source bounds')
        bounds.append(pair)
    if ast.literal_eval(offsets.elts[2])!=0.:raise ValueError('Bag height must remain the original countertop height')
    return bounds,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      sampling='bag_offset_m=[uniform(source_x),uniform(source_y),0]')


def domain(initial_plan):
    from evaluation.tasks.coffee_machine.initial_state import package
    manifest,binding,_=package('1-2',initial_plan['source_episode'])
    source_origin=binding['world']['bag'][12:15]
    old_offset=manifest['facts']['randomization']['bag_offset_m']
    nominal=[float(a-b) for a,b in zip(source_origin,old_offset)]
    offsets,provenance=producer_offset_bounds()
    lower=[nominal[i]+offsets[i][0] for i in range(2)]
    upper=[nominal[i]+offsets[i][1] for i in range(2)]
    return dict(nominal_bag_origin_m=nominal,offset_bounds_m=offsets,origin_bounds_world_xy_m=[lower,upper],
                coordinate='bag rigid-body root, not full visual footprint',source=provenance)


def contains_origin(position,definition):
    if len(position)<2 or not all(math.isfinite(v) for v in position[:2]):raise ValueError('Finite bag origin required')
    lower,upper=definition['origin_bounds_world_xy_m']
    return all(lo<=v<=hi for lo,v,hi in zip(lower,position,upper))


def maximum_displacement(position,definition):
    lower,upper=definition['origin_bounds_world_xy_m']
    return max(math.hypot(corner[0]-position[0],corner[1]-position[1])
               for corner in product(*zip(lower,upper)))
