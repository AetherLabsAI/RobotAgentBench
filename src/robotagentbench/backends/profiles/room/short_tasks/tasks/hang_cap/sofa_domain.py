"""User-approved middle-two-cushion domain; no change to shared sofa policy."""
import numpy as np
import json
from pathlib import Path
from .range_initial_state import sofa_frame, planar_rotation

POLICY = 'middle_two_cushions_backrest_clearance_v2'
STATION_SOURCE = Path(__file__).with_name('robot_station_l1.json')
CURRENT_STATION_SOURCE = Path(__file__).with_name('robot_station_v5.json')

def robot_station_domain(sofa, inverse_delta):
    evidence=json.loads(CURRENT_STATION_SOURCE.read_text())
    near,far=evidence['front_edge_distance_m']
    center=(near+far)/2
    canonical=[evidence['canonical_front_edge_x_m']-center,
               evidence['canonical_centerline_y_m'],0.,1.]
    base=np.asarray(canonical) @ np.linalg.inv(inverse_delta)
    _,heading=sofa_frame(sofa)
    lateral=evidence['lateral_offset_m']
    return [float(base[0]),float(base[1]),float(heading)], [[center-far,lateral[0]],[center-near,lateral[1]]]

def fixed_station(sofa, inverse_delta):
    evidence=json.loads(STATION_SOURCE.read_text())
    base=np.r_[evidence['canonical_centered_xy_m'],0.,1.] @ np.linalg.inv(inverse_delta)
    _,heading=sofa_frame(sofa)
    return [float(base[0]),float(base[1]),float(heading)]

def usable_bounds(policy, structure):
    # The shared region spans three cushion widths (two full + two halves).
    half = policy['reference_approximate_width_m'] / 3
    edge = policy['edge_margin_m']
    back = min(o['bounds'][1] for o in structure['obstacles'] if o['kind']=='backrest')
    return np.array([[-half+edge, edge],
                     [half-edge, min(policy['reference_depth_m']-edge,
                                     back-policy['ordinary_object_padding_xy_m'])]])

def domain_local(sofa, inverse_delta, policy, structure):
    lo,hi = usable_bounds(policy, structure)
    uv = np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    canonical = np.c_[uv[:,1]-.3503563938, -uv[:,0]-.8710, np.zeros(4), np.ones(4)]
    world = (canonical @ np.linalg.inv(inverse_delta))[:,:2]
    origin,heading=sofa_frame(sofa)
    local=(world-origin[:2]) @ planar_rotation(-heading).T
    return np.array([local.min(0),local.max(0)])

def footprint_inside(uv, policy, structure):
    lo,hi=usable_bounds(policy,structure)
    return bool(np.all(np.min(uv,0)>=lo-1e-9) and np.all(np.max(uv,0)<=hi+1e-9))
