"""User-approved short entry, a subset of the existing reserved corridor.

No new destination, orientation, timings or physical settings. The short
length is the existing side-entry candidate: twice the measured footprint.
"""
from copy import deepcopy
import numpy as np


def shorten_entry(reservation, asset):
    result=deepcopy(reservation)
    points=np.asarray(result['waypoints'],dtype=float)
    target=np.asarray(result['initial_target_bounds'],dtype=float)
    direction=points[0]-points[1]
    original=float(np.linalg.norm(direction))
    length=2*float(max(target[1,:2]-target[0,:2]))
    if not 0<length<original:
        raise ValueError('No shorter existing footprint-derived candidate')
    points[0]=points[1]+direction*(length/original)
    orientation=np.asarray(result['hand_orientation'])
    vertices=asset.sample(0.)[0]@orientation.T+points[0]
    for bone,ids in asset.collision_groups().items():
        box=np.array([vertices[ids].min(0),vertices[ids].max(0)])
        if np.all(box[0]<target[1]) and np.all(target[0]<box[1]):
            raise ValueError('Nearby waiting hand overlaps target: '+bone)
    result['waypoints']=points.tolist()
    result['diagnostic_entry_variant']=dict(policy='short_existing_corridor_from_near_target',
        original_length_m=original,entry_length_m=length,
        destination_unchanged=True,orientation_unchanged=True,source_reservation_unchanged=True)
    return result


def check_away_from_gripper(facts, destination_delta_target_local):
    from .proximity import distance_to_box
    bounds=facts['target_bounds_local_m']
    shifted=(np.asarray(bounds)+np.asarray(destination_delta_target_local)).tolist()
    before={h:distance_to_box(p,bounds) for h,p in facts['gripper_centers_target_local_m'].items()}
    after={h:distance_to_box(p,shifted) for h,p in facts['gripper_centers_target_local_m'].items()}
    hand=min(before,key=before.get)
    if after[hand]<=before[hand] or min(after.values())<=min(before.values()):
        raise ValueError('Reserved destination is not farther from the nearby grippers')
    return dict(nearby_hand=hand,before_m=before,after_m=after,
                reserved_destination_retained=True)
