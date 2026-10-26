"""Product selection and frozen-joint selection are independent of object poses."""
import hashlib
import random
from ..shelf_drinks_v1.catalog import APIS,LABELS
POLICY='frozen_robot_continuous_shelf_objects.v2'

def generator(seed,channel):
    if type(seed) is not int or not 1<=seed<2**32:raise ValueError('Positive uint32 seed')
    return random.Random(int.from_bytes(hashlib.sha256(f'{POLICY}:{seed}:{channel}'.encode()).digest(),'big'))

def select(bank,seed,api=None,profile_id=None):
    if api is not None and api not in APIS:raise ValueError('Existing beverage APIs only')
    api=api or generator(seed,'product').choice(APIS)
    profiles=bank['cases']['shelf_drink']['profiles']
    if profile_id is None:profile=generator(seed,'robot_pose').choice(profiles)
    else:
        rows=[p for p in profiles if p['id']==profile_id]
        if len(rows)!=1:raise ValueError('Unknown diagnostic pose')
        profile=rows[0]
    return api,profile
