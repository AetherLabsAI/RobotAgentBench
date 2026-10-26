"""Existing shelf beverage identities only; no snack/remote substitution."""
import hashlib
import random

APIS=tuple('4-'+str(i) for i in range(1,9))
LABELS={'4-1':'酒精饮料','4-2':'橙色运动饮料','4-3':'蓝色运动饮料',
        '4-4':'含咖啡因饮料','4-5':'儿童果汁','4-6':'气泡水','4-7':'可乐','4-8':'雪碧'}
POLICY='frozen_multidrink_empty_grasp.v1'

def rng(seed,channel):
    if type(seed) is not int or not 1<=seed<2**32:raise ValueError('Positive uint32 seed required')
    return random.Random(int.from_bytes(hashlib.sha256(f'{POLICY}:{seed}:{channel}'.encode()).digest(),'big'))

def choose(bank,seed,api=None):
    if bank.get('schema')!='failure_recovery.shelf_drinks.bank.v1':raise ValueError('Wrong bank')
    if api is not None and api not in APIS:raise ValueError('Not a beverage API')
    api=api or rng(seed,'product').choice(APIS)
    rows=bank['products'][api]['profiles']
    if not rows:raise ValueError('Selected product has no accepted frozen initial; no product fallback: '+api)
    return api,rng(seed,'profile:'+api).choice(rows)
