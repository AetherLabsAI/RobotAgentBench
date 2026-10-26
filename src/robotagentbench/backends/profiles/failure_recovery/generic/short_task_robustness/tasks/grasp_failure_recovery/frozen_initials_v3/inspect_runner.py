import argparse
from .. import sim2_runner as io
from .runner import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('shelf_drink','coffee_bag','sofa'),required=True)
    p.add_argument('--inventory-key',required=True);p.add_argument('--key',required=True);a=p.parse_args()
    for k in (a.inventory_key,a.key):
        if not k.replace('_','').isalnum():raise ValueError('Simple own key required')
    raise SystemExit(run('inspect',['--variant',a.variant,'--inventory',str(io.GUEST/'frozen_initials_v3/mine'/a.inventory_key/'inventory.json'),
        '--output',str(io.GUEST/'frozen_initials_v3/inspect'/a.key)],a.key))
