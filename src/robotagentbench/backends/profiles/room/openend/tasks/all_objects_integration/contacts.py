"""Native pair reduction for source penalties, real hooks and exact targets."""
import math
from openend.tasks.glasses_transfer.contacts import reduce_pairs,under

def reduce(pairs,*,target,destination,cabinet,floors,robot,pads,mode):
    pairs=list(pairs)
    c=reduce_pairs(pairs,target=target,source=None,bag=destination,cabinet=cabinet,
                   floors=floors,robot=robot,pads=pads)
    furniture=set();loads={};supports=set(c['supports'])
    from evaluation.tasks.tidy_sofa.hang_hat.process import ALLOWED_HOOKS
    for paths,samples in pairs:
        for i in (0,1):
            subject,other=paths[i],paths[1-i]
            if under(subject,robot) and not under(other,robot) and not under(other,target):
                # Touching the movable drawer is allowed; the table frame is not.
                if mode!='drawer' or not under(other,destination):furniture.add(other)
            if mode!='hang' or not under(subject,target):continue
            sign=1 if i==0 else -1
            vertical=sum(sign*n[2]*math.sqrt(sum(x*x for x in impulse)) for n,_,impulse in samples)
            branch=next((b for b in ALLOWED_HOOKS if any(under(other,destination+'/'+b+part) for part in ('Shaft','Tip'))),None)
            if branch:loads[branch]=loads.get(branch,0.)+vertical
            # Preserve old hat rule: even a small positive upward load counts as foreign support.
            if vertical>0 and not under(other,robot):supports.add(destination if under(other,destination) else other)
    c.update(supports=frozenset(supports),furniture_contacts=frozenset(furniture),
             supporting_hooks=frozenset(b for b,v in loads.items() if v>0))
    return c
