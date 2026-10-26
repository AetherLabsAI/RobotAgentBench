"""Supplementary touch evidence; anonymous events are ignored, never fatal."""
import math


def under(path,root):
    return path==root or path.startswith(root.rstrip('/')+'/')


class ContactEdges:
    def __init__(self, declarations, fingers, decode, event_types):
        self.declarations=declarations;self.fingers=fingers;self.decode=decode
        self.found,self.persist,self.lost=event_types
        self.pairs={};self.paths={};self.touching=set()

    def update(self,report):
        headers,data=report[:2]
        for h in headers:
            # Same conservative policy as the public contact adapter. No path
            # fallback and no guessed pair for an anonymous separation event.
            ids=[getattr(h,k,None) for k in ('stage_id','actor0','actor1','collider0','collider1')]
            if any(x is None or int(x)<=0 for x in ids):continue
            key=(int(h.stage_id),*sorted((int(h.collider0),int(h.collider1))))
            if h.type==self.lost:
                self.pairs.pop(key,None);continue
            if key not in self.paths:
                self.paths[key]=(str(self.decode(h.collider0)),str(self.decode(h.collider1)))
            paths=self.paths[key];matched=set()
            for target,row in self.declarations.items():
                for hand,fingers in self.fingers.items():
                    if any(any(under(paths[i],f) for f in fingers) and
                           any(under(paths[1-i],r) for r in row['roots']) for i in (0,1)):
                        matched.add((target,hand))
            if not matched:continue
            if h.type not in (self.found,self.persist):raise ValueError('Unknown relevant contact event')
            offset,count=int(h.contact_data_offset),int(h.num_contact_data)
            if count<0 or offset<0 or offset+count>len(data):raise ValueError('Invalid relevant contact range')
            force=False
            for point in data[offset:offset+count]:
                values=tuple(float(getattr(point.impulse,k)) for k in ('x','y','z'))
                if not all(math.isfinite(x) for x in values):raise ValueError('Invalid contact impulse')
                force |= sum(x*x for x in values)>1e-18
            self.pairs[key]=matched if force else set()
        current=set().union(*self.pairs.values()) if self.pairs else set()
        entered=current-self.touching
        self.touching=current
        return entered
