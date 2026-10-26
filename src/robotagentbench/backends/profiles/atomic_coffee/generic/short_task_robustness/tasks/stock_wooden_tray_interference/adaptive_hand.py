"""One common motion envelope for all cylinders; native contacts stop digits.

No product-specific finger angles or target scale changes. Skin is reconstructed
from the same independent bone transforms that drive the anatomical colliders.
"""
import json
import numpy as np
from .human_asset import HumanAsset,transform_points,HERE
from .contact_closure import DIGITS,digit_for_bone


class AdaptiveHand(HumanAsset):
    def __init__(self):
        super().__init__(HERE/'assets/derived/left_hand_side_entry_012.json')
        rest=json.loads((HERE/'static_short_hand_011/cropped_rig.json').read_text())
        self.rest_points=np.asarray(rest['frames'][0]['points'])
        self.rest_inverse={k:np.linalg.inv(v) for k,v in rest['rest_bones'].items()}
        self.weights={k:np.array([w.get(k,0.) for w in rest['skin_weights']]) for k in self.bones}
        self.support={k:np.flatnonzero(w) for k,w in self.weights.items()}

    def sample(self,closure):
        values=closure if isinstance(closure,dict) else {d:closure for d in DIGITS}
        if set(values)!=set(DIGITS):raise ValueError('All five digit closures required')
        samples={v:super(AdaptiveHand,self).sample(v)[1] for v in set(values.values())}
        base=super().sample(0.)[1]
        bones={k:(samples[values[digit_for_bone(k)]][k] if digit_for_bone(k) else base[k]) for k in self.bones}
        p=np.zeros_like(self.rest_points)
        for k,m in bones.items():
            ix=self.support[k]
            p[ix]+=transform_points(m@self.rest_inverse[k],self.rest_points[ix])*self.weights[k][ix,None]
        return p,bones

    def palm_center(self,radius):
        if not np.isfinite(radius) or radius<=0:raise ValueError('Invalid measured radius')
        p,b=self.sample(0.)
        palm=p[self.collision_groups()['hand.R']]
        # Preserve common anatomical anchor; change only wrist placement to
        # accommodate measured object width. Full frozen palm hull is used.
        x,y=self.data['grip_center_local'][:2]
        eligible=palm[np.abs(palm[:,1]-y)<radius]
        if not len(eligible):raise ValueError('No palm patch at object section')
        z=float(np.max(eligible[:,2]+np.sqrt(radius**2-(eligible[:,1]-y)**2)))
        return np.array([x,y,z])
