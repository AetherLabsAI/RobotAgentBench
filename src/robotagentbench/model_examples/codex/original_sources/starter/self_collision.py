"""Robot-only predictive collision query, inspired by YAM CollisionApprover.

No physics stepping, environment geometry or live hidden state. This editable
guard is not a trusted safety boundary and covers only the published primitives.
"""
import json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

class PrimitiveSelfGuard:
    def __init__(self,directory=None,*,clearance=.002,sweep_resolution=.005):
        if not np.isfinite(clearance) or clearance<0 or not np.isfinite(sweep_resolution) or sweep_resolution<=0:raise ValueError('invalid guard settings')
        directory=Path(directory) if directory is not None else Path(__file__).parent/'robot'
        self.description=json.loads((directory/'robot_self_primitives.json').read_text())
        tree=ET.parse(directory/'robot_arm_kinematics.xml');root=tree.getroot();bodies={b.get('name'):b for b in root.iter('body')}
        for g in self.description['geometry']:
            node=bodies[g['body']]
            ET.SubElement(node,'geom',name=g['name'],type=g['type'],size=self._nums(g['size']),pos=self._nums(g['pos']),quat=self._nums(g['quat']),contype='0',conaffinity='0',mass='0')
        contacts=ET.SubElement(root,'contact')
        for x,y in self.description['pairs']:
            ET.SubElement(contacts,'pair',geom1=x,geom2=y,condim='1',margin=str(clearance),gap=str(clearance))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));self.data=mujoco.MjData(self.model)
        self.names=tuple(self.model.joint(i).name for i in range(self.model.njnt))
        self.clearance=clearance;self.sweep_resolution=sweep_resolution

    @staticmethod
    def _nums(v):return ' '.join(format(float(x),'.17g') for x in v)

    def check(self,q):
        if isinstance(q,dict):q=np.array([q[n] for n in self.names])
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.model.nq,) or not np.isfinite(q).all():raise ValueError('invalid robot state for collision query')
        self.data.qpos[:]=q;mujoco.mj_kinematics(self.model,self.data);mujoco.mj_collision(self.model,self.data)
        for c in self.data.contact[:self.data.ncon]:
            if c.dist<self.clearance:
                return dict(pair=[self.model.geom(int(c.geom1)).name,self.model.geom(int(c.geom2)).name],distance_m=float(c.dist),coverage=self.description['scope'])
        return None

    def sweep(self,start,target):
        start=np.asarray(start);target=np.asarray(target)
        count=max(1,math.ceil(float(np.max(np.abs(target-start)))/self.sweep_resolution))
        if count>2000:raise ValueError('collision sweep exceeds bounded query budget')
        for i in range(count+1):
            hit=self.check(start+(target-start)*i/count)
            if hit:return hit
        return None
