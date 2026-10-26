"""Read-only authored USD cache, invalidated on EVERY stage change notice.

No time quantization, dropped validation or persistent cache across episodes.
Native dynamic poses are not cached here. Matrices are defensive copies;
point tuples are immutable. Existing comparisons/thresholds stay in callers.
"""
import weakref
from pxr import Usd, UsdGeom, Tf, Gf

_caches=weakref.WeakKeyDictionary()

class StaticCache:
    def __init__(self,stage):
        self.stage=weakref.ref(stage)
        self.generation=0;self.reads=0;self.hits=0
        self.matrices={};self.points={};self.xforms={}
        self.notice=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self.invalidate,stage)
    def invalidate(self,*args):
        self.generation+=1
        self.matrices.clear();self.points.clear();self.xforms.clear()
    def matrix(self,path,time=None):
        key=(str(path),time);self.reads+=1
        if key not in self.matrices:
            stage=self.stage();prim=stage.GetPrimAtPath(path)
            if not prim:raise RuntimeError('Missing static prim '+str(path))
            if time not in self.xforms:
                self.xforms[time]=UsdGeom.XformCache(Usd.TimeCode.Default() if time is None else Usd.TimeCode(time))
            self.matrices[key]=self.xforms[time].GetLocalToWorldTransform(prim)
        else:self.hits+=1
        return Gf.Matrix4d(self.matrices[key])
    def mesh_points(self,path):
        key=str(path);self.reads+=1
        if key not in self.points:
            prim=self.stage().GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdGeom.Mesh):raise RuntimeError('Missing static mesh '+key)
            self.points[key]=tuple(map(tuple,UsdGeom.Mesh(prim).GetPointsAttr().Get()))
        else:self.hits+=1
        return self.points[key]

def for_stage(stage):
    if stage not in _caches:_caches[stage]=StaticCache(stage)
    return _caches[stage]

def close_stage(stage):
    cache=_caches.pop(stage,None)
    if cache is not None:cache.notice.Revoke()

def exact_pose_result(owner,pose,compute):
    """Only reuse an exact unchanged pose under unchanged authored geometry."""
    stage=owner.stage
    key=(for_stage(stage).generation,tuple(pose.position_m),tuple(pose.quaternion_wxyz))
    previous=getattr(owner,'_exact_pose_result',None)
    if previous is not None and previous[0]==key:return previous[1]
    value=compute(pose)
    owner._exact_pose_result=(key,value)
    return value
