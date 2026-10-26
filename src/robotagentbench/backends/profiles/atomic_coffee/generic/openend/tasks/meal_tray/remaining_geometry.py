"""Read-only source bindings. No wooden tray or microwave dependency."""
from types import SimpleNamespace
from dataclasses import replace
import itertools
import numpy as np

class SaladSource:
    def __init__(self,stage,row):
        from pxr import UsdGeom,UsdPhysics,Gf
        from scipy.spatial.transform import Rotation
        from evaluation.tasks.refrigerator_take.random_reset import PARTS
        from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.scene import FRIDGE,DOORS,vertices
        self.stage=stage;self.target=row['path'];self.doors=DOORS;self.root=FRIDGE
        if row.get('initialization_owner')!='unified_scene_fridge_boxes_v1':raise ValueError('Unknown salad initial recipe')
        slot=row.get('slot')
        if slot not in ('low_out','low_in','up_out','up_in'):raise ValueError('Missing salad compartment')
        self.source_paths=tuple(DOORS[1]+'/'+p for p in PARTS['shelf4' if slot.startswith('up') else 'bin34'])
        cache=UsdGeom.XformCache();inv=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DOORS[1])).GetInverse()
        points=[]
        for path in self.source_paths:
            p=stage.GetPrimAtPath(path)
            if not p or not p.IsA(UsdGeom.Cube):raise ValueError('Missing original compartment '+path)
            h=UsdGeom.Cube(p).GetSizeAttr().Get()/2;m=cache.GetLocalToWorldTransform(p)*inv
            points.extend(m.Transform(Gf.Vec3d(*v)) for v in itertools.product((-h,h),repeat=3))
        points=np.asarray(points);self.lower,self.upper=points.min(0),points.max(0)
        self.vertices=vertices(stage,self.target);self.joints={}
        for name in ('left','right'):
            joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(FRIDGE+'/'+name+'_door_hinge'))
            if not joint:raise ValueError('Missing fridge joint')
            bodies=[joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()]
            if any(len(b)!=1 for b in bodies):raise ValueError('Unbound hinge')
            frames=[]
            for getter in (joint.GetLocalRot0Attr,joint.GetLocalRot1Attr):
                q=getter().Get();frames.append(Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix())
            self.joints[name]=(tuple(str(b[0]) for b in bodies),frames,str(joint.GetAxisAttr().Get()))

class DrinkSource:
    def __init__(self,stage,manifest,product,target):
        from pxr import Usd,UsdGeom,UsdPhysics
        from evaluation.tasks.shelf_pick.targets import SPECS
        from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
        from evaluation.tasks.shelf_pick.shelf_collision import SupportFace
        from evaluation.tasks.shelf_pick.physical_detection import DetectionConfig
        from openend.tasks.scene_requirements.shelf_binding import one,collider
        from openend.tasks.scene_requirements.shelf_geometry import Geometry
        self.shelf=one([o['path'] for o in manifest['objects'] if o['id']=='snack_shelf'],'shelf')
        self.stock={};self.supports={};self.body_by_wrapper={};wrappers=[]
        self.shelf_colliders=tuple(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(self.shelf)) if collider(p))
        boards={tier:one([p for p in self.shelf_colliders if p.endswith('/'+name)],tier)
            for tier,name in [('tier2','Tier2Overlay'),('tier3','Tier3Overlay')]}
        for row in manifest['objects']:
            if not row['path'].startswith(self.shelf+'_tier'):continue
            root=stage.GetPrimAtPath(row['path'])
            body=one([p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)],'stock body')
            if body.GetAttribute('physics:kinematicEnabled').Get() is True:raise ValueError('Non-dynamic stock')
            path=str(body.GetPath());wrappers.append(row['path']);self.stock[path]=row['id'];self.body_by_wrapper[row['path']]=path
            tier='tier3' if row['path'].startswith(self.shelf+'_tier3_') else 'tier2';self.supports[path]=boards[tier]
        if target not in self.stock or self.stock[target]!=product:raise ValueError('Wrong stock target')
        self.support=boards[PRODUCTS[product].tier]
        if self.supports[target]!=self.support:raise ValueError('Wrong drink source tier')
        spec=replace(SPECS[PRODUCTS[product].pick_api],targets=(target,),support=self.support,external_target=True)
        self.workspace=Geometry(stage,spec,stock_roots=wrappers,targets=(target,),supports=list(boards.values()))
        prim=stage.GetPrimAtPath(self.support)
        self.face=SupportFace(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0),
            UsdGeom.Cube(prim).GetSizeAttr().Get(),DetectionConfig().support_face_tolerance_m,self.support)
