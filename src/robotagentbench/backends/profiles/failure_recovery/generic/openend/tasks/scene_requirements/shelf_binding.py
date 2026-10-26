"""Read-only remapping of original geometry to unified all_assets instances."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from scipy.spatial import ConvexHull
from evaluation.tasks.shelf_pick.targets import SPECS
from openend.tasks.scene_requirements.shelf_geometry import Geometry, rotation
from evaluation.tasks.shelf_pick.shelf_collision import SupportFace
from evaluation.tasks.shelf_pick.physical_detection import DetectionConfig
from evaluation.tasks.coffee_table_place.isaac_evidence import TableGeometry
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS, normalize_parameters
from openend.tasks.scene_requirements.shelf_composite import Binding


def one(values,role):
    if len(values)!=1:raise ValueError(f'{role}: expected one, got {len(values)}')
    return values[0]


def collider(p):
    return bool(p and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI)
                and p.GetAttribute('physics:collisionEnabled').Get() is not False)


class SceneBinding:
    def __init__(self,stage,manifest,product,episode_id,instance_id):
        normalize_parameters({'product':product})
        self.stage=stage;self.product=PRODUCTS[product]
        objects=manifest['objects']
        self.shelf=one([o['path'] for o in objects if o['id']=='snack_shelf'],'shelf')
        self.tray=one([o['path'] for o in objects if o['id']=='japanese_tray'],'wooden tray')
        self.stock={};self.wrappers=[];self.supports={};declared_targets=[]
        self.body_by_wrapper={}
        shelf_prim=stage.GetPrimAtPath(self.shelf)
        if not shelf_prim:raise ValueError('Missing shelf')
        self.shelf_colliders=tuple(str(p.GetPath()) for p in Usd.PrimRange(shelf_prim) if collider(p))
        boards={tier:one([p for p in self.shelf_colliders if p.endswith('/'+name)],tier)
                for tier,name in [('tier2','Tier2Overlay'),('tier3','Tier3Overlay')]}
        for o in objects:
            if not o['path'].startswith(self.shelf+'_tier'):continue
            key=o['id'];wrapper=o['path'];p=stage.GetPrimAtPath(wrapper)
            if not p:raise ValueError('Missing stock wrapper '+wrapper)
            body=one([x for x in Usd.PrimRange(p) if x.HasAPI(UsdPhysics.RigidBodyAPI)],key+' body')
            if body.GetAttribute('physics:kinematicEnabled').Get() is True or body.GetAttribute('physics:rigidBodyEnabled').Get() is False:
                raise ValueError('Stock must remain dynamic: '+key)
            path=str(body.GetPath())
            if path in self.stock:raise ValueError('Duplicate manifest body')
            # Unified manifest must refer to its catalog asset, not a mislabeled instance.
            entry=Path(o['entry'])
            if entry.parent.name!=key or entry.parent.parent.name!='all_assets':
                raise ValueError('Stock identity disagrees with all_assets entry')
            self.stock[path]=key;self.wrappers.append(wrapper)
            self.body_by_wrapper[wrapper]=path
            if o.get('task_role')=='requested_full_product' and o.get('task_id')==instance_id and key==product:
                declared_targets.append(path)
            tier='tier3' if wrapper.startswith(self.shelf+'_tier3_') else 'tier2'
            self.supports[path]=boards[tier]
        aliases={t.rsplit('/',1)[-1] for t in SPECS[self.product.pick_api].targets}
        targets=tuple(p for p in declared_targets if self.stock[p] in aliases)
        if len(targets)!=1:raise ValueError('Exactly one explicit target instance per shelf task required')
        if not targets:raise ValueError('Requested product absent from unified scene')
        support=boards[self.product.tier]
        if any(self.supports[p]!=support for p in targets):raise ValueError('Wrong source tier')
        self.identity=Binding(episode_id,instance_id,product,targets,support,self.tray)
        spec=replace(SPECS[self.product.pick_api],targets=targets,support=support,external_target=True)
        self.workspace=Geometry(stage,spec,stock_roots=self.wrappers,targets=targets,supports=list(boards.values()))
        prim=stage.GetPrimAtPath(support)
        if not prim.IsA(UsdGeom.Cube):raise ValueError('Source board is not original box support')
        tf=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        self.face=SupportFace(tf,UsdGeom.Cube(prim).GetSizeAttr().Get(),DetectionConfig().support_face_tolerance_m,support)
        self.tray_geometry=TrayGeometry(stage,self.tray,targets,self.product.place_api)
        self.static={p:UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)
                     for p in self.shelf_colliders}

    def validate_static(self):
        self.workspace.validate_static(self.stage)
        self.tray_geometry.check_destination_unchanged()
        if any(UsdGeom.Xformable(self.stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)!=m for p,m in self.static.items()):
            raise RuntimeError('Shelf collision geometry changed')


class TrayGeometry:
    """Original exact mesh-floor contour in the new static tray placement."""
    def __init__(self,stage,tray,targets,place_api):
        self.stage=stage
        root=stage.GetPrimAtPath(tray)
        if not root:raise ValueError('Missing tray')
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(root)):
            raise ValueError('This binding requires the existing static wooden tray')
        mesh=one([p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and collider(p)],'tray collider')
        self.path=str(mesh.GetPath());self.table_tf=UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0)
        asset=stage.GetPrimAtPath(tray+'/Asset')
        if not asset:raise ValueError('Missing canonical tray asset root')
        asset_tf=UsdGeom.Xformable(asset).ComputeLocalToWorldTransform(0).RemoveScaleShear()
        from evaluation.tasks.coffee_table_place import isaac_evidence
        source=Path(isaac_evidence.__file__).resolve().parent/'resources/newdata_20260918'/place_api/'fixed.json'
        reference=json.loads(source.read_text())['tray']
        # Preserve the original accepted floor plane relative to the canonical
        # tray rigid frame, instead of selecting the outer rim or global table Z.
        sr=rotation(reference['quaternion_wxyz'])
        local=sr.inv().apply([0.,0.,reference['floor_z_m']-reference['position_m'][2]])
        normal=np.asarray(asset_tf.TransformDir(Gf.Vec3d(*sr.inv().apply([0.,0.,1.]))))
        if not np.allclose(normal,[0,0,1],atol=1e-6,rtol=0):raise ValueError('Tilted tray not supported by original horizontal-floor reader')
        expected=float(asset_tf.Transform(Gf.Vec3d(*local))[2])
        a=np.asarray([self.table_tf.Transform(Gf.Vec3d(*v)) for v in mesh.GetAttribute('points').Get()])
        floor=a[np.abs(a[:,2]-expected)<=1e-6]
        if len(floor)<3 or np.ptp(floor[:,2])>1e-7:raise ValueError('Original tray floor contract does not match current mesh')
        self.top=float(floor[:,2].mean());self.region=ConvexHull(floor[:,:2]).equations
        self.destination_inverse=self.table_tf.GetInverse()
        self.vertices={p:TableGeometry.body_vertices(stage,p) for p in targets}
        self.source=str(source)

    pose_in_destination=TableGeometry.pose_in_destination
    check_destination_unchanged=TableGeometry.check_destination_unchanged

    def in_region(self,path,position,q):
        a=rotation(q).apply(self.vertices[path])+position
        return bool(np.all(a[:,:2]@self.region[:,:2].T+self.region[:,2]<=1e-7))
