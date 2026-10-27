"""Auditable pre-physics hooks for a saved cloth variation, not a controller."""
import hashlib
import json
from pathlib import Path

from . import randomization as reset

INITIAL_MARKER = '    if args.commands:\n'
SCENE_MARKER = '    existing_cloth = [p for p in stage.Traverse() if p.HasAPI(PhysxSchema.PhysxParticleClothAPI)]\n'
INITIAL_HOOK = '''    # Seeded reset occurs after the unchanged original fold recipe, before audit.
    initial = CLOTH_RESET.initial(initial, cloth_ids, uv, faces)
'''
SCENE_HOOK = '''    # Background layout/colour only; before native cloth creation and physics.
    CLOTH_RESET.scene(stage)
'''


def extend_initialization(source):
    if source.count(INITIAL_MARKER)!=1 or source.count(SCENE_MARKER)!=1:
        raise ValueError('Original initialization boundary changed')
    result=source.replace(INITIAL_MARKER,INITIAL_HOOK+INITIAL_MARKER,1)
    result=result.replace(SCENE_MARKER,SCENE_HOOK+SCENE_MARKER,1)
    if result.replace(INITIAL_HOOK,'',1).replace(SCENE_HOOK,'',1)!=source:
        raise AssertionError('Unexpected edit outside task-local reset hooks')
    return result


def _snapshot(stage):
    """Compare every non-pose property too, including collision/drive/light data."""
    result={}
    for prim in stage.Traverse():
        result[str(prim.GetPath())]=dict(type=str(prim.GetTypeName()),schemas=list(prim.GetAppliedSchemas()),
            attributes={a.GetName():repr(a.Get()) for a in prim.GetAttributes()},
            relationships={r.GetName():list(map(str,r.GetTargets())) for r in prim.GetRelationships()})
    return result


def _check_scene_edits(before,after,moved,wall):
    if set(before)!=set(after):raise ValueError('Reset changed room prim inventory')
    changed=[]
    for path,old in before.items():
        new=after[path]
        if old['type']!=new['type'] or old['schemas']!=new['schemas'] or old['relationships']!=new['relationships']:
            raise ValueError('Reset changed a schema, material binding or relationship: '+path)
        for name in set(old['attributes'])|set(new['attributes']):
            if old['attributes'].get(name)==new['attributes'].get(name):continue
            if not ((path in moved and (name.startswith('xformOp:') or name=='xformOpOrder'))
                    or (path==wall and name=='inputs:diffuseColor')):
                raise ValueError('Reset changed a non-layout/non-colour property: '+path+'.'+name)
            if name.startswith('xformOp:scale'):
                raise ValueError('Reset changed a collision or visual scale')
            changed.append(path+'.'+name)
    return changed


class NativeReset:
    def __init__(self,*,record,baseline,prepared,output):
        reset.verify_initial(baseline,prepared,record['task_name'],record)
        reset.verify_scene_choices(record)
        self.record=record;self.baseline=baseline;self.prepared=prepared;self.output=Path(output)
        self.initial_applied=False;self.scene_applied=False

    def initial(self,initial,cloth_ids,uv,faces):
        import numpy as np
        if self.initial_applied:raise ValueError('Reset may run only once, before physics')
        for name,value in (('initial_cloth_q',initial),('cloth_id',cloth_ids),('uv',uv),('faces',faces)):
            if not np.array_equal(self.baseline[name],value):
                raise ValueError('Native fold recipe differs from the pinned baseline: '+name)
        q=reset.transform_initial(initial,cloth_ids,self.record['task_name'],self.record['poses'])
        if not np.array_equal(q,self.prepared['initial_cloth_q']):
            raise ValueError('Native reset differs from the saved instance')
        self.initial_applied=True
        return q

    def scene(self,stage):
        if not self.initial_applied or self.scene_applied:
            raise ValueError('Reset scene must run once after the original initial recipe')
        from pxr import Usd,UsdGeom,Gf
        from real2sim.production import house_layout
        layouts=house_layout.load_layouts(self.record['layout']['catalog'])
        layout=next(r for r in layouts if r.layout_id==self.record['layout']['id'])
        # Newer house catalogs include two optional sofa collision overlays.
        # The older native cloth house has neither. Do not add them or change
        # its collision inventory; apply the same SE(2) recipe to live roots.
        absent={p for paths in layout.cluster_prim_roots.values() for p in paths
                if not stage.GetPrimAtPath(p)}
        optional={'/World/Assets/SeatOverlay','/World/Assets/SeatRiser'}
        if absent-optional:raise ValueError('Required furniture root absent: '+repr(sorted(absent-optional)))
        if absent:
            from dataclasses import replace
            layout=replace(layout,cluster_prim_roots={k:tuple(p for p in paths if p not in absent)
                           for k,paths in layout.cluster_prim_roots.items()})
        if any(layout.transforms['coffee_table_cluster'].translate_xy_m) or layout.transforms['coffee_table_cluster'].yaw_deg:
            raise ValueError('Cloth workstation transform must stay identity')
        protected=['/World/Robot',*layout.cluster_prim_roots['coffee_table_cluster']]
        cache=UsdGeom.XformCache()
        old_pose={p:cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in protected}
        before=_snapshot(stage)
        disposable=self.output/'prepared_room_layout.usda'
        if disposable.exists():raise FileExistsError(disposable)
        if not stage.GetRootLayer().Export(str(disposable)):
            raise RuntimeError('Cannot export task-local anonymous room layer')
        result=house_layout.apply_scene_overlay(disposable,layout,frame='coffee_table',robot_path=None)
        edited=Usd.Stage.Open(str(disposable))
        wall='/World/Background/Shell/WallMat/S'
        colour=edited.GetPrimAtPath(wall).GetAttribute('inputs:diffuseColor')
        if not colour or not colour.Set(Gf.Vec3f(*self.record['appearance']['wall_rgb'])):
            raise ValueError('Original wall colour shader is missing')
        edited.GetRootLayer().Save()
        after=_snapshot(edited)
        moved={row['path'] for row in result['authored_prims']}
        changed=_check_scene_edits(before,after,moved,wall)
        cache=UsdGeom.XformCache()
        for path,pose in old_pose.items():
            if not Gf.IsClose(pose,cache.GetLocalToWorldTransform(edited.GetPrimAtPath(path)),1e-12):
                raise ValueError('Reset changed the task workstation or robot frame: '+path)
        stage.GetRootLayer().TransferContent(edited.GetRootLayer())
        self.scene_applied=True
        receipt=dict(schema='robotagentbench.cloth_native_reset.v1',seed=self.record['seed'],
            task_name=self.record['task_name'],poses=self.record['poses'],layout=result,
            appearance=self.record['appearance'],changed_properties=changed,
            physical_properties_unchanged=True,workstation_unchanged=True,
            source_edited=False,native_settling_accepted=False,
            absent_optional_catalog_roots=sorted(absent),collision_inventory_preserved=True,
            saved_initial_sha256=self.record['initial_sha256'],
            room_layer_sha256=hashlib.sha256(disposable.read_bytes()).hexdigest())
        (self.output/'native_reset.json').write_text(json.dumps(receipt,indent=2)+'\n')
