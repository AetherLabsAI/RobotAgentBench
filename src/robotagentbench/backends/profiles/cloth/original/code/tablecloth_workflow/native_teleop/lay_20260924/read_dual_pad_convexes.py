import json,numpy as np
from pxr import Usd,UsdGeom,UsdUtils,PhysicsSchemaTools
from omni.physx import get_physx_cooking_interface

def read_pad_convexes(stage,output,hand="r"):
 result={};cache=UsdGeom.XformCache();sid=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
 for side in ['inner','outer']:
  body='/World/Robot/gripper_'+hand+'_'+side+'_link4';root=stage.GetPrimAtPath(body);meshes=[x for x in Usd.PrimRange(root) if x.IsA(UsdGeom.Mesh) and x.HasAPI('PhysicsCollisionAPI')]
  rows=[]
  for prim in meshes:
   mat=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(root).GetInverse());row=dict(path=str(prim.GetPath()),mesh_to_body_row_major=mat.tolist(),convexes=[])
   def received(status,convexes):
    row['status']=str(status)
    for convex in convexes:
     v=np.array([[p.x,p.y,p.z]for p in convex.vertices]);vb=(np.c_[v,np.ones(len(v))]@mat)[:,:3];row['convexes'].append(dict(vertices_mesh=v.tolist(),vertices_body=vb.tolist(),indices=list(convex.indices)))
   get_physx_cooking_interface().request_convex_collision_representation(stage_id=sid,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,on_result=received)
   rows.append(row)
  result[side]=rows
 (output/('native_pad_cooked_convexes.json' if hand=='r' else 'native_left_pad_cooked_convexes.json')).write_text(json.dumps(result,indent=2)+'\n');print('Native pad collision representation',[(k,[len(r['convexes'])for r in v])for k,v in result.items()],flush=True)
