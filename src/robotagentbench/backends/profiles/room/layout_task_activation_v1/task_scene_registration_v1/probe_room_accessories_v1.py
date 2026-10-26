"""Read-only identify room accessory roots and authored physics."""
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics

API=Path('/home/lingjun/robot_agent_arena/api-protocol')
ep=json.loads((API/'evaluation/tasks/air_fryer/resources/source_episodes/2.1/ruleyellowinner-42032/episode.json').read_text())
sources={
 'yellow_cake':str(API/'evaluation/tasks/air_fryer/resources/producer_assets_20260922/cake_wide125_v1/plain_cake_a.usda'),
 'white_cake':str(API/'evaluation/tasks/air_fryer/resources/producer_assets_20260922/cake_wide125_v1/frosted_cake.usda'),
 'layout':'/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/ID_01_LIVING_STRAIGHT.usdc',
 'cloth':str(API/'evaluation/tasks/tabletop_pick/book_cloth/source_16_seed1000000/initial_scene.usdc'),
}
for label,src in sources.items():
 stage=Usd.Stage.Open(src);assert stage,src
 print('SOURCE',label,src,flush=True)
 paths=([str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Task66_67') and len(p.GetPath().GetPrefixes())<=5]
 if label=='fryer' else ['/World/FoodTray','/World/Cutlery','/World/Cutlery/Holder','/World/Assets/DiningTableMicrowave'] if label=='layout'
 else ['/World/Assets/Task69Items/DirtyTablecloth'] if label=='cloth' else [str(stage.GetDefaultPrim().GetPath())])
 for path in paths:
  p=stage.GetPrimAtPath(path)
  if not p:print('MISSING',path);continue
  r=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(p).ComputeAlignedRange()
  children=[str(c.GetPath()) for c in p.GetChildren()]
  bodies=[str(c.GetPath()) for c in Usd.PrimRange(p) if c.HasAPI(UsdPhysics.RigidBodyAPI)]
  collisions=[str(c.GetPath()) for c in Usd.PrimRange(p) if c.HasAPI(UsdPhysics.CollisionAPI)]
  print(json.dumps(dict(path=path,type=p.GetTypeName(),children=children,bodies=bodies,colliders=collisions,
   bounds=None if r.IsEmpty() else [list(r.GetMin()),list(r.GetMax())],apis=list(p.GetAppliedSchemas()),
   refs=str(p.GetMetadata('references')))),flush=True)
