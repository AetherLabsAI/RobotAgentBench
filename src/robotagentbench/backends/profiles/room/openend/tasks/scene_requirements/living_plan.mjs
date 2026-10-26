import fs from 'node:fs';
import {createRequire} from 'node:module';
import {combination} from './placement_sources/small-table-combinations.mjs';
const require=createRequire(import.meta.url),root=new URL('./placement_sources/',import.meta.url);
const read=n=>JSON.parse(fs.readFileSync(new URL(n,root),'utf8'));
const S=require('./placement_sources/sofa-placement-geometry.js'),C=require('./placement_sources/coffee-table-boxes-v1.js');
const boxes=read('sofa-object-boxes-v1.json').objects,policy={...read('sofa-placement-policy-v1.json'),structural_obstacles:read('sofa-structural-obstacles-v2.json').obstacles};
const rand=seed=>{let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=Math.imul(s^s>>>15,s|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296}};
export function build(inventory){
 const rng=rand(inventory.seed),objects=inventory.objects.map(o=>({...o,source:o.source||
  (o.trash_category?'coffee_table.objects':['wallet','glasses_case','remote_tv'].includes(o.asset_id)?'small_table':'sofa')}));
 const sofa=objects.filter(o=>o.source==='sofa'),small=objects.filter(o=>o.source==='small_table');
 const plan=S.solve(sofa.map(o=>({...boxes.find(b=>b.id===o.asset_id),id:o.instance_id})),policy,inventory.seed);
 if(plan.missing.length)throw Error('沙发摆放搜索未找到全部目标的解：'+plan.missing.join(','));
 const sm=small.length?combination(read('small-table-candidates-v1.json'),inventory.seed):{};
 const table=objects.filter(o=>o.source==='coffee_table.objects').map(o=>{
  const a=C.assets.find(a=>a.id===o.asset_id);if(!a)throw Error('No table sampler for '+o.asset_id);
  const paperVariant=o.asset_id==='paper_ball'?(rng()<.5?'large':'small'):null;
  const upright=o.asset_id==='paper_cup'&&rng()<.5;
  return {...o,assetId:o.asset_id,size:paperVariant?[...C.paperVariants[paperVariant].authored_pose_size_m]:upright?[a.size[0],a.size[2],a.size[1]]:a.size,
   ...(paperVariant?{paperVariant,effectiveMeshAxisScale:C.paperVariants[paperVariant].effective_mesh_axis_scale}:{}),rollRad:upright?Math.PI/2:0};
 });
 const region=read('room_preview_plan_seed42.json').coffee_table;
 const width=region.end-region.start,depth=region.right-region.left;
 let placed=[];
 for(let restart=0;restart<30;restart++){
  placed=[];
  for(const a of [...table].sort((a,b)=>b.size[0]*b.size[1]-a.size[0]*a.size[1])){
   for(let attempt=0;attempt<600;attempt++){
    const yaw=rng()*2*Math.PI,shape=C.boxCorners([0,0],a.size,yaw);
    const hx=Math.max(...shape.map(p=>Math.abs(p[0]))),hy=Math.max(...shape.map(p=>Math.abs(p[1])));
    if(2*hx>width||2*hy>depth)continue;
    const center=[hx+rng()*(width-2*hx),hy+rng()*(depth-2*hy)],corners=C.boxCorners(center,a.size,yaw);
    const p={...a,center,yaw,corners,protection:C.boxCorners(center,a.size,yaw,C.padding)};
    if(placed.every(q=>C.pairClear(p,q))){p.world_center=region.u.map((u,i)=>u*(region.start+center[0])+region.v[i]*(region.left+center[1]));placed.push(p);break}
   }
  }
  if(placed.length===table.length)break;
 }
 if(placed.length!==table.length)throw Error('茶几摆放失败；未减少目标或放宽间距');
 return {objects,sofa:plan.placements,small_table:small.map(o=>({...o,placement:sm[o.asset_id==='remote_tv'?'remote':o.asset_id]})),coffee_table:placed};
}
