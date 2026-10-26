// Constraint adapter over the existing planar placement geometry. No USD writes.
'use strict';
const crypto=require('node:crypto');
const EPS=1e-7; // Same numerical epsilon as upstream geometry, not physical clearance.
function check(ok,message){if(!ok)throw Error(message);}
function fields(o,allowed,required=[]){check(o&&typeof o==='object'&&!Array.isArray(o),'Expected object');check(Object.keys(o).every(k=>allowed.includes(k))&&required.every(k=>k in o),'Unknown/missing fields');}
function number(x){check(typeof x==='number'&&Number.isFinite(x),'Finite number required');return x;}
function pair(x){check(Array.isArray(x)&&x.length===2,'Expected two metre coordinates');return x.map(number);}
const angleDiff=(a,b)=>Math.abs(((a-b+180)%360+360)%360-180);
function edgeFrame(s,i){
 const a=s.polygon[i],b=s.polygon[(i+1)%s.polygon.length],d=[b[0]-a[0],b[1]-a[1]],length=Math.hypot(...d);
 check(length>EPS,'Degenerate edge');const u=d.map(x=>x/length),v=[-u[1],u[0]];
 return {origin:a,u,v,length,index:i,front:[-v[0],-v[1]]};
}
function localXY(world,f){const d=world.map((x,i)=>x-f.origin[i]);return [d[0]*f.u[0]+d[1]*f.u[1],d[0]*f.v[0]+d[1]*f.v[1]];}
function validateConstraint(c,layout){
 fields(c,['surface_group','surface_path','edge_index','local_xy_m','local_bounds_m','world_xy_m','front_yaw_deg','edge_gap_m']);
 for(const k of ['surface_group','surface_path'])if(k in c)check(typeof c[k]==='string'&&c[k].length,`${k} must be a string`);
 if('surface_group' in c)check(layout.surfaces.some(s=>s.group===c.surface_group),'Unknown surface group');
 if('surface_path' in c)check(layout.surfaces.some(s=>s.path===c.surface_path),'Unknown surface path');
 if('edge_index' in c)check(Number.isInteger(c.edge_index)&&c.edge_index>=0&&'surface_path' in c,'Edge requires exact surface path and nonnegative index');
 const selectors=['local_xy_m','local_bounds_m','world_xy_m'].filter(k=>k in c);
 check(selectors.length<=1,'Choose fixed local XY, local bounds, or fixed world XY, not several');
 if(selectors.some(k=>k.startsWith('local_')))check('surface_path' in c&&'edge_index' in c,'Local coordinates require an explicit surface and edge frame');
 if('local_xy_m' in c)pair(c.local_xy_m);
 if('world_xy_m' in c)pair(c.world_xy_m);
 if('local_bounds_m' in c){const b=c.local_bounds_m;check(Array.isArray(b)&&b.length===2,'Bounds must be [minimum, maximum]');pair(b[0]);pair(b[1]);check(b[0].every((x,i)=>x<=b[1][i]),'Reversed bounds');}
 if('front_yaw_deg' in c)number(c.front_yaw_deg);
 if('edge_gap_m' in c){pair(c.edge_gap_m);check(c.edge_gap_m[0]>=0&&c.edge_gap_m[0]<=c.edge_gap_m[1],'Invalid edge gap interval');}
 const surfaces=layout.surfaces.filter(s=>(!c.surface_group||s.group===c.surface_group)&&(!c.surface_path||s.path===c.surface_path));
 check(surfaces.length,'Surface group/path conflict');
 if('edge_index' in c)check(surfaces.every(s=>c.edge_index<s.polygon.length),'Edge index out of range');
 return surfaces;
}
function randomStream(seed,id){let s=crypto.createHash('sha256').update(`${seed}:${id}`).digest().readUInt32LE();return()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function candidates(station,layout,pack,G,seed){
 fields(station,['id','type','placement'],['id','type']);
 check(typeof station.id==='string'&&/^[A-Za-z][A-Za-z0-9_-]*$/.test(station.id),'Safe station ID required');
 check(['coffee','fryer','microwave','tray'].includes(station.type),'Only approved coffee/fryer/microwave/tray profiles supported here');
 const z=pack.zones.find(z=>z.id===station.type);check(z,'Missing upstream standard zone');
 const c=station.placement??{},surfaces=validateConstraint(c,layout),random=randomStream(seed,station.id),reject={};
 const gap=[Math.max(z.gap[0],c.edge_gap_m?.[0]??z.gap[0]),Math.min(z.gap[1],c.edge_gap_m?.[1]??z.gap[1])];
 const edges=surfaces.flatMap(s=>s.polygon.map((_,i)=>({s,i}))).filter(e=>!('edge_index' in c)||e.i===c.edge_index)
  .filter(e=>{const a=e.s.polygon[e.i],b=e.s.polygon[(e.i+1)%e.s.polygon.length];return Math.hypot(b[0]-a[0],b[1]-a[1])>.06;})
  .map(e=>({...e,f:edgeFrame(e.s,e.i)}));
 const good=[];function no(reason){reject[reason]=(reject[reason]||0)+1;}
 if(gap[0]>gap[1])return {good,rejections:{approved_gap_conflict:1}};
 const fixed='local_xy_m' in c||'world_xy_m' in c;
 const attempts=fixed?edges.length:1600; // Upstream solve's existing default budget.
 for(let i=0;i<attempts&&edges.length;i++){
  const e=fixed?edges[i]:edges[Math.floor(random()*edges.length)],f=e.f;
  const yaw=Math.atan2(f.front[1],f.front[0])*180/Math.PI;
  if('front_yaw_deg' in c&&angleDiff(yaw,c.front_yaw_deg)>EPS*180/Math.PI){no('front_not_facing_selected_edge');continue;}
  let xy;
  if('local_xy_m' in c)xy=c.local_xy_m;
  else if('world_xy_m' in c)xy=localXY(c.world_xy_m,f);
  else {
   const bounds=c.local_bounds_m??[[0,z.body.max[1]+gap[0]],[f.length,z.body.max[1]+gap[1]]];
   const lo=[Math.max(0,bounds[0][0]),Math.max(z.body.max[1]+gap[0],bounds[0][1])];
   const hi=[Math.min(f.length,bounds[1][0]),Math.min(z.body.max[1]+gap[1],bounds[1][1])];
   if(lo.some((x,j)=>x>hi[j])){no('range_does_not_intersect_approved_edge_gap');continue;}
   xy=lo.map((x,j)=>x+random()*(hi[j]-x));
  }
  const actualGap=xy[1]-z.body.max[1];
  if(xy[0]<-EPS||xy[0]>f.length+EPS){no('beyond_selected_edge');continue;}
  if(actualGap<gap[0]-EPS||actualGap>gap[1]+EPS){no('approved_gap_conflict');continue;}
  const p=G.geometry(z,e.s,e.s.polygon[e.i],e.s.polygon[(e.i+1)%e.s.polygon.length],xy[0]/f.length,actualGap);
  const reason=G.validate(p,layout);if(reason){no(reason);continue;}
  if((layout.station_obstacles||[]).some(b=>b.station===p.id&&p.objects.some(obj=>G.overlap(obj,G.rect(b))))){no('actual_scene_obstacle');continue;}
  good.push({...p,task_id:station.id,local_xy_m:[...xy],surface_edge_index:e.i,
   coordinate_frame:{origin_world_xy_m:f.origin,u_world:f.u,v_inward_world:f.v},
   constraint_receipt:{original_constraint:c,approved_edge_gap_m:z.gap,effective_edge_gap_m:gap,
    origin_reference:'standard station origin, NOT device bbox front',z_owner:'existing surface.top',
    assembly_policy:'rigid transform of approved initial assembly; do not independently move basket/cup/bag'}});
 }
 return {good,rejections:reject};
}
function build(request,layout,pack,G){
 fields(request,['layout_id','seed','stations'],['layout_id','seed','stations']);
 check(Number.isInteger(request.layout_id)&&request.layout_id===layout.id,'Exact layout ID required; no world-coordinate reuse on another layout');
 check(Number.isInteger(request.seed)&&request.seed>=1&&request.seed<=4294967295,'Seed must be 1..4294967295');
 check(Array.isArray(request.stations),'Explicit station list required');
 check(!layout.errors.length,'Layout geometry unavailable');
 const ids=request.stations.map(s=>s.id),types=request.stations.map(s=>s.type);
 check(new Set(ids).size===ids.length&&new Set(types).size===types.length,'One station per task family; duplicate ID/type');
 const lists=request.stations.map(s=>({id:s.id,...candidates(s,layout,pack,G,request.seed)})).sort((a,b)=>a.good.length-b.good.length);
 let solution=null,nodes=0,limited=false;
 function visit(i,chosen){if(i===lists.length){solution=chosen;return true;}if(++nodes>30000){limited=true;return false;}
  for(const p of lists[i].good){if(chosen.every(q=>G.compatible(p,q))&&visit(i+1,[...chosen,p]))return true;if(limited)return false;}return false;}
 visit(0,[]);
 return {schema:'openend.constrained_stations.plan.v1',request,status:solution?'planar_candidate':'constraints_unsatisfied_or_search_exhausted',
  placements:solution??[],candidate_counts:Object.fromEntries(lists.map(x=>[x.id,x.good.length])),
  rejections:Object.fromEntries(lists.map(x=>[x.id,x.rejections])),search_nodes:nodes,search_limit_reached:limited,
  partial_room_allowed:false,physics_validated:false,scene_generated:false,initial_assembly_instantiated:false,
  height_clearance_checked:false,planar_checks:'unchanged upstream validate and compatible',
  caveat:'Bounded planar search, not proof of infeasibility or full assembly clearance. Upper obstacles and coffee accessories need later validation.'};
}
module.exports={build,candidates,edgeFrame,localXY,validateConstraint};
