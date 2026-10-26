// Reuse the existing table footprint, padding and polygon separation policy.
const fs=require('fs'),path=require('path');
const [source,input]=process.argv.slice(2);
const C=require(path.join(source,'openend/tasks/scene_requirements/placement_sources/coffee-table-boxes-v1.js'));
const region=JSON.parse(fs.readFileSync(path.join(source,'openend/tasks/scene_requirements/placement_sources/room_preview_plan_seed42.json'),'utf8')).coffee_table;
const plan=JSON.parse(fs.readFileSync(input,'utf8'));
const width=region.end-region.start,depth=region.right-region.left;
let s=plan.seed>>>0;
const rng=()=>{s=(s+0x6D2B79F5)>>>0;let t=Math.imul(s^s>>>15,s|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296};
let placed=[];
for(let restart=0;restart<30;restart++){
 placed=[];
 for(const a of [...plan.candidates].sort((a,b)=>b.dimensions_m[0]*b.dimensions_m[1]-a.dimensions_m[0]*a.dimensions_m[1])){
  const size=a.dimensions_m;
  for(let attempt=0;attempt<600;attempt++){
   const yaw=rng()*2*Math.PI,shape=C.boxCorners([0,0],size,yaw);
   const hx=Math.max(...shape.map(p=>Math.abs(p[0]))),hy=Math.max(...shape.map(p=>Math.abs(p[1])));
   if(2*hx>width||2*hy>depth)continue;
   const center=[hx+rng()*(width-2*hx),hy+rng()*(depth-2*hy)];
   const p={instance_id:a.instance_id,center,yaw,corners:C.boxCorners(center,size,yaw),protection:C.boxCorners(center,size,yaw,C.padding)};
   if(placed.every(q=>C.pairClear(p,q))){p.world_center=region.u.map((u,i)=>u*(region.start+center[0])+region.v[i]*(region.left+center[1]));placed.push(p);break;}
  }
 }
 if(placed.length===plan.candidates.length)break;
}
if(placed.length!==plan.candidates.length)throw Error('Cannot fit all candidates; no dropping/resizing/spacing relaxation');
console.log(JSON.stringify({region,padding:C.padding,placements:placed}));
