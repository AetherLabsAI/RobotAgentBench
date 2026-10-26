// Scene-family selection -> jointly validated workspace plans, not runtime registration.
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const web=path.resolve(__dirname,'../web');
const G=require(path.join(web,'random-placement-geometry.js'));
const read=name=>JSON.parse(fs.readFileSync(path.join(web,name),'utf8'));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(path.join(web,file))).digest('hex');
function build(request){
 if(!request||typeof request!=='object'||Array.isArray(request)||Object.keys(request).some(k=>!['layout_id','seed','task_types'].includes(k)))throw Error('Expected layout_id, seed, task_types only');
 if(!Number.isInteger(request.layout_id))throw Error('Explicit integer layout_id required');
 if(!Number.isInteger(request.seed)||request.seed<1||request.seed>4294967295)throw Error('Seed must be 1..4294967295');
 const all=read('four-zone-pack-v1.json'),data=read('random-layouts-v1.json');
 const types=request.task_types;
 if(!Array.isArray(types)||!types.length||new Set(types).size!==types.length||types.some(t=>!all.zones.some(z=>z.id===t)))throw Error('Select distinct supported task types: fryer, microwave, coffee, tray');
 const layout=data.layouts.find(l=>l.id===request.layout_id);if(!layout)throw Error('Unknown layout; no fallback');
 // Ordering the request differently must not move identical task selections.
 const pack={...all,zones:all.zones.filter(z=>types.includes(z.id))};
 const result=G.solveDiverse(layout,pack,request.seed);
 const parts=pack.zones.map(z=>{
  const placement=result.placements.find(p=>p.id===z.id)||null;
  return {part_id:z.id,name:z.name,placement_status:placement?'placed':'unplaced',placement,
   standard_frame:z.zone,physical_support_footprints:z.supports,edge_gap_range_m:z.gap,
   initialization_status:'not_instantiated',physics_status:'not_initialized',
   asset_source_root:'all_assets',asset_bindings_status:'pending_task_initial_state_adapter',
   instance_policy:'one station per selected task family; subtasks share this station',
   activation_policy:'bind actual generated objects to existing sector activation; entry must not resample or reset'};
 });
 return {schema:'arena.openended.workspace_plan.v1',
  status:parts.every(p=>p.placement)?'workspace_plan_ready':'workspace_plan_incomplete',
  request:{layout_id:layout.id,seed:request.seed,task_types:pack.zones.map(z=>z.id)},
  parts,requested_part_count:parts.length,placed_part_count:result.placements.length,
  runtime_registered:false,usd_written:false,assets_instantiated:false,physics_validated:false,
  scene_generation_allowed:false,
  pending:['resolve selected task initial states and all_assets bindings','instantiate episode-local scene parts','bind existing activation and runtime adapters','validate initial support and native physics'],
  selection_semantics:'Only selected families get workspace parts. All selected parts must be placed before scene generation; never silently omit a failed part.',
  source_hashes:Object.fromEntries(['random-layouts-v1.json','four-zone-pack-v1.json','random-placement-geometry.js'].map(f=>[f,hash(f)])),
  placement_evidence:result};
}
module.exports={build};
if(require.main===module){try{
 if(process.argv.length!==3)throw Error('Usage: node compose_workspaces.cjs request.json');
 console.log(JSON.stringify(build(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
}catch(e){console.error(e.message);process.exitCode=1;}}
