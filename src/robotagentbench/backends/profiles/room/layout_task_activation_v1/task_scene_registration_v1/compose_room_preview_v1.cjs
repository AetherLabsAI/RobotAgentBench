// Episode-level allocation and geometry plan, not a physical scene or video.
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {pathToFileURL}=require('node:url');
const {spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..'),web=path.join(root,'web');
const read=name=>JSON.parse(fs.readFileSync(path.join(web,name),'utf8'));
async function build(seed=42){
 const kitchen=require('./compose_workspaces.cjs').build({layout_id:1,seed,task_types:['fryer','microwave','coffee','tray']});
 if(kitchen.status!=='workspace_plan_ready')throw Error('Not all four kitchen stations fit');
 const allocations={book_red:'sofa',baseball_cap:'sofa',power_bank:'sofa',water_bottle:'sofa',notebook_green:'sofa',wallet:'small_table',glasses_case:'small_table',remote_tv:'small_table'};
 const boxes=read('sofa-object-boxes-v1.json'),policy=read('sofa-placement-policy-v1.json');
 policy.structural_obstacles=read(policy.structural_obstacles_file).obstacles;
 const selected=boxes.objects.filter(o=>allocations[o.id]==='sofa'||o.id==='paper_cup');
 const sofa=require(path.join(web,'sofa-placement-geometry.js')).solve(selected,policy,seed);
 if(sofa.missing.length)throw Error(`Sofa missing: ${sofa.missing}`);
 const {combination}=await import(pathToFileURL(path.join(web,'small-table-combinations.mjs')).href);
 const smallData=read('small-table-candidates-v1.json');
 const small=combination(smallData,seed);
 // Reuse the exact approved table rectangle computation before its DOM drawing.
 const sandbox={window:{}};vm.createContext(sandbox);
 for(const file of ['scene-data.js','coffee-table-items-v1.js'])vm.runInContext(fs.readFileSync(path.join(web,file),'utf8'),sandbox);
 const source=fs.readFileSync(path.join(web,'coffee-table-end-v1.js'),'utf8');
 const split=source.indexOf(" const ns='http://www.w3.org/2000/svg';");
 if(split<0)throw Error('Table geometry source changed; refuse guessed bounds');
 vm.runInContext(source.slice(0,split)+' window.roomTable={placement,band,cloth,start,end,left,right,u,v};})();',sandbox);
 const table=JSON.parse(JSON.stringify(sandbox.window.roomTable));
 const localObjects=require(path.join(web,'coffee-table-boxes-v1.js')).generate(seed,table.end-table.start,table.right-table.left);
 const world=p=>[table.u[0]*(table.start+p[0])+table.v[0]*(table.left+p[1]),table.u[1]*(table.start+p[0])+table.v[1]*(table.left+p[1])];
 // Preview singleton alternatives are not additive. Their one instance is on the small table.
 const trash=localObjects.filter(o=>!Object.hasOwn(allocations,o.assetId)).map(o=>({...o,world_center:world(o.center),world_corners:o.corners.map(world),world_protection:o.protection.map(world)}));
 const shelfResult=spawnSync('python3',['-B','-c','import json; from compose_shelf_inventory import compose, LABELS; print(json.dumps(compose({"seed":'+seed+',"required_categories":list(LABELS)})))'],{cwd:__dirname,encoding:'utf8'});
 if(shelfResult.status!==0)throw Error(shelfResult.stderr);
 const shelf=JSON.parse(shelfResult.stdout);
 const identities=[...sofa.placements.map(o=>o.id),...Object.keys(small).map(n=>n==='remote'?'remote_tv':n),...trash.map(o=>o.assetId)];
 for(const id of Object.keys(allocations))if(identities.filter(x=>x===id).length!==1)throw Error(`Singleton allocation failed: ${id}`);
 return {schema:'arena.room_preview_plan.v1',layout_id:1,seed,status:'geometry_and_inventory_plan_not_scene',
  control_hz:20,base_scene:sandbox.window.SCENE_DATA.scene,
  singleton_allocations:allocations,base_existing_instances_audited:false,
  kitchen,sofa:{...sofa,asset_definitions:selected},small_table:{placements:small,surface:smallData.surface},
  coffee_table:{...table,objects:trash,mode:'trash_present_large_tray_destination_blocked',wooden_tray_fixed:true,cloth_fixed:true},
  drawer:{count:0,allowed_count:[0,1],reason:'wallet and glasses allocated to small table; do not clone'},
  shelf,video_created:false,usd_created:false,physics_validated:false,
  blockers:['Missing all_assets accessories require source-approved episode-local packaging',
   'Audit base existing objects before resolving singleton identities',
   'Bind and validate actual assets, support heights, shelf placement and device initial states',
   'Run new 20 Hz native navigation in the initialized scene and render measured motion'],
  scope:'One room demonstration plan; not all layouts, no manipulation success claim'};
}
module.exports={build};
if(require.main===module)build().then(p=>{
 if(process.argv[2])fs.writeFileSync(process.argv[2],JSON.stringify(p,null,2));
 else console.log(JSON.stringify(p,null,2));
}).catch(e=>{console.error(e);process.exitCode=1;});
