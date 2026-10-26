const fs=require('node:fs'),path=require('node:path');
const {build}=require('./compose_workspaces.cjs');
const output={};
for(const id of [2,3,4]){
 const attempts=[];
 for(let seed=42+id;seed<62+id;seed++){
  const plan=build({layout_id:id,seed,task_types:['fryer','microwave','coffee','tray']});
  attempts.push({seed,status:plan.status});
  if(plan.status==='workspace_plan_ready'){output[id]={plan,attempts};break;}
 }
 if(!output[id])throw Error('No complete plan for layout '+id);
 console.log(id,output[id].plan.request.seed,output[id].plan.parts.map(p=>[p.part_id,p.placement.supportId,p.placement.origin]));
}
fs.writeFileSync(path.join(__dirname,'other_layouts_plans_v1.json'),JSON.stringify(output,null,2));
