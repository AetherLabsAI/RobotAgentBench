const fs=require('node:fs'),path=require('node:path');
const G=require('../web/random-placement-geometry.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../web/random-layouts-v1.json')));
const pack=JSON.parse(fs.readFileSync(path.join(__dirname,'../web/four-zone-pack-v1.json')));
const output={},failures={};
for(const id of [2,3,4]){
 const attempts=[],layout=data.layouts.find(l=>l.id===id);
 for(let seed=42+id;seed<102+id;seed++){
  const r=G.solveDiverse(layout,pack,seed,[],{attempts:1000,nodes:200,variants:2});
  attempts.push({seed,status:r.status,missing:r.missing});
  console.log(id,seed,r.status,r.missing);
  if(r.status!=='preview_candidate')continue;
  const plan={request:{layout_id:id,seed},parts:pack.zones.map(z=>({part_id:z.id,placement:r.placements.find(p=>p.id===z.id)})),placement_evidence:r};
  output[id]={plan,attempts};break;
 }
 if(!output[id])failures[id]={status:'bounded_search_incomplete_no_scene_generated',attempts};
 fs.writeFileSync(path.join(__dirname,'other_layouts_failures_v1.json'),JSON.stringify(failures,null,2));
 fs.writeFileSync(path.join(__dirname,'other_layouts_plans_v1.json'),JSON.stringify(output,null,2));
}
