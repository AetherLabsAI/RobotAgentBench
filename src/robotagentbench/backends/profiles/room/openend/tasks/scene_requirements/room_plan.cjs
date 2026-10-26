'use strict';
const fs=require('node:fs'),P=require('./placement_constraints.cjs');
const G=require('./placement_sources/random-placement-geometry.js');
const pack=require('./station_pack.cjs').expand(require('./placement_sources/four-zone-pack-v1.json'),require('./station_geometry_v2.json'));
const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
// A microwave is required by the fridge handoff / microwave task, not by
// unrelated living-room tasks. Preserve explicit task ownership.
const request=JSON.parse(JSON.stringify(input.requirements.station_request));
const needsMicrowave=(input.requirements.fixed_task_requirements||[]).some(t=>t.type==='fridge_to_microwave') ||
 (input.requirements.meal_task_requirements||[]).some(t=>t.type==='microwave_box_to_meal_tray');
if(needsMicrowave)request.stations.push({id:'infrastructure_microwave',type:'microwave',placement:{}});
input.plan=P.build(request,input.layout,pack,G);
import('./living_plan.mjs').then(({build})=>{
 input.living_plan=build(input.requirements.portable_inventory);
 console.log(JSON.stringify(input));
 if(input.plan.status!=='planar_candidate')process.exitCode=2;
}).catch(e=>{console.error(e.stack);process.exitCode=1});
