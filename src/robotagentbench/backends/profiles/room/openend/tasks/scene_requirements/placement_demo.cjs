'use strict';
const fs=require('node:fs'),path=require('node:path');
const P=require('./placement_constraints.cjs'),G=require('./placement_sources/random-placement-geometry.js');
const data=require('./placement_sources/random-layouts-v1.json'),pack=require('./placement_sources/four-zone-pack-v1.json');
const envelope=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const r=envelope.station_request,layout=data.layouts.find(l=>l.id===r.layout_id);
if(!layout)throw Error('Missing selected layout');
const group=P.build(r,layout,pack,G);if(group.status!=='planar_candidate')throw Error('Group example did not find complete plan');
const fixed={...r,stations:group.placements.map(p=>({id:p.task_id,type:p.id,
 placement:{surface_path:p.support,edge_index:p.surface_edge_index,local_xy_m:p.local_xy_m,front_yaw_deg:p.yaw_deg}}))};
const fixedPlan=P.build(fixed,layout,pack,G);if(fixedPlan.status!=='planar_candidate')throw Error('Exact round trip failed');
const range=JSON.parse(JSON.stringify(fixed));
const coffee=range.stations.find(s=>s.type==='coffee'),xy=coffee.placement.local_xy_m;
delete coffee.placement.local_xy_m;coffee.placement.local_bounds_m=[xy.map(x=>x-.02),xy.map(x=>x+.02)];
const rangePlan=P.build(range,layout,pack,G);if(rangePlan.status!=='planar_candidate')throw Error('Bounded example failed');
const rejected=JSON.parse(JSON.stringify(fixed));rejected.stations[0].placement.local_xy_m=[999,999];
const rejectedPlan=P.build(rejected,layout,pack,G);if(rejectedPlan.placements.length)throw Error('Invalid fixed target was accepted');
console.log(JSON.stringify({group,fixed_request:fixed,fixed_plan:fixedPlan,range_request:range,range_plan:rangePlan,
 rejected_request:rejected,rejected_plan:rejectedPlan,
 example_only:'Coordinates derived from layout01 legal planar candidate. Not new defaults or physical certification.'},null,2));
