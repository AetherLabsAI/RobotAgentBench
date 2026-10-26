'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const P=require('./placement_constraints.cjs'),G=require('./placement_sources/random-placement-geometry.js');
const pack=require('./station_pack.cjs').expand(require('./placement_sources/four-zone-pack-v1.json'),require('./station_geometry_v2.json'));
const data=require('./placement_sources/random-layouts-v1.json');
test('every actual coffee member participates in packing',()=>{
 assert.equal(pack.zones.find(z=>z.id==='coffee').supports.length,3);
 assert.equal(pack.zones.find(z=>z.id==='fryer').supports.length,4);
});
test('layout26 seed42 jointly places one microwave with coffee and fryer',()=>{
 const layout=data.layouts.find(l=>l.id===26);
 const request={layout_id:26,seed:42,stations:['coffee','fryer','microwave'].map(type=>({id:type,type,placement:{}}))};
 const plan=P.build(request,layout,pack,G);
 assert.equal(plan.status,'planar_candidate');assert.equal(plan.placements.length,3);
 for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)assert(G.compatible(plan.placements[i],plan.placements[j]));
 const moved=P.build({...request,seed:43},layout,pack,G);
 assert.equal(moved.status,'planar_candidate');
 assert.notDeepEqual(plan.placements.find(p=>p.id==='microwave').origin,moved.placements.find(p=>p.id==='microwave').origin);
 assert.notDeepEqual(plan.placements.find(p=>p.id==='fryer').origin,moved.placements.find(p=>p.id==='fryer').origin);
 for(const p of [...plan.placements,...moved.placements]){
  const zone=pack.zones.find(z=>z.id===p.id);
  assert.deepEqual(p.constraint_receipt.approved_edge_gap_m,zone.gap);
  assert.equal(G.validate(p,layout),null);
 }
});
test('exactly overlapping coffee/microwave cannot be accepted',()=>{
 const layout=data.layouts.find(l=>l.id===26);
 const station={surface_path:'/World/Background/Shell/ConstructorKitchen/MetricScale/BP_Kitchen9/Body/Worktop',world_xy_m:[1.877040263406847,6.667875775708991],front_yaw_deg:-90.00000216524357};
 const request={layout_id:26,seed:42,stations:['coffee','microwave'].map(type=>({id:type,type,placement:station}))};
 assert.notEqual(P.build(request,layout,pack,G).status,'planar_candidate');
});
