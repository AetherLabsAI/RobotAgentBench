'use strict';
const assert=require('node:assert/strict');
const {build}=require('./compose_room_preview_v1.cjs');
(async()=>{
 const p=await build(42);
 assert.deepEqual(p,await build(42));
 assert.equal(p.kitchen.placed_part_count,4);
 assert.equal(p.sofa.missing.length,0);
 assert.equal(p.drawer.count,0);
 assert.equal(p.control_hz,20);
 assert.equal(p.shelf.required_categories.length,14);
 assert.equal(p.shelf.required_category_coverage_passed,true);
 assert.equal(p.video_created,false);
 assert.equal(p.physics_validated,false);
 assert.equal(p.base_existing_instances_audited,false);
 assert.ok(p.blockers.length);
 assert.ok(!p.coffee_table.objects.some(o=>['wallet','glasses_case','remote_tv'].includes(o.assetId)));
 assert.equal(p.coffee_table.mode,'trash_present_large_tray_destination_blocked');
 for(const layer of p.shelf.layers)assert.ok(layer.objects.length>=layer.required_categories.length);
 console.log(JSON.stringify({passed:true,seed:p.seed,kitchen_parts:p.kitchen.placed_part_count,
  sofa_objects:p.sofa.placements.length,small_table_objects:Object.keys(p.small_table.placements).length,
  table_trash_objects:p.coffee_table.objects.length,shelf_objects:p.shelf.layers.reduce((n,l)=>n+l.count,0),
  singleton_allocation:'exactly one per required ID in planned inventory; existing USD audit pending',
  scope:'CPU plan only; no physical scene or video'}));
})().catch(e=>{console.error(e);process.exitCode=1;});
