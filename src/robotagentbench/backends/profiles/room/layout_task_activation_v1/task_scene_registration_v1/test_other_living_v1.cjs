const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const plans=require('./other_living_plans_v1.json');
const boxes=require('../web/coffee-table-boxes-v1.js');
async function main(){
 const {overlaps}=await import('../web/small-table-combinations.mjs');
 for(const p of plans){
  assert.equal(p.sofa.missing.length,0);assert.equal(p.sofa.placements.length,6);
  const small=Object.values(p.small_table.placements);assert.equal(small.length,3);
  for(let i=0;i<small.length;i++){
   assert(small[i].overhang_fraction<=.5+1e-7);
   for(let j=i+1;j<small.length;j++)assert(!overlaps(small[i].footprint_xy_m,small[j].footprint_xy_m));
  }
  const trash=p.coffee_table.objects;
  for(let i=0;i<trash.length;i++)for(let j=i+1;j<trash.length;j++)assert(boxes.pairClear(trash[i],trash[j]));
  assert(!trash.some(x=>Object.hasOwn(p.singleton_allocations,x.assetId)));
  assert(p.coffee_table.cloth_fixed&&p.coffee_table.wooden_tray_fixed);
 }
 assert.notDeepEqual(plans[0].sofa.placements,plans[1].sofa.placements);
 assert.notDeepEqual(plans[0].small_table.placements,plans[1].small_table.placements);
 assert.notDeepEqual(plans[0].coffee_table.objects,plans[1].coffee_table.objects);
 console.log('PASS: distinct layout seeds; sofa complete; small-table non-overlap / overhang <= 50%; trash 3.5cm clearance; no duplicate singletons; fixed cloth/tray');
}
main().catch(e=>{console.error(e);process.exitCode=1});
