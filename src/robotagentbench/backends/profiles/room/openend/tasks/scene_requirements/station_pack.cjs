'use strict';
// Keep approved gaps/activity envelopes. Only fix the physical member inventory.
function expand(pack,geometry){
 const result=JSON.parse(JSON.stringify(pack));
 for(const zone of result.zones){
  const members=geometry.assemblies[zone.id];if(!members)continue;
  if(!members.length)throw Error('Missing station geometry '+zone.id);
  zone.supports=members.map(x=>({min:x.min,max:x.max}));
 }
 return result;
}
module.exports={expand};
