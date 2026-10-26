(function(root){
 'use strict';
 const eps=1e-7,cross=(a,b)=>a[0]*b[1]-a[1]*b[0],sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
 const area=p=>Math.abs(p.reduce((s,a,i)=>s+cross(a,p[(i+1)%p.length]),0))/2;
 const rect=b=>[[b.min[0],b.min[1]],[b.max[0],b.min[1]],[b.max[0],b.max[1]],[b.min[0],b.max[1]]];
 function clip(p,a,b,inside=true){
  const side=x=>cross(sub(b,a),sub(x,a))*(inside?1:-1),out=[];
  if(!p.length)return out;
  // Complementary half-planes must not both include a tolerance strip.
  // Tiny mesh edges otherwise duplicate whole residual polygons exponentially.
  // Coverage tolerance remains the existing final residual-area test below.
  for(let i=0;i<p.length;i++){const x=p[i],y=p[(i+1)%p.length],dx=side(x),dy=side(y),ix=dx>=0,iy=dy>=0;
   if(ix)out.push(x);if(ix!==iy){const t=dx/(dx-dy);out.push([x[0]+(y[0]-x[0])*t,x[1]+(y[1]-x[1])*t]);}}
  return out;
 }
 function subtract(p,q){let rest=p,out=[];for(let i=0;i<q.length&&rest.length;i++){
  const a=q[i],b=q[(i+1)%q.length];if(a[0]===b[0]&&a[1]===b[1])continue;
  const outside=clip(rest,a,b,false);
  if(area(outside)>1e-9)out.push(outside);rest=clip(rest,a,b,true);
 }return out;}
 function covered(p,surfaces){let rest=[p];for(const s of surfaces)for(const polygon of (s.support_patches||[s.polygon])){rest=rest.flatMap(r=>subtract(r,polygon));if(!rest.length)return true;}
  return rest.reduce((s,p)=>s+area(p),0)<1e-7;}
 function overlap(a,b){
  for(const p of [a,b])for(let i=0;i<p.length;i++){
   const d=sub(p[(i+1)%p.length],p[i]),axis=[-d[1],d[0]];if(Math.hypot(...axis)<eps)continue;
   const pa=a.map(v=>v[0]*axis[0]+v[1]*axis[1]),pb=b.map(v=>v[0]*axis[0]+v[1]*axis[1]);
   if(Math.min(Math.max(...pa),Math.max(...pb))-Math.max(Math.min(...pa),Math.min(...pb))<=eps)return false;
  }return true;
 }
 function inside(p,poly){let yes=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){
  const a=poly[i],b=poly[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;
 }return yes;}
 // Ear clipping preserves concave room boundaries. Checking only four corners
 // misses frames bridging a wall notch; a convex hull would fill that notch.
 function triangulate(poly){
  const p=poly.filter((v,i)=>Math.hypot(...sub(v,poly[(i+poly.length-1)%poly.length]))>eps).map(v=>[...v]);
  if(p.reduce((s,v,i)=>s+cross(v,p[(i+1)%p.length]),0)<0)p.reverse();
  const out=[];let guard=p.length*p.length;
  while(p.length>3&&guard-->0){let found=false;
   for(let i=0;i<p.length;i++){const a=p[(i+p.length-1)%p.length],b=p[i],c=p[(i+1)%p.length];
    const turn=cross(sub(b,a),sub(c,b));if(Math.abs(turn)<1e-10){p.splice(i,1);found=true;break;}if(turn<0)continue;
    const contains=q=>cross(sub(b,a),sub(q,a))>=-eps&&cross(sub(c,b),sub(q,b))>=-eps&&cross(sub(a,c),sub(q,c))>=-eps;
    if(p.some((q,j)=>j!==i&&j!==(i+p.length-1)%p.length&&j!==(i+1)%p.length&&contains(q)))continue;
    out.push([a,b,c]);p.splice(i,1);found=true;break;
   }if(!found)return [];
  }if(p.length===3)out.push(p);return out;
 }
 const floorCache=new WeakMap();
 function withinFloor(p,floors){
  if(!floorCache.has(floors))floorCache.set(floors,floors.flatMap(triangulate).map(polygon=>({polygon})));
  return covered(p,floorCache.get(floors));
 }
 const footprint=(b,o,n)=>rect(b).map(p=>[o[0]+n[1]*p[0]+n[0]*p[1],o[1]-n[0]*p[0]+n[1]*p[1]]);
 function rng(seed){let s=(Number(seed)>>>0)||1;return()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
 function geometry(z,s,a,b,t,gap){const edge=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t],d=sub(b,a),len=Math.hypot(...d),n=[d[1]/len,-d[0]/len];
  const o=[edge[0]-n[0]*(gap+z.body.max[1]),edge[1]-n[1]*(gap+z.body.max[1])];
  return {id:z.id,name:z.name,color:z.color,origin:o,front:n,edge_point:edge,gap,support:s.path,group:s.group,top:s.top,
   polygon:footprint(z.zone,o,n),body:footprint(z.body,o,n),objects:z.supports.map(x=>footprint(x,o,n)),
   zmin:s.top+z.zone.min[2],zmax:s.top+z.zone.max[2],object_zranges:z.supports.map(b=>[s.top+b.min[2],s.top+b.max[2]]),object_zmax:s.top+Math.max(...z.supports.map(b=>b.max[2])),yaw_deg:Math.atan2(n[1],n[0])*180/Math.PI};
 }
 // Planar-design policy: elevated fixtures (e.g. upper cabinets) are omitted,
 // not certified clear. Floor-standing walls/furniture and sink holes remain.
 function planarBlockers(layout,top){return layout.blockers.filter(b=>!b.support_surface&&(b.always_exclude_xy||(b.min[2]<=top+.005&&b.max[2]>.02)));}
 function supportSurfaces(p,layout){
  const anchor=layout.surfaces.find(s=>s.path===p.support||(s.collider_paths||[]).includes(p.support));
  if(!anchor)return [];
  return layout.surfaces.filter(s=>anchor.support_set_id?s.support_set_id===anchor.support_set_id:(s.group===p.group&&Math.abs(s.top-p.top)<.002));
 }
 function checkFootprints(p,layout){
  if(!withinFloor(p.polygon,layout.floors)||!p.objects.every(obj=>withinFloor(obj,layout.floors)))return 'room_boundary';
  const near=supportSurfaces(p,layout);
  if(!p.objects.every(obj=>covered(obj,near)))return 'support';
  const blockers=planarBlockers(layout,p.top);
  for(const b of blockers){
   if(p.objects.some(obj=>overlap(obj,rect(b))))return 'physical_obstacle';
  }
  for(const b of layout.blockers.filter(b=>b.support_surface))for(let i=0;i<p.objects.length;i++){
   const z=p.object_zranges?.[i];
   if(!z)return 'missing_object_height';
   if(Math.min(z[1],b.max[2])-Math.max(z[0],b.min[2])>eps&&overlap(p.objects[i],rect(b)))return 'raised_support_penetration';
  }
  return null;
 }
 function validate(p,layout){
  const reason=checkFootprints(p,layout);if(reason)return reason;
  const near=supportSurfaces(p,layout),blockers=planarBlockers(layout,p.top);
  // The entire operated asset front must face an exposed edge, not a tile seam.
  const external=[p.edge_point[0]+p.front[0]*.03,p.edge_point[1]+p.front[1]*.03];
  if(near.some(s=>inside(external,s.polygon)))return 'internal_edge';
  const width=Math.max(...p.body.map(v=>v[0]*p.front[1]-v[1]*p.front[0]))-Math.min(...p.body.map(v=>v[0]*p.front[1]-v[1]*p.front[0]));
  const approach=footprint({min:[-width/2,.03],max:[width/2,.4]},p.edge_point,p.front);
  // This is a front-side clearance strip, not a certified robot footprint.
  if(!withinFloor(approach,layout.floors)||layout.surfaces.some(s=>overlap(approach,s.polygon)))return 'operator_side';
  if(blockers.some(b=>overlap(approach,rect(b))))return 'operator_obstacle';
  for(const b of blockers){
   if((!b.activity_projection_allowed&&overlap(p.polygon,rect(b)))||p.objects.some(obj=>overlap(obj,rect(b))))return 'fixed_obstacle';
  }return null;
 }
 function compatible(a,b){
  // Activity frames may overlap. Static task objects must not be inside
  // another task's activity frame, nor intersect each other's footprints.
  return !a.objects.some(x=>b.objects.some(y=>overlap(x,y))||overlap(x,b.polygon))&&!b.objects.some(x=>overlap(x,a.polygon));
 }
 function physicalSeparate(a,b){return !a.objects.some(x=>b.objects.some(y=>overlap(x,y)));}
 function changeMetrics(placements,previous){
  let groups=0,turns=0,moved=0,distance=0;
  for(const p of placements){const q=previous.find(q=>q.id===p.id);if(!q)continue;
   const d=Math.hypot(...sub(p.origin,q.origin)),dot=p.front[0]*q.front[0]+p.front[1]*q.front[1];
   groups+=Number(p.group!==q.group);turns+=Number(dot<.7071);moved+=Number(d>=.15);distance+=Math.min(d,2);
  }return [groups,turns,moved,distance];
 }
 const compare=(a,b)=>{for(let i=0;i<a.length;i++){if(Math.abs(a[i]-b[i])>1e-9)return a[i]-b[i];}return 0;};
 function novelty(placements,history){return history.length?history.map(h=>changeMetrics(placements,h.placements)).sort(compare)[0]:[0,0,0,0];}
 function solve(layout,pack,seed=1,options={}){
  if(layout.errors.length)return {status:'geometry_unavailable',placements:[],errors:layout.errors,seed,layout_id:layout.id,height_clearance_checked:false};
  const random=rng(seed),counts={},rejections={},attempts=options.attempts??1600;
  const edges=layout.surfaces.flatMap(s=>s.polygon.map((a,i)=>({s,a,b:s.polygon[(i+1)%s.polygon.length]}))).filter(e=>Math.hypot(...sub(e.b,e.a))>.06);
  const lists=pack.zones.map(z=>{const good=[],reject={};for(let i=0;i<attempts&&edges.length;i++){
   // Diversity mode samples support groups equally; numerous stove tiles must
   // not outweigh the island merely because they contain more mesh edges.
   const groups=[...new Set(edges.map(e=>e.s.group))],group=options.diversity?groups[Math.floor(random()*groups.length)]:null;
   const pool=group?edges.filter(e=>e.s.group===group):edges;
   const e=pool[Math.floor(random()*pool.length)],p=geometry(z,e.s,e.a,e.b,random(),z.gap[0]+random()*(z.gap[1]-z.gap[0]));
   const reason=validate(p,layout);if(reason)reject[reason]=(reject[reason]||0)+1;else good.push(p);
  }if(options.history?.length){const ranked=good.map(p=>({p,score:novelty([p],options.history),tie:random()}));
   ranked.sort((a,b)=>compare(b.score,a.score)||a.tie-b.tie);good.splice(0,good.length,...ranked.map(x=>x.p));}
  counts[z.id]=good.length;rejections[z.id]=reject;return {id:z.id,good};}).sort((a,b)=>a.good.length-b.good.length);
  let best=[],nodes=0,limited=false;const cap=options.nodes??30000;
  function search(i,chosen){if(chosen.length>best.length)best=[...chosen];if(best.length===pack.zones.length)return true;
   if(++nodes>cap){limited=true;return false;}if(i===lists.length||chosen.length+lists.length-i<=best.length)return false;
   for(const p of lists[i].good){if(chosen.every(q=>compatible(p,q))&&search(i+1,[...chosen,p]))return true;if(limited)return false;}
   return search(i+1,chosen);
  }search(0,[]);
  const strictCount=best.length;
  const warnings=Object.fromEntries(best.map(p=>[p.id,[]]));
  for(const p of best){const reason=validate(p,layout);if(reason)warnings[p.id].push(reason);}
  for(let i=0;i<best.length;i++)for(let j=i+1;j<best.length;j++)if(!compatible(best[i],best[j])){
   warnings[best[i].id].push('other_task_object');warnings[best[j].id].push('other_task_object');
  }
  const conflicts=Object.values(warnings).some(x=>x.length);
  const pairs=[];for(let i=0;i<best.length;i++)for(let j=i+1;j<best.length;j++)if(overlap(best[i].polygon,best[j].polygon))pairs.push([best[i].id,best[j].id]);
  return {schema:'arena.random_four_zone.preview.v1',layout_id:layout.id,scene_sha256:layout.scene_sha256,seed:Number(seed)>>>0,
   status:best.length===pack.zones.length&&!conflicts?'preview_candidate':(best.length===pack.zones.length?'preview_conflicts':'search_incomplete'),placements:best,placement_warnings:warnings,strict_search_placed_count:strictCount,missing:pack.zones.filter(z=>!best.some(p=>p.id===z.id)).map(z=>z.id),
   candidate_counts:counts,rejections,search_nodes:nodes,search_limit_reached:limited,overlapping_zone_pairs:pairs,zone_overlap_allowed:true,
   physics_validated:false,simultaneous_execution_certified:false,original_scene_modified:false,
   validation_mode:'planar_no_overhead_clearance',height_clearance_checked:false,sink_activity_projection_allowed:true,
   caveat:'Bounded search; failure is not proof of infeasibility. Visual support / obstacle design check only.'};
 }
 function solveDiverse(layout,pack,seed=1,history=[],options={}){
  // Keep a previously valid plan as a fallback: novelty must never be bought
  // by dropping an otherwise placeable task or weakening any geometry rule.
  const recent=history.filter(h=>h.layout_id===layout.id&&h.scene_sha256===layout.scene_sha256).slice(-6);
  const reusable=recent.filter(h=>h.placements.every(p=>!validate(p,layout))&&h.placements.every((p,i)=>h.placements.slice(i+1).every(q=>compatible(p,q))));
  const score=r=>[r.placements.length,...novelty(r.placements,recent)];
  let best=null,selectedSeed=null;const seeds=[];
  for(let i=0;i<(options.variants??4);i++){
   const candidateSeed=(Number(seed)+Math.imul(i,2654435761))>>>0;seeds.push(candidateSeed);
   const r=solve(layout,pack,candidateSeed,{...options,diversity:i>0,history:i>0?recent:[],nodes:options.nodes??(i===0?30000:10000)});
   if(!best||compare(score(r),score(best))>0){best=r;selectedSeed=candidateSeed;}
  }
  let reused=false;
  for(const r of reusable)if(compare(score(r),score(best))>0){
   const placements=r.placements,pairs=[];
   for(let i=0;i<placements.length;i++)for(let j=i+1;j<placements.length;j++)if(overlap(placements[i].polygon,placements[j].polygon))pairs.push([placements[i].id,placements[j].id]);
   best={...solve(layout,pack,seed,{attempts:0}),placements,
    status:placements.length===pack.zones.length?'preview_candidate':'search_incomplete',
    strict_search_placed_count:placements.length,placement_warnings:Object.fromEntries(placements.map(p=>[p.id,[]])),
    missing:pack.zones.filter(z=>!placements.some(p=>p.id===z.id)).map(z=>z.id),overlapping_zone_pairs:pairs};
   selectedSeed=null;reused=true;
  }
  const last=recent.at(-1),change=last?changeMetrics(best.placements,last.placements):null;
  return {...best,seed:Number(seed)>>>0,diversity:{history_size:recent.length,trial_seeds:seeds,selected_candidate_seed:selectedSeed,
   reused_previous:reused,changes_from_previous:change?{support_groups:change[0],orientations:change[1],moved_at_least_15cm:change[2],capped_total_distance_m:change[3]}:null,
   score:score(best),priority:'placed_count, then support_group_changes, orientation_changes, moves_15cm, distance; maximin over last 6 plans',
   search_is_bounded:true},reproduction_history:recent.map(h=>({layout_id:h.layout_id,scene_sha256:h.scene_sha256,placements:h.placements}))};
 }
 const api={area,rect,subtract,covered,overlap,inside,triangulate,withinFloor,footprint,geometry,planarBlockers,supportSurfaces,checkFootprints,validate,compatible,physicalSeparate,changeMetrics,solve,solveDiverse};
 if(typeof module!=='undefined')module.exports=api;else root.RandomPlacement=api;
})(typeof self==='undefined'?{}:self);
