(function(root){
  'use strict';
  const EPS=1e-9;
  function random(seed){let n=seed>>>0;return()=>{n=(n+0x6D2B79F5)>>>0;let t=n;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
  function region(policy){
    const width=policy.reference_approximate_width_m,depth=policy.reference_depth_m;
    if(!(width>0&&depth>0&&policy.edge_margin_m>=0&&policy.ordinary_object_padding_xy_m>=0))throw Error('无效摆放规则');
    if(policy.require_structural_obstacles&&(!Array.isArray(policy.structural_obstacles)||!policy.structural_obstacles.length))throw Error('缺少真实靠背／扶手边界，拒绝生成');
    const obstacles=policy.structural_obstacles||[];
    if(obstacles.some(o=>!Array.isArray(o.bounds)||o.bounds.length!==4||!o.bounds.every(Number.isFinite)||o.bounds[0]>=o.bounds[2]||o.bounds[1]>=o.bounds[3]))throw Error('结构边界无效');
    return {width,depth,seatWidth:width/3,totalWidth:width/3*4,margin:policy.edge_margin_m,padding:policy.ordinary_object_padding_xy_m,obstacles};
  }
  function bounds(p,padding=0){return [p.x-p.width/2-padding,p.y-p.depth/2-padding,p.x+p.width/2+padding,p.y+p.depth/2+padding];}
  function separated(a,b,gap){const x=bounds(a),y=bounds(b);return x[2]+gap<=y[0]+EPS||y[2]+gap<=x[0]+EPS||x[3]+gap<=y[1]+EPS||y[3]+gap<=x[1]+EPS;}
  function inside(p,r){const b=bounds(p);return b[0]>=-r.width/2+r.margin-EPS&&b[2]<=r.width/2-r.margin+EPS&&b[1]>=r.margin-EPS&&b[3]<=r.depth-r.margin+EPS;}
  function clearStructures(p,r){const a=bounds(p,r.padding);return r.obstacles.every(o=>{const b=o.bounds;return a[2]<=b[0]+EPS||a[0]>=b[2]-EPS||a[3]<=b[1]+EPS||a[1]>=b[3]-EPS;});}
  function solve(objects,policy,seed){
    if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw Error('种子应为 0 至 4294967295 的整数');
    if(new Set(objects.map(o=>o.id)).size!==objects.length)throw Error('物品重复');
    const r=region(policy),rng=random(seed);let best=[];
    for(let attempt=0;attempt<100;attempt++){
      const order=objects.map(o=>({o,rank:rng()})).sort((a,b)=>attempt%3?a.rank-b.rank:(b.o.body_size_m[0]*b.o.body_size_m[1]-a.o.body_size_m[0]*a.o.body_size_m[1]));
      const placed=[];
      for(const {o} of order){
        const [w,d]=o.body_size_m;if(!(w>0&&d>0))throw Error('物品尺寸无效');
        for(let trial=0;trial<300;trial++){
          const yaw=Math.floor(rng()*4)*90,swap=yaw%180!==0,width=swap?d:w,depth=swap?w:d;
          const spanX=r.width-2*r.margin-width,spanY=r.depth-2*r.margin-depth;
          if(spanX<0||spanY<0)continue;
          const p={id:o.id,name:o.name,x:-spanX/2+rng()*spanX,y:r.margin+depth/2+rng()*spanY,width,depth,yaw_deg:yaw};
          if(inside(p,r)&&clearStructures(p,r)&&placed.every(q=>separated(p,q,r.padding))){placed.push(p);break;}
        }
      }
      if(placed.length>best.length)best=placed;
      if(best.length===objects.length)break;
    }
    best.sort((a,b)=>objects.findIndex(o=>o.id===a.id)-objects.findIndex(o=>o.id===b.id));
    if(!best.every(p=>inside(p,r)&&clearStructures(p,r)&&best.every(q=>p===q||separated(p,q,r.padding))))throw Error('摆放校验失败');
    return {schema:'arena.sofa_planar_preview.v2',seed,region:r,placements:best,missing:objects.filter(o=>!best.some(p=>p.id===o.id)).map(o=>o.id),
      coordinate_frame:'reference sofa: X lateral, Y front to rear, origin at usable front edge center; not scene world coordinates',
      pose:'asset default attitude plus quarter-turn yaw; no automatic rest-pose change',
      checks:{region_edge:true,ordinary_object_gap:true,structural_obstacles:r.obstacles.length>0,arm_sweep:false,cushions:false,seat_seams:false,physics:false},
      scene_generation_connected:false};
  }
  const api={region,bounds,separated,inside,clearStructures,solve};if(typeof module!=='undefined')module.exports=api;else root.SofaPlacement=api;
})(typeof window==='undefined'?{}:window);
