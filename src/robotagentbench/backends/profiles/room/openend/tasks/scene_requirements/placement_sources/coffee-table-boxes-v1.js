(function(root){
 'use strict';
 const paperDefinitions=typeof module!=='undefined'&&module.exports?require('./paper-variants-v1.js'):root.PAPER_VARIANTS;
 if(!paperDefinitions)throw Error('纸团大小规格未加载');
 // CPU USD measurement, 2026-10-04: visible render/default mesh vertices.
 // XY bounds at authored orientation, rotated together with the preview yaw.
 const assets=[
  {id:'wallet',name:'钱包',size:[.16500000655651093,.12999999523162842,.03999999910593033],color:'#a75438'},
  {id:'glasses_case',name:'墨镜盒',size:[.060000003276652336,.17000000183097796,.03501075640325357],color:'#386ac5'},
  {id:'paper_ball',name:'纸团',size:[.058117708467195356,.0855660897200112,.15538805167546588],color:'#8661b7'},
  {id:'zbar_t',name:'空包装模板',size:[.13500000536441803,.03999999910593033,.13500000536441803],color:'#b47616'},
  {id:'empty_can',name:'空罐',size:[.06599999219179153,.06599999219179153,.12200000137090683],color:'#278175'},
  {id:'paper_cup',name:'空杯',size:[.08900000154972076,.12999999523162842,.08900000154972076],color:'#b13f75'}
 ].map(a=>({...a,source:`/home/lingjun/robot_agent_arena/api-protocol/all_assets/${a.id}/asset.usda`}));
 function random(seed){let s=seed>>>0;return()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
 function overlaps(a,b){
  for(const p of [a,b])for(let i=0;i<4;i++){
   const next=p[(i+1)%4],axis=[-(next[1]-p[i][1]),next[0]-p[i][0]];
   const aa=a.map(q=>q[0]*axis[0]+q[1]*axis[1]),bb=b.map(q=>q[0]*axis[0]+q[1]*axis[1]);
   if(Math.max(...aa)<=Math.min(...bb)+1e-12||Math.max(...bb)<=Math.min(...aa)+1e-12)return false;
  }
  return true;
 }
 const padding=.035;
 function boxCorners(center,size,yaw,pad=0){
  const c=Math.cos(yaw),s=Math.sin(yaw),w=size[0]/2+pad,h=size[1]/2+pad;
  return [[-w,-h],[w,-h],[w,h],[-w,h]].map(([x,y])=>[center[0]+c*x-s*y,center[1]+s*x+c*y]);
 }
 function pairClear(a,b){
  // Protection/protection overlap is allowed: do not turn 3.5 cm into 7 cm.
  return !overlaps(a.corners,b.protection)&&!overlaps(b.corners,a.protection);
 }
 function generate(seed,width,depth){
  if(!(width>0&&depth>0))throw Error('摆放区域无效');
  const rng=random(seed);
  // Separate deterministic stream: variant choices do not consume the existing
  // inventory/cup-pose stream. Pick once per instance, never to rescue a fit.
  const paperRng=random((seed>>>0)^0x70504150);
  // Sample inventory once, before placement retries: never silently drop objects.
  const instances=[];
  for(const asset of assets){
   const count=['wallet','glasses_case'].includes(asset.id)?1:1+Math.floor(rng()*3);
   for(let index=1;index<=count;index++){
    const upright=asset.id==='paper_cup'&&rng()<.5;
    const paperVariant=asset.id==='paper_ball'?(paperRng()<.5?'large':'small'):null;
    const paper=paperVariant?paperDefinitions.variants[paperVariant]:null;
    // The authored cup axis is +Y. Rotate +90° about X for upright +Z.
    const size=paper?[...paper.authored_pose_size_m]:upright?[asset.size[0],asset.size[2],asset.size[1]]:[...asset.size];
    instances.push({...asset,assetId:asset.id,id:`${asset.id}_${index}`,size,
     ...(paperVariant?{paperVariant,effectiveMeshAxisScale:[...paper.effective_mesh_axis_scale]}:{}),
     pose:asset.id==='paper_cup'?(upright?'upright':'lying'):'authored',
     rollRad:upright?Math.PI/2:0,
     name:`${asset.name}${count>1?index:''}${paperVariant?(paperVariant==='large'?'（大）':'（小）'):''}${asset.id==='paper_cup'?(upright?'（竖）':'（横）'):''}`});
   }
  }
  for(let restart=0;restart<30;restart++){
   const result=[];
   for(const asset of [...instances].sort((a,b)=>b.size[0]*b.size[1]-a.size[0]*a.size[1])){
    let found=false;
    for(let attempt=0;attempt<600;attempt++){
     const yaw=rng()*Math.PI*2,c=Math.cos(yaw),s=Math.sin(yaw),[w,h]=asset.size;
     const shape=[[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]].map(([x,y])=>[c*x-s*y,s*x+c*y]);
     const hx=Math.max(...shape.map(p=>Math.abs(p[0]))),hy=Math.max(...shape.map(p=>Math.abs(p[1])));
     if(2*hx>width||2*hy>depth)continue;
     const center=[hx+rng()*(width-2*hx),hy+rng()*(depth-2*hy)];
     const corners=shape.map(p=>[p[0]+center[0],p[1]+center[1]]);
     const protection=boxCorners(center,asset.size,yaw,padding);
     const candidate={...asset,center,yaw,corners,protection};
     if(result.some(item=>!pairClear(candidate,item)))continue;
     result.push(candidate);found=true;break;
    }
    if(!found)break;
   }
   if(result.length===instances.length)return result;
  }
  throw Error(`未找到 ${instances.length} 件满足 3.5 cm 保护规则的摆放，请换一组；没有减少数量、缩小资产或放宽规则。`);
 }
 const api={assets,generate,overlaps,padding,boxCorners,pairClear,paperVariants:paperDefinitions.variants};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 else root.CoffeeTableBoxes=api;
})(typeof window==='undefined'?globalThis:window);
