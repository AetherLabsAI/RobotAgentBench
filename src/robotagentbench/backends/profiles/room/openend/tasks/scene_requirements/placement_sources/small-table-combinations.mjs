export const kinds=['glasses_case','remote','wallet'];
export function overlaps(a,b){
  for(const poly of [a,b])for(let i=0;i<poly.length;i++){
    const p=poly[i],q=poly[(i+1)%poly.length],axis=[p[1]-q[1],q[0]-p[0]];
    const pa=a.map(v=>v[0]*axis[0]+v[1]*axis[1]),pb=b.map(v=>v[0]*axis[0]+v[1]*axis[1]);
    if(Math.max(...pa)<=Math.min(...pb)+1e-12||Math.max(...pb)<=Math.min(...pa)+1e-12)return false;
  }
  return true;
}
export function combination(data,seed,edge=false){
  let state=(seed>>>0)^0xa3c59ac3;
  const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
  const pools=kinds.map(kind=>{
    const rows=data.objects[kind].candidates.filter(r=>!edge||r.overhang_fraction>=.2).slice();
    for(let i=rows.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[rows[i],rows[j]]=[rows[j],rows[i]];}
    return rows;
  });
  const selected=[];
  function search(depth){
    if(depth===kinds.length)return true;
    for(const row of pools[depth]){
      if(selected.some(other=>overlaps(row.footprint_xy_m,other.footprint_xy_m)))continue;
      selected.push(row);if(search(depth+1))return true;selected.pop();
    }
    return false;
  }
  if(!search(0))throw Error('当前候选中没有符合条件的三件互不重叠组合。');
  return Object.fromEntries(kinds.map((kind,i)=>[kind,selected[i]]));
}
