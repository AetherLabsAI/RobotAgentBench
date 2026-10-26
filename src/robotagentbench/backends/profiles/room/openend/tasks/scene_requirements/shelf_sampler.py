"""Target- and robot-independent shelf inventories and physical placement."""
import copy
import hashlib
import json
import numpy as np
from snack_diversity import choose_without_early_repeats

SCENE_POLICY='scene_first_visible360_v1'
PACKAGES=frozenset(['chomps_r','oreo_gf','honeymaid','zbar_t','skit_littles'])
PACKAGE_NOMINAL_YAW_DEG=90.
PACKAGE_YAW_JITTER_DEG=30.


def draw_heading(rng, product):
    return float(rng.uniform(-180.,180.))


def scene_digest(objects):
    """Ignore assigned roles; preserve exact physical scene identity."""
    rows=[{k:o[k] for k in ['product_family','prim_path','support_id','xyz_m','wxyz']} for o in objects]
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def select_target(placed, product, rng):
    """Select an instance in a frozen scene; never edit its pose or neighbors."""
    indices=[i for i,(row,_,_) in enumerate(placed) if row['product_family']==product]
    if not indices:
        raise RuntimeError('Requested product absent from independently sampled scene')
    index=int(rng.choice(indices))
    return index,indices


def sample_scene(rng, catalog, supports, fixtures, layer_counts, kinds_by_layer,
                 pose_item, separated, record_round=None, focus=None,
                 fixed_target=None,candidate_allowed=None,planned_inventory=None,
                 priority_products=(),heading_ranges=None):
    from scipy.spatial.transform import Rotation
    from shapely.geometry import Polygon
    # No target, arm, wrist, robot or grasp parameters are accepted here.
    spacing={sid:(.085 if n<=5 else .055 if n<=9 else .035) for sid,n in layer_counts.items()}
    rejects={k:0 for k in ['support','object','fixture','target_visibility_or_corridor']}
    history=[]
    for packing_round in range(8):
        inventory=copy.deepcopy(planned_inventory) if planned_inventory is not None else {sid:choose_without_early_repeats(rng,kinds_by_layer[sid],count)
            for sid,count in layer_counts.items()}
        if set(inventory)!=set(layer_counts) or any(len(inventory[sid])!=count or not set(inventory[sid])<=set(kinds_by_layer[sid]) for sid,count in layer_counts.items()):
            raise ValueError('Invalid planned shelf inventory')
        if focus:
            sid=focus['support_id']
            if focus['product'] not in inventory[sid]:inventory[sid][-1]=focus['product']
        placed=[];instances={}
        chosen=[(name,sid) for sid,names in inventory.items() for name in names]
        # This packing order depends only on physical size, never a target role.
        rng.shuffle(chosen)
        chosen.sort(key=lambda pair:-np.prod(np.ptp(catalog[pair[0]]['vertices'],axis=0)[:2]))
        # Opt-in task policy: reserve targets before any background placement.
        # Default callers retain the original size-based ordering and RNG.
        if priority_products:chosen.sort(key=lambda pair:pair[0] not in priority_products)
        if focus:chosen.sort(key=lambda pair:pair[0]!=focus['product'])
        if fixed_target is not None:
            target_row,target_bodies,target_fp=fixed_target
            chosen.remove((target_row['product_family'],target_row['support_id']))
            placed=[(copy.deepcopy(target_row),target_bodies,target_fp)]
            instances[target_row['product_family']]=1
        for name,sid in chosen:
            sp=Polygon(supports[sid]['polygon_xy_m'])
            neighbors=[fp for row,_,fp in placed if row['support_id']==sid]
            placed_item=False
            # Draw orientation once for this item before XY support/packing.
            heading=draw_heading(rng,name)
            if heading_ranges and name in heading_ranges:
                heading=float(rng.uniform(*heading_ranges[name]))
            guided=bool(focus and name==focus['product'] and instances.get(name,0)==0)
            if guided:heading=float((focus['yaw_deg']+rng.uniform(-6.,6.)+180.)%360.-180.)
            if guided and focus.get('diverse'):heading=float(rng.uniform(-65.,55.))
            if guided and focus.get('edge'):heading=float((focus['yaw_deg']+rng.uniform(-30.,30.)+180.)%360.-180.)
            rot=Rotation.from_euler('z',heading,degrees=True).as_matrix()[:2,:2]
            outline=catalog[name]['footprint_xy']@rot.T
            low,high=outline.min(0),outline.max(0)
            for batch in range(12):
                candidates=[]
                for _ in range(64):
                    xy=rng.uniform(sp.bounds[:2],sp.bounds[2:])
                    if guided:xy=np.array(focus['xyz_m'][:2])+rng.uniform([-.025,-.018],[.025,.018])
                    if guided and focus.get('diverse'):
                        # Alternate left/right and two depth bands across replacement slots.
                        xlo,ylo,xhi,yhi=sp.bounds;middle=(xlo+xhi)/2
                        side=int(focus['diversity_index'])%2
                        xa,xb=(xlo+.10,middle-.04) if side==0 else (middle+.04,xhi-.10)
                        ya,yb=(ylo+.15,ylo+.20) if (int(focus['diversity_index'])//2)%2==0 else (ylo+.20,min(ylo+.26,yhi-.08))
                        xy=rng.uniform([xa,ya],[xb,yb])
                        if any(np.linalg.norm(xy-np.array(previous))<.045 for previous in focus.get('avoid_xy',[])):continue
                    if guided and focus.get('edge'):
                        xlo,ylo,xhi,yhi=sp.bounds
                        xa=max(xlo-low[0]+.012,focus['edge_x_range_m'][0]);xb=min(xhi-high[0]-.012,focus['edge_x_range_m'][1])
                        if xa>=xb:continue
                        # Search lateral extremes while retaining depth variation.
                        ya=max(ylo-low[1]+.012,focus['xyz_m'][1]-.06);yb=min(yhi-high[1]-.012,focus['xyz_m'][1]+.06)
                        if ya>=yb:continue
                        xy=rng.uniform([xa,ya],[xb,yb])
                        if any(np.linalg.norm(xy-np.array(previous))<focus.get('minimum_xy_separation_m',.025) for previous in focus.get('avoid_xy',[])):continue
                    fp=Polygon(outline+xy)
                    if not sp.covers(fp):rejects['support']+=1;continue
                    gap=min((fp.distance(old) for old in neighbors),default=1.)
                    if gap<spacing[sid]:rejects['object']+=1;continue
                    candidates.append((gap,xy))
                # Uniform proposal order, with minimum spacing only. No
                # largest-gap ranking that would favor edges or particular XY.
                rng.shuffle(candidates)
                for _,xy in candidates[:12]:
                    row,bodies,fp=pose_item(name,xy,heading,sid,'unassigned')
                    if any(row['support_id']==old['support_id'] and
                           any(not separated(a,b,.012) for a in bodies for b in old_bodies)
                           for old,old_bodies,_ in placed):
                        rejects['object']+=1;continue
                    if any(not separated(a,b,.00015) for a in bodies for b in fixtures
                           if b.name!=supports[sid]['prim_path']):
                        rejects['fixture']+=1;continue
                    if candidate_allowed is not None and not candidate_allowed((row,bodies,fp),placed):
                        rejects['target_visibility_or_corridor']+=1;continue
                    instance=instances.get(name,0);instances[name]=instance+1
                    if instance:
                        original=row['prim_path'];clone=original+f'_copy_{instance:02d}'
                        row['source_prim_path']=original;row['prim_path']=clone
                        row['collision_paths']=[p.replace(original,clone,1) for p in row['collision_paths']]
                        for body in bodies:body.name=body.name.replace(original,clone,1)
                    row['placement_zone']='rear' if (fp.bounds[1]+fp.bounds[3])/2>(sp.bounds[1]+sp.bounds[3])/2 else 'front'
                    row['orientation_sampling']=dict(reference='fixed shelf frame',nominal_yaw_deg=None,uniform_range_deg=[-180.,180.])
                    if guided:
                        row['targeted_grasp_hint']=dict(height_fraction=float(np.clip(focus['height_fraction']+rng.uniform(-.012,.012),.48,.76)),depth_mm=focus['depth_mm'])
                        row['orientation_sampling']=dict(reference='successful target neighborhood',source_episode_id=focus['source_episode_id'],center_yaw_deg=focus['yaw_deg'],jitter_deg=6.)
                        if focus.get('diverse'):row['orientation_sampling']=dict(reference='diverse BBQ quota',uniform_range_deg=[-65.,55.])
                        if focus.get('diverse'):row['bbq_diverse']=True
                        if focus.get('edge'):
                            row['coverage_edge']=focus['edge_side']
                            if name=='pring_bbq':row['bbq_diverse']=True
                            row['orientation_sampling']=dict(reference='same-side accepted reference',source_episode_id=focus['source_episode_id'],center_yaw_deg=focus['yaw_deg'],jitter_deg=30.)
                    placed.append((row,bodies,fp));placed_item=True;break
                if placed_item:break
            if not placed_item:break
        complete=len(placed)==sum(layer_counts.values())
        receipt=dict(round=packing_round,inventory=inventory,placed=len(placed),complete=complete)
        history.append(receipt)
        if record_round:record_round(receipt)
        if complete:
            metrics={}
            for sid in layer_counts:
                footprints=[fp for row,_,fp in placed if row['support_id']==sid]
                nearest=[min((fp.distance(other) for j,other in enumerate(footprints) if i!=j),default=float('inf'))
                    for i,fp in enumerate(footprints)]
                metrics[sid]=dict(objects=len(footprints),required_edge_gap_m=spacing[sid],
                    minimum_edge_gap_m=min(nearest) if len(nearest)>1 else None,median_nearest_edge_gap_m=float(np.median(nearest)) if len(nearest)>1 else None)
                assert min(nearest)>=spacing[sid]-1e-8
            return placed,dict(policy=SCENE_POLICY,packing_round=packing_round,rounds=history,
                layer_counts=layer_counts,spacing_metrics=metrics,rejections=rejects,
                package_yaw_deg=[-180.,180.],container_yaw_deg=[-180.,180.],
                uses_target=bool(focus),uses_robot=fixed_target is not None,reserves_grasp_corridor=fixed_target is not None,
                targeted_sampling=focus)
    raise RuntimeError(('Independent shelf layout sampling exhausted',layer_counts,history,rejects))
