"""Task-owned, read-only navigation progress. No physics or action policy."""
from copy import deepcopy
import math

VERSION='short_tasks.navigation_progress.v1'
SPACING_M=.2
POINTS=.2
SUPPORTED=frozenset(('stock_wooden_tray','retrieve_from_drawer','sofa_to_drawer',
    'hang_cap','fridge_to_heated_box','pack_object','carry_meal_tray',
    'sofa_to_coffee_table','table_to_storage_basket'))

def attach(plan):
    if plan.get('available') and plan['task_name'] in SUPPORTED:
        plan['navigation_policy']=dict(version=VERSION,spacing_m=SPACING_M,points=POINTS,
            operation_timeout_s=60.,navigation_timeout_s=120.,
            distance='base_xy_to_fixed_destination_workspace_boundary',
            historical_best_only=True,stage_switch_resets_clock=False,
            maximum_mode='base_maximum_plus_route_budgets_bound_at_activation',
            success_unchanged=True)
        plan['no_progress_timeout_by_phase_s']={'operation':60.,'navigation':120.}
    return plan

def distance_to_polygon(point, polygon):
    x,y=map(float,point);p=tuple(tuple(map(float,v)) for v in polygon)
    if len(p)<3 or not all(math.isfinite(v) for row in ((x,y),)+p for v in row):
        raise ValueError('Finite fixed workspace polygon required')
    inside=False;distance=math.inf
    for a,b in zip(p,p[1:]+p[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
        u=0. if den==0 else max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/den))
        distance=min(distance,math.hypot(x-a[0]-u*dx,y-a[1]-u*dy))
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return 0. if inside else distance

class Progress:
    def __init__(self, regions, radius):
        if not regions or not math.isfinite(radius) or radius<=0:raise ValueError('Bound workspaces required')
        self.regions=deepcopy(regions);self.radius=float(radius)
        self.routes={};self.events=[];self.last_time=None;self.active=[]
    def observe(self,time_s,xy,eligible,*,baseline=False):
        if (not math.isfinite(time_s) or time_s<0 or
                self.last_time is not None and time_s<=self.last_time):raise ValueError('Fresh navigation clock required')
        if set(eligible)!=set(self.regions) or any(type(v) is not bool for v in eligible.values()):
            raise ValueError('Explicit stage eligibility for every route required')
        self.last_time=float(time_s);self.active=[]
        for key,region in self.regions.items():
            distance=max(0.,distance_to_polygon(xy,region['polygon'])-self.radius)
            row=self.routes.get(key)
            if row is None:
                if baseline or not eligible[key]:continue
                row=dict(start_distance_m=distance,best_distance_m=distance,current_distance_m=distance,
                    budget_steps=int(math.floor((distance+1e-9)/SPACING_M)),earned_steps=0,crossed_steps=0,arrived=distance<=1e-9)
                self.routes[key]=row
            else:
                row['current_distance_m']=distance
                if row['arrived']:continue
                row['best_distance_m']=min(row['best_distance_m'],distance)
                steps=min(row['budget_steps'],int(math.floor((row['start_distance_m']-row['best_distance_m']+1e-9)/SPACING_M)))
                if steps>row['crossed_steps'] and eligible[key] and not baseline:
                    delta=steps-row['crossed_steps']
                    self.events.append(dict(route=key,time_s=float(time_s),steps=delta,
                        points=round(POINTS*delta,10),distance_m=distance))
                    row['earned_steps']+=delta
                # Unheld/paused travel never earns retrospective navigation credit.
                row['crossed_steps']=steps
                row['arrived']=distance<=1e-9
            if eligible[key] and not row['arrived']:self.active.append(key)
        return self.summary()
    def summary(self):
        return dict(version=VERSION,phase='navigation' if self.active else 'operation',
            timeout_s=120. if self.active else 60.,active_routes=list(self.active),
            points=round(sum(r['earned_steps'] for r in self.routes.values())*POINTS,10),
            maximum=round(sum(r['budget_steps'] for r in self.routes.values())*POINTS,10),
            routes=deepcopy(self.routes),events=deepcopy(self.events),
            distance_reference='fixed_workspace_polygon_plus_existing_sensor_range',
            operation_range_m=self.radius,maximum_mode='activated_route_budgets')
    def augment(self,snapshot):
        out=deepcopy(snapshot)
        if 'navigation_progress' in out:raise ValueError('Navigation added twice')
        info=self.summary();scores=out['scores'];info['base_scores']=deepcopy(scores)
        scores['process']+=info['points'];scores['total']+=info['points'];scores['maximum']+=info['maximum']
        out['navigation_progress']=info
        out.setdefault('stages',[]).append(dict(name='navigation',score=info['points'],maximum=info['maximum'],
            supplementary=True,rule_version=VERSION,phase=info['phase']))
        return out
