"""One frontal workspace owns scene Hz. No scoring state is an input.

Uses the existing calibrated sector/object-polygon intersection. This selector
does not sleep bodies, disable collisions, move assets or write the USD stage.
"""
from dataclasses import dataclass, asdict
import math
from openend.sector import sector_polygon, angle_error


@dataclass(frozen=True)
class Workspace:
    id: str
    path: str
    polygon: tuple
    hz: int

    def __post_init__(self):
        if not self.id or not self.path.startswith('/') or type(self.hz) is not int or self.hz not in (60, 240):
            raise ValueError('Explicit bound workspace and approved frequency required')
        p=self.polygon
        if len(p)<3 or any(len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for v in p):
            raise ValueError('Finite measured workspace polygon required')
        if abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1])))<=1e-9:
            raise ValueError('Degenerate workspace polygon')


class WorkspacePhysics:
    def __init__(self, workspaces, sensor, *, unavailable=()):
        self.workspaces=tuple(workspaces)
        if len({w.id for w in self.workspaces})!=len(self.workspaces):raise ValueError('Duplicate workspace')
        self.radius=sensor['range'];self.half=sensor['halfAngle']
        if not math.isfinite(self.radius) or self.radius<=0 or not 0<self.half<=180:raise ValueError('Calibrated sector required')
        self.unavailable=tuple(unavailable)

    def select(self, robot):
        if len(robot['xy'])!=2 or not all(math.isfinite(v) for v in (*robot['xy'],robot['yaw'])):
            raise ValueError('Measured robot pose required')
        candidates=[]
        for w in self.workspaces:
            if not sector_polygon(robot,self.radius,self.half,w.polygon):continue
            centre=tuple(sum(p[i] for p in w.polygon)/len(w.polygon) for i in (0,1))
            dx,dy=(centre[i]-robot['xy'][i] for i in (0,1));distance=math.hypot(dx,dy)
            angle=0. if distance<=1e-9 else angle_error(robot['yaw'],math.degrees(math.atan2(dy,dx)))
            candidates.append(dict(id=w.id,path=w.path,heading_error_deg=angle,distance_m=distance,physics_hz=w.hz))
        # Frequency NEVER participates in ranking. Distance only breaks an
        # equal-heading tie; stable identity is the final deterministic tie.
        candidates.sort(key=lambda r:(r['heading_error_deg'],r['distance_m'],r['id']))
        chosen=candidates[0] if candidates else None
        return dict(current_workspace=None if chosen is None else chosen['id'],
                    workspace_path=None if chosen is None else chosen['path'],
                    requested_physics_hz=60 if chosen is None else chosen['physics_hz'],
                    workspace_candidates=candidates,physics_rate_owner='single_frontal_workspace')

    def receipt(self):
        return dict(policy='sector_hit_then_front_then_distance',navigation_hz=60,
                    control_hz=20,workspaces=[asdict(w) for w in self.workspaces],
                    unavailable=list(self.unavailable),scoring_state_used=False,
                    physical_sleep_or_collision_changes=False)


def from_scene(stage, manifest, sensor):
    """Bind current scene bounds, never copy layout01 world coordinates.

    Rates retained from physics_rate_design.json: explicit sofa/table overrides
    first, other regions' single observed rate. Unresolved cloth/prep/dining are
    reported, not assigned guessed frequencies or support-proxy coordinates.
    """
    from pxr import UsdGeom, Usd
    fixed=(('sofa','/World/Assets/Sofa',60),
           ('side','/World/Assets/SmallCoffeeTable',60),('tv','/World/Assets/TvTable',60),
           ('coat','/World/Assets/CoatRack',240),('bag','/World/Assets/Task5PackStation',240),
           ('microwave','/World/Assets/DiningTableMicrowave',240))
    objects=(('coffee','coffee_machine',240),('fryer','air_fryer',240),
             ('shelf','snack_shelf',60),('fridge','refrigerator',60))
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
    entries=[];unavailable=[dict(id=i,reason='unresolved_rate_or_binding_in_existing_design') for i in ('cloth','prep','dining')]
    rows=list(fixed)
    for ident,asset,hz in objects:
        matches=[o for o in manifest['objects'] if o['id']==asset]
        if len(matches)>1:raise ValueError('Ambiguous workspace '+ident)
        if matches:rows.append((ident,matches[0]['path'],hz))
        else:unavailable.append(dict(id=ident,reason='asset_absent'))
    for ident,path,hz in rows:
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive() or UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':
            unavailable.append(dict(id=ident,reason='missing_or_invisible_anchor',path=path));continue
        r=bounds.ComputeWorldBound(prim).ComputeAlignedRange();lo,hi=r.GetMin(),r.GetMax()
        polygon=((float(lo[0]),float(lo[1])),(float(hi[0]),float(lo[1])),
                 (float(hi[0]),float(hi[1])),(float(lo[0]),float(hi[1])))
        entries.append(Workspace(ident,path,polygon,hz))
    # The table has three independently selectable regions, not a fourth
    # competing whole-table candidate. This does not activate object scorers.
    from openend.tasks.coffee_table_workspaces.binding import bind
    entries.extend(bind(stage, manifest))
    return WorkspacePhysics(entries,sensor,unavailable=unavailable)
