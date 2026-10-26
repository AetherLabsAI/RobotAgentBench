"""Read geometry and interaction anchors from the referenced machine, in metres."""
from dataclasses import dataclass, asdict
from pxr import Gf, UsdGeom, UsdPhysics
from coffee_logic import Hopper

@dataclass(frozen=True)
class MachineContract:
    hopper: Hopper
    outlet: tuple
    cup: tuple
    button: tuple
    press_limit: float

    def to_dict(self):
        return asdict(self)

def read_contract(stage, path):
    cache=UsdGeom.XformCache()
    root=stage.GetPrimAtPath(path)
    inv=cache.GetLocalToWorldTransform(root).GetInverse()
    def local_point(prim,point):
        return inv.Transform(cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(*point)))
    def anchor(name):
        p=stage.GetPrimAtPath(path+'/TaskAnchors/'+name)
        if not p:raise ValueError('Missing machine anchor: '+name)
        return tuple(local_point(p,(0,0,0)))
    segments=stage.GetPrimAtPath(path+'/Body/Collision').GetChildren()
    rings=[p for p in segments if p.GetName().startswith('HopperSegment_')]
    if len(rings)<8:raise ValueError('Missing segmented hopper collisions')
    inner=[];edge=[]
    for p in sorted(rings,key=lambda p:str(p.GetPath())):
        pts=UsdGeom.Mesh(p).GetPointsAttr().Get()
        if len(pts)!=8:raise ValueError('Unexpected hopper collision schema')
        inner.extend(local_point(p,pts[i]) for i in (2,3,6,7))
        edge.append(local_point(p,pts[3]))
    xmin,xmax=min(v[0] for v in inner),max(v[0] for v in inner)
    zmin,zmax=min(v[2] for v in inner),max(v[2] for v in inner)
    if abs(xmin+xmax)>1e-5:raise ValueError('Expected hopper centred on local X')
    cz=(zmin+zmax)/2; hw=(xmax-xmin)/2; hd=(zmax-zmin)/2
    radius=hd-abs(edge[0][2]-cz)
    floor=stage.GetPrimAtPath(path+'/Body/Collision/HopperFloor')
    if not floor.IsA(UsdGeom.Cube):raise ValueError('Expected cube hopper floor')
    side=UsdGeom.Cube(floor).GetSizeAttr().Get()/2
    fy=max(local_point(floor,(x,y,z))[1] for x in (-side,side) for y in (-side,side) for z in (-side,side))
    hopper=Hopper(half_width=hw,half_depth=hd,corner_radius=radius,center_z=cz,
                  floor_y=fy,rim_y=max(v[1] for v in inner))
    if not (0<radius<min(hw,hd) and fy<hopper.rim_y):raise ValueError('Invalid hopper geometry')
    j=UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(path+'/Joints/TopButtonPress'))
    if j.GetAxisAttr().Get()!='Y':raise ValueError('Only local Y button motion supported')
    result=MachineContract(hopper,anchor('Outlet'),anchor('CupPlacement'),anchor('ButtonContact'),j.GetLowerLimitAttr().Get())
    if result.outlet[1]<=result.cup[1]+.10614:raise ValueError('Current cup does not fit below outlet')
    return result
