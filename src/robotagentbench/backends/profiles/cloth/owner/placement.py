"""Measured triangle-union coverage and exact source basket profile containment."""
import numpy as np


def convex_polygon(points):
    points=sorted(set(map(tuple,np.asarray(points,float))))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    halves=[]
    for rows in (points,points[::-1]):
        half=[]
        for p in rows:
            while len(half)>=2 and cross(half[-2],half[-1],p)<=0:half.pop()
            half.append(p)
        halves.append(half[:-1])
    polygon=halves[0]+halves[1]
    if len(polygon)<3:raise ValueError('No physical table top polygon')
    return polygon


def projected_coverage(mesh,points,*,table_z,table_polygon_xy,support_tolerance_m,denominator_area):
    from shapely import polygons,union_all,Polygon
    q=mesh.positions(points);f=mesh.faces
    if support_tolerance_m<=0 or denominator_area<=0:raise ValueError('Explicit source/approved geometry required')
    region=Polygon(np.asarray(table_polygon_xy,float))
    if not region.is_valid or region.area<=0:raise ValueError('Bound physical table polygon required')
    tri=q[f];supported=np.all(abs(tri[:,:,2]-table_z)<=support_tolerance_m,axis=1)
    # A triangle whose XY AABB is strictly disjoint from the table AABB has
    # exactly zero intersection with that table. Do not union thousands of
    # such faces at every native substep while the cloth is still in a basket.
    # Keep touching/crossing/enclosing faces, including faces with ALL vertices
    # outside the table. Round-off only widens this broad phase; it never
    # changes the actual support threshold, clipping region or final union.
    bounds=np.asarray(region.bounds,float)
    guard=256*np.finfo(float).eps*max(1.,float(abs(bounds).max()),float(abs(q).max()))
    lo=tri[:,:,:2].min(1);hi=tri[:,:,:2].max(1)
    candidate=np.all(hi>=bounds[:2]-guard,axis=1)&np.all(lo<=bounds[2:]+guard,axis=1)
    # Union is essential: a folded double layer must not count its area twice.
    all_projected=union_all(polygons(tri[candidate,:,:2])).intersection(region)if candidate.any()else Polygon()
    eligible=supported&candidate
    contact_projected=union_all(polygons(tri[eligible,:,:2])).intersection(region)if eligible.any()else Polygon()
    return dict(coverage=float(min(1.,contact_projected.area/denominator_area)),
        projected_area_on_table_m2=float(all_projected.area),supported_projected_area_m2=float(contact_projected.area),
        material_area_m2=mesh.area,unfolded_projection_ratio=float(all_projected.area/mesh.area),
        maximum_face_height_deviation_m=float(abs(q[:,2]-table_z).max()),
        method='projected_triangle_union_clipped_to_measured_table_no_double_counting')


class BasketProfile:
    def __init__(self,source,*,rim_z):
        self.profiles={int(r['axis']):np.asarray(r['rows'],float)for r in source['profiles']}
        if set(self.profiles)!={0,1}:raise ValueError('Both measured wall profiles required')
        for rows in self.profiles.values():
            if rows.ndim!=2 or rows.shape[1]!=5 or np.any(np.diff(rows[:,0])<=0)or not np.isfinite(rows).all():raise ValueError('Invalid source wall profile')
        floor=next(r for r in source['colliders']if r['name']=='floor')
        self.floor_z=float(np.asarray(floor['world_vertices_m'])[:,2].max());self.rim_z=float(rim_z)
        if self.rim_z<=self.floor_z:raise ValueError('Invalid basket vertical range')

    def _inside(self,points):
        valid=(points[:,2]>=self.floor_z)&(points[:,2]<=self.rim_z)
        for axis,rows in self.profiles.items():
            lo=np.interp(points[:,2],rows[:,0],rows[:,2]);hi=np.interp(points[:,2],rows[:,0],rows[:,3])
            valid&=(points[:,axis]>=lo)&(points[:,axis]<=hi)
        return valid

    def contains(self,mesh,points):
        q=mesh.positions(points)
        if not self._inside(q).all():return False
        edges=np.unique(np.sort(np.concatenate([mesh.faces[:,[0,1]],mesh.faces[:,[1,2]],mesh.faces[:,[2,0]]]),axis=1),axis=0)
        a,b=q[edges[:,0]],q[edges[:,1]];dz=b[:,2]-a[:,2]
        # Check edge/profile-break intersections too. Vertex-only tests can
        # miss triangles cutting through a tapered/concave wall between nodes.
        levels=np.unique(np.concatenate([r[:,0]for r in self.profiles.values()]))
        levels=levels[(levels>q[:,2].min())&(levels<q[:,2].max())]
        for z in levels:
            active=(np.minimum(a[:,2],b[:,2])<z)&(np.maximum(a[:,2],b[:,2])>z)
            fraction=(z-a[active,2])/dz[active]
            intersections=a[active]+fraction[:,None]*(b[active]-a[active])
            if len(intersections)and not self._inside(intersections).all():return False
        return True


class ReleasedStability:
    """Anchor-span test; the existing approved scorer owns the 1s duration."""
    def __init__(self,*,position_span_m):
        if position_span_m<=0:raise ValueError('Existing source stability tolerance required')
        self.tolerance=position_span_m;self.anchor=None
    def observe(self,points,eligible):
        q=np.asarray(points,float)
        if not eligible:self.anchor=None;return False
        if self.anchor is None:self.anchor=q.copy();return True
        if np.linalg.norm(q-self.anchor,axis=1).max()>self.tolerance:
            self.anchor=q.copy();return False
        return True


def material_stability(mesh,frames,*,position_span_m,area_fraction_min):
    stack=np.asarray(frames,float)
    if stack.ndim!=3 or stack.shape[1:]!=(len(mesh.uv),3)or not len(stack)or not np.isfinite(stack).all():raise ValueError('Complete finite material history required')
    if position_span_m<=0 or not 0<area_fraction_min<=1:raise ValueError('Explicit approved cloth stability policy required')
    maximum=np.linalg.norm(stack-stack[0],axis=2).max(0)
    return _stable_area(mesh,maximum,position_span_m,area_fraction_min)


def _stable_area(mesh,maximum,position_span_m,area_fraction_min):
    # A linear material triangle is guaranteed within the displacement ball if
    # all its vertices are. This is a conservative area bound, not a vertex %.
    good=np.all(maximum[mesh.faces]<=position_span_m,axis=1)
    fraction=float(mesh.areas[good].sum()/mesh.area)
    return dict(stable=bool(fraction+1e-12>=area_fraction_min),stable_material_area_fraction=fraction,
        maximum_nodal_displacement_m=float(maximum.max()),position_span_m=position_span_m,
        minimum_material_area_fraction=area_fraction_min,method='persistent_rest_triangle_area_bound')


class MaterialReleasedStability:
    """Track the same 95% area against one anchor; scorer owns 1s duration."""
    def __init__(self,mesh,*,position_span_m,area_fraction_min):
        self.mesh,self.tolerance,self.fraction=mesh,position_span_m,area_fraction_min
        self.anchor=self.maximum=None;self.last_measure=None
    def observe(self,points,eligible):
        q=self.mesh.positions(points)
        if not eligible:self.anchor=self.maximum=None;return False
        if self.anchor is None:self.anchor=q.copy();self.maximum=np.zeros(len(q))
        self.maximum=np.maximum(self.maximum,np.linalg.norm(q-self.anchor,axis=1))
        self.last_measure=_stable_area(self.mesh,self.maximum,self.tolerance,self.fraction)
        if not self.last_measure['stable']:
            self.anchor=q.copy();self.maximum=np.zeros(len(q));return False
        return True
