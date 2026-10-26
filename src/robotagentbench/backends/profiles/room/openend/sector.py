"""Python mirror of web/logic.js for the isolated native navigation probe.

Parity is checked against the unchanged JS, including every recorded native pose.
No centre-distance substitute for the closed sector/polygon intersection.
"""
import math
EPS = 1e-9

def distance(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def angle_error(a, b): return abs((a-b+540) % 360-180)
def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def on_segment(a,b,p):
    return abs(cross(a,b,p)) <= EPS and min(a[0],b[0])-EPS <= p[0] <= max(a[0],b[0])+EPS and min(a[1],b[1])-EPS <= p[1] <= max(a[1],b[1])+EPS
def segments(a,b,c,d):
    return (any((on_segment(a,b,c),on_segment(a,b,d),on_segment(c,d,a),on_segment(c,d,b)))
            or cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0)
def inside_polygon(p,poly):
    inside=False
    for i,a in enumerate(poly):
        b=poly[i-1]
        if on_segment(a,b,p): return True
        if (a[1]>p[1]) != (b[1]>p[1]) and p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]: inside=not inside
    return inside
def sector_polygon(robot,radius,half,poly):
    o=robot['xy']
    def in_angle(p):
        return distance(o,p)<=EPS or angle_error(robot['yaw'],math.degrees(math.atan2(p[1]-o[1],p[0]-o[0])))<=half+EPS
    if inside_polygon(o,poly) or any(distance(o,p)<=radius+EPS and in_angle(p) for p in poly): return True
    rays=[[o[0]+radius*math.cos(math.radians(robot['yaw']+v)),o[1]+radius*math.sin(math.radians(robot['yaw']+v))] for v in (-half,half)]
    for i,a in enumerate(poly):
        b=poly[(i+1)%len(poly)]
        if half<180 and any(segments(o,r,a,b) for r in rays): return True
        dx,dy=b[0]-a[0],b[1]-a[1];ax,ay=a[0]-o[0],a[1]-o[1]
        A=dx*dx+dy*dy
        if A<EPS*EPS: continue
        B=2*(ax*dx+ay*dy);C=ax*ax+ay*ay-radius*radius;disc=B*B-4*A*C
        if disc < -EPS: continue
        for t in ((-B-math.sqrt(max(0,disc)))/(2*A),(-B+math.sqrt(max(0,disc)))/(2*A)):
            if -EPS<=t<=1+EPS and in_angle([a[0]+t*dx,a[1]+t*dy]): return True
    return False
def step(previous,c,robot,flags):
    d=distance(robot['xy'],c['center']);td=distance(robot['xy'],c['target'])
    bearing=math.degrees(math.atan2(c['target'][1]-robot['xy'][1],c['target'][0]-robot['xy'][0]))
    he=None if td<1e-9 else angle_error(robot['yaw'],bearing)
    hit=sector_polygon(robot,c['wake'],c['wakeHalfAngle'],c['targetPolygon'])
    protected=flags.get('held',False) or flags.get('moving',False)
    near=td<=c['nearWake']
    reason='protected' if protected else 'near' if near else 'object_overlap' if hit else 'retained' if previous.get('active',False) and d<=c['exit'] else 'inactive'
    active=True if hit or near or protected else False if d>c['exit'] else previous.get('active',False)
    state=('PINNED' if protected else 'ACTIVE') if active else 'PRELOAD' if d<=c['preload'] else 'DORMANT'
    pd=distance(robot['xy'],c['pose']);ae=angle_error(robot['yaw'],c['poseYaw'])
    ready=active and flags.get('task',False) and flags.get('stopped',False) and flags.get('loaded',False) and pd<=c['poseRadius'] and ae<=c['poseYawTolerance']
    return dict(active=active,state=state,ready=bool(ready),distance=d,poseDistance=pd,angle=ae,targetDistance=td,bearing=bearing,headingError=he,facing=he is not None and he<=c['wakeHalfAngle'],reason=reason,overlap=hit)
