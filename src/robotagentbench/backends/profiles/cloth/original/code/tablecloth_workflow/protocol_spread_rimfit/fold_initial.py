"""Before-time-zero folded cloth, leaning into real inner wall and following oval rim."""
import numpy as np

def long_midpoint_fold(uv,size,bounds,particle_radius=.0015):
 uv=np.asarray(uv,float);width,length=map(float,size);s=uv[:,1]-length/2;distance=np.abs(s);sign=np.sign(s);rb=.002;floor=float(bounds[0,2])+.024;z0=floor+particle_radius+.004
 rim_y=float(bounds[0,1])+.010;rim_z=float(bounds[1,2])-.020;crease_y=rim_y+.030
 angle=np.minimum(distance/rb,np.pi/2);y=crease_y+sign*rb*np.sin(angle);z=z0+rb*(1-np.cos(angle));after=distance-np.pi*rb/2
 for side in [-1,1]:
  ids=np.flatnonzero((sign==side)&(after>0));ry=.014+side*rb;rz=.027+side*rb;start_y=crease_y+side*rb;start_z=z0+rb;end_y=rim_y+ry;rise=rim_z-start_z;dy=end_y-start_y;leg=np.hypot(rise,dy);d=after[ids];ratio=np.minimum(d/leg,1);y[ids]=start_y+ratio*dy;z[ids]=start_z+ratio*rise
  over=ids[d>leg]
  if len(over):
   phi=np.linspace(0,np.pi,2049);speed=np.sqrt((ry*np.sin(phi))**2+(rz*np.cos(phi))**2);arc=np.r_[0,np.cumsum((speed[1:]+speed[:-1])*.5*np.diff(phi))];travel=after[over]-leg;t=np.interp(np.minimum(travel,arc[-1]),arc,phi);y[over]=rim_y+ry*np.cos(t);z[over]=rim_z+rz*np.sin(t)-np.maximum(travel-arc[-1],0)
 q=np.c_[bounds[:,0].mean()+uv[:,0]-width/2,y,z];crease=abs(s)<1e-6
 return q,dict(material_long_axis='v',crease_v_m=length/2,crease_direction_world='X',basket_edge='y_min',support='existing tapered inner wall, oval rim and floor only',bottom_radius_m=rb,rim_horizontal_radii_m=[.012,.016],rim_vertical_radii_m=[.025,.029],crease_y_m=crease_y,crease_min_z_m=float(z[crease].min()),crease_max_z_m=float(z[crease].max()),basket_floor_m=floor,initialization_only=True,attachments=False)
