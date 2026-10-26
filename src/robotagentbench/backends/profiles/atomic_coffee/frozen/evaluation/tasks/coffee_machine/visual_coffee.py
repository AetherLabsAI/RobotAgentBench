"""Visual-only coffee. No particles, rigid bodies, collisions, or solver writes.

Accounting units retain the old 0.004096 ml quantum ONLY for API compatibility;
they are not simulated particles. Cup pose comes from the physical rigid body.
"""
import math
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

MODE = 'visual_cpu_v1'
ML_PER_UNIT = 0.004096

class VisualCoffee:
    def __init__(self, stage, root, cup_geometry, outlet, prefilled=0):
        self.geometry = cup_geometry
        self.outlet = outlet
        self.count = int(prefilled)
        self.in_cup = int(prefilled)
        self.missed = 0
        self.time = self.remainder = 0.
        self.pending = []
        self.cup_matrix = Gf.Matrix4d(1)
        mat = UsdShade.Material.Define(stage, root+'/Material')
        shader = UsdShade.Shader.Define(stage, root+'/Material/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.30,.105,.027))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.12)
        shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.33)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        self.surface = UsdGeom.Cylinder.Define(stage, root+'/CupSurface')
        self.surface.CreateAxisAttr('Y'); self.surface.CreateHeightAttr(.001)
        self.surface_op = self.surface.AddTransformOp()
        self.stream = UsdGeom.Cylinder.Define(stage, root+'/Stream')
        self.stream.CreateAxisAttr('Z'); self.stream.CreateRadiusAttr(.0025)
        self.stream_op = self.stream.AddTransformOp()
        for shape in (self.surface,self.stream):
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
            shape.CreateDisplayColorAttr([Gf.Vec3f(.30,.105,.027)])
            shape.MakeInvisible()
        # Numerically integrate the actual tapered cavity, all cup variants.
        self.heights=np.linspace(cup_geometry.floor_y,cup_geometry.rim_y-.002,256)
        radii=np.interp(self.heights,cup_geometry.heights,cup_geometry.inner)
        areas=math.pi*radii*radii
        self.volumes=np.r_[0,np.cumsum((areas[1:]+areas[:-1])*.5*np.diff(self.heights))]*1e6

    def sync_cup(self, position, quaternion):
        w,x,y,z=map(float,quaternion)
        self.cup_matrix=Gf.Matrix4d().SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
        self.cup_matrix.SetTranslateOnly(Gf.Vec3d(*map(float,position)))
        self.draw_surface()

    def draw_surface(self):
        if not self.in_cup:
            self.surface.MakeInvisible(); return
        height=float(np.interp(self.in_cup*ML_PER_UNIT,self.volumes,self.heights))
        radius=float(np.interp(height,self.geometry.heights,self.geometry.inner))-.0005
        self.surface.GetRadiusAttr().Set(radius)
        self.surface_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0,height,0))*self.cup_matrix)
        self.surface.MakeVisible()

    def update(self, dt, dispensing, machine_to_world):
        self.time+=dt
        start=machine_to_world.Transform(Gf.Vec3d(self.outlet[0],self.outlet[1]-.007,self.outlet[2]))
        emitted=0
        if dispensing and self.count<16000:
            self.remainder+=dt*120
            pulses=int(self.remainder+1e-9); self.remainder-=pulses
            emitted=min(pulses*7,16000-self.count)
            self.count+=emitted
            if emitted:self.pending.append((self.time+.10,emitted,tuple(start)))
        else:self.remainder=0.
        later=[]
        for arrival,n,origin in self.pending:
            if arrival>self.time+1e-9:later.append((arrival,n,origin));continue
            # Vertical visual stream intersects the real cup opening plane.
            up=self.cup_matrix.TransformDir(Gf.Vec3d(0,1,0))
            rim=self.cup_matrix.Transform(Gf.Vec3d(0,self.geometry.rim_y,0))
            o=Gf.Vec3d(*origin)
            travel=Gf.Dot(o-rim,up)/up[2] if up[2]>.1 else -1
            hit=o-Gf.Vec3d(0,0,max(0,travel))
            local=self.cup_matrix.GetInverse().Transform(hit)
            caught=travel>=0 and math.hypot(local[0],local[2])<self.geometry.opening_radius
            if caught:self.in_cup+=n
            else:self.missed+=n
        self.pending=later
        if dispensing or later:
            bottom=self.cup_matrix.Transform(Gf.Vec3d(0,float(np.interp(self.in_cup*ML_PER_UNIT,self.volumes,self.heights)),0))
            end=Gf.Vec3d(start[0],start[1],min(start[2]-.001,bottom[2]))
            direction=end-start; length=direction.GetLength()
            transform=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),direction/length))
            transform.SetTranslateOnly((start+end)*.5)
            self.stream.GetHeightAttr().Set(length)
            self.stream.GetRadiusAttr().Set(.0025*(1+.06*math.sin(self.time*31)))
            self.stream_op.Set(transform);self.stream.MakeVisible()
        else:self.stream.MakeInvisible()
        self.draw_surface()
        return emitted

    def counts(self):
        return dict(particles=self.count,in_cup=self.in_cup,airborne=sum(n for _,n,_ in self.pending),spilled=self.missed)
