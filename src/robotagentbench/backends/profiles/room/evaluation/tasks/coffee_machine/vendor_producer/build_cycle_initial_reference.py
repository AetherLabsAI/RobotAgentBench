"""Author a seeded bag-cycle start before physics; never move runtime objects."""
import argparse, copy, json, random, shutil
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from fit_bag_bottom_visual import fit_bottom
from appearance_randomization import apply_appearance
from hopper_funnel import apply_funnel

ROOT = Path(__file__).resolve().parent.parent
BAG = '/World/CoffeeTaskPreview/BeanBag'
MACHINE = '/World/CoffeeTaskPreview/Machine'
CUP = '/World/CoffeeTaskPreview/Cup'

def matrix(value):
    return Gf.Matrix4d(*[value[j][i] for i in range(4) for j in range(4)])

def serial(value):
    return [[float(value[j][i]) for j in range(4)] for i in range(4)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--reference-layout', action='store_true')
    p.add_argument('--machine-pose-json',help='Pre-physics stratified machine sample [x,y,yaw_deg]')
    p.add_argument('--bag-width-m', type=float, help='Opt-in narrow rigid bag, baked before physics')
    p.add_argument('--bag-depth-m', type=float, help='Optional compact bag-local Z depth')
    a = p.parse_args(); out = a.out.resolve(); out.mkdir(exist_ok=False); out.chmod(0o777)
    source = ROOT/'runs/bag_load_100_scale150_126'
    meta = json.loads((ROOT/'runs/pour_initial_034/initialization.json').read_text())
    old = copy.deepcopy(meta['randomization']); rng = random.Random(a.seed)
    specs = json.loads((ROOT/'cup_place/cup_variants.json').read_text())
    sample = copy.deepcopy(old) if a.reference_layout else dict(
        machine_offset_m=[rng.uniform(-.10,.10),rng.uniform(-.10,.10),0.],
        machine_yaw_deg=rng.uniform(-20,20),
        bag_offset_m=[rng.uniform(-.05,.05),rng.uniform(-.05,.05),0.],
        bag_yaw_deg=rng.uniform(0,90),cup_style=rng.choice(list(specs)))
    if a.machine_pose_json:
        if a.reference_layout:raise ValueError('Cannot override reference layout')
        x,y,yaw=map(float,json.loads(a.machine_pose_json))
        if not (-.1<=x<=.1 and -.1<=y<=.1 and -20<=yaw<=20):raise ValueError('Machine sample outside approved range')
        sample.update(machine_offset_m=[x,y,0.],machine_yaw_deg=yaw)
    for name in ['shell_override.usda','parent_shell.usda','original_mass.json','build_report.json','bean_load.json']:
        (out/name).write_text((source/name).read_text().replace(str(source),str(out)))
    # The source scene already contains the complete enlarged-bean geometry.
    stage = Usd.Stage.CreateNew(str(out/'bean_load.usda'))
    stage.GetRootLayer().subLayerPaths = [str(source/'initial.usda')]
    cache = UsdGeom.XformCache()
    worlds = {path:cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)) for path in [BAG,MACHINE,CUP]}
    def delta(key,world):
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),sample[key+'_yaw_deg']-old[key+'_yaw_deg']))
        origin = world.ExtractTranslation()
        offset = Gf.Vec3d(*[sample[key+'_offset_m'][i]-old[key+'_offset_m'][i] for i in range(3)])
        rot.SetTranslateOnly(origin+offset-rot.TransformDir(origin)); return rot
    db, dm = delta('bag',worlds[BAG]), delta('machine',worlds[MACHINE])
    load = json.loads((out/'bean_load.json').read_text())
    poses = {BAG:worlds[BAG]*db,MACHINE:worlds[MACHINE]*dm,CUP:worlds[CUP]*dm}
    # Cup style and machine-relative tray placement are initialized together.
    cup_layer = Sdf.Layer.FindOrOpen(str(ROOT/'assets/cups'/(sample['cup_style']+'.usda')))
    Sdf.CreatePrimInLayer(stage.GetRootLayer(),CUP)
    assert Sdf.CopySpec(cup_layer,'/CoffeeCup',stage.GetRootLayer(),CUP)
    stage.GetPrimAtPath(CUP).GetReferences().SetReferences([Sdf.Reference(str(ROOT/'coffee_cup_dynamic.usda'))])
    for entry in load['entries']:
        poses[entry['path']] = matrix(entry['local_matrix'])*poses[BAG]
    for path,world in poses.items():
        prim = stage.GetPrimAtPath(path)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(world*cache.GetParentToWorldTransform(prim).GetInverse())
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateVelocityAttr(Gf.Vec3f(0)); rb.CreateAngularVelocityAttr(Gf.Vec3f(0))
    report = json.loads((out/'build_report.json').read_text())
    report['visual_adjustment'] = fit_bottom(stage, report)
    report['visual_meshes_unchanged'] = False
    report['machine_hopper_adjustment'] = apply_funnel(stage)
    appearance = apply_appearance(stage, a.seed)
    (out/'appearance.json').write_text(json.dumps(appearance, indent=2))
    stage.GetRootLayer().Save()
    layer = Sdf.Layer.CreateNew(str(out/'initial.usda'))
    layer.subLayerPaths = [str(out/'shell_override.usda'),str(out/'bean_load.usda')]; layer.Save()
    if a.bag_width_m is not None:
        from resize_bag import resize
        resized_stage = Usd.Stage.Open(str(out/'initial.usda'))
        resize(resized_stage, report, load, a.bag_width_m, a.bag_depth_m)
        resized_stage.GetRootLayer().Save()
        (out/'bean_load.json').write_text(json.dumps(load, indent=2))
        meta['bag_geometry_revision'] = report['bag_resize']
    meta.update(seed=a.seed,randomization=sample,appearance=appearance,machine_hopper_revision=report['machine_hopper_adjustment']['revision'],bag_world=serial(poses[BAG]),
        machine_world=serial(poses[MACHINE]),cup_world=serial(poses[CUP]),
        bean_paths=[e['path'] for e in load['entries']],
        scope='Authored pre-physics bag-cycle initialization; no physical success claim.',
        checks={'authored_layout':True},passed=True,
        bag_cycle_initialization={'version':1,'reference_layout':a.reference_layout,
        'source':'runs/pour_initial_034','ranges':{'machine_xy_m':.10,'machine_yaw_deg':20.,'bag_xy_m':.05,'bag_yaw_deg':[0.,90.]}})
    (out/'initialization.json').write_text(json.dumps(meta,indent=2))
    report['cycle_initialization'] = 'initialization.json'
    (out/'build_report.json').write_text(json.dumps(report,indent=2))
    (out/'collision_model').mkdir(); (out/'collision_model').chmod(0o777)
    shutil.copy2(source/'collision_model/gripper_calibration.json',out/'collision_model/gripper_calibration.json')
    print('CYCLE_INITIALIZED',json.dumps(sample),flush=True)

if __name__ == '__main__':main()
