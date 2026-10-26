"""Task-independent House material repairs and fixed furniture lighting.

Only the prepared furniture frame and palette determine these opinions. No
trajectory, target, task ID or moving drawer is used to position a light.
"""
from __future__ import annotations

import math

SCHEMA = "real2sim.shared_house_environment_appearance.v1"
ROOT = "/World/HouseEnvironmentLighting"
TABLE = "/World/Assets/CoffeeTable"


def default_plan():
    return {"style": "canonical", "lights": [{"color_temperature_k": 5000.0}],
            "materials": [{"furniture_id": "coffee_table", "rgb_scale": [1., 1., 1.]}]}


def apply(stage, plan):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    palette = plan["style"]
    lights = plan["lights"]
    temperature = sum(float(x["color_temperature_k"]) for x in lights) / len(lights)
    if not math.isfinite(temperature):
        raise ValueError("shared House light temperature is not finite")
    group = UsdGeom.Xform.Define(stage, ROOT)
    group.SetResetXformStack(True)
    group.GetPrim().CreateAttribute("real2sim:centralAppearanceLighting", Sdf.ValueTypeNames.Bool).Set(True)
    group.GetPrim().CreateAttribute("real2sim:renderOnlyDecoration", Sdf.ValueTypeNames.Bool).Set(True)
    authored = []
    # Ambient illumination is shared by default and generated House layouts.
    # Keep ordinary shadow-casting room lights and use one environment dome.
    for path in ("/World/HouseTopologyCandidate/Lights/Ambient",):
        previous = stage.GetPrimAtPath(path)
        if previous:
            previous.GetAttribute("inputs:intensity").Set(0.0)
    dome = UsdLux.DomeLight.Define(stage, ROOT + "/Ambient")
    dome.CreateIntensityAttr(1600.0)
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(temperature)
    authored.append(str(dome.GetPath()))
    table = stage.GetPrimAtPath(TABLE)
    materials = []
    if table and table.IsActive():
        entry = next(x for x in plan["materials"] if x["furniture_id"] == "coffee_table")
        # The imported drawer liner has exactly-zero reflectance. Correct all
        # four drawers in every task, retaining the selected furniture tint.
        rgb = [0.32 * float(x) for x in entry["rgb_scale"]]
        for i in range(1, 5):
            path = f"{TABLE}/black_box_{i}/box/Looks/BlackMat/Shader"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdShade.Shader):
                raise ValueError("shared drawer material missing: " + path)
            diffuse = UsdShade.Shader(prim).GetInput("diffuse_color_constant")
            if not diffuse or diffuse.HasConnectedSource():
                raise ValueError("unexpected shared drawer diffuse binding: " + path)
            before = diffuse.Get()
            if before is None:
                raise ValueError("shared drawer diffuse has no value: " + path)
            changed = all(float(x) == 0.0 for x in before)
            if changed:
                diffuse.Set(Gf.Vec3f(*rgb))
            materials.append({"path": str(diffuse.GetAttr().GetPath()), "before": list(before),
                              "after": list(diffuse.Get()), "changed": changed})
        # Positions are in metres in the furniture's rigid frame, independent
        # of which drawer opens and of the motion or target being rendered.
        frame = UsdGeom.Xformable(table).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).RemoveScaleShear()
        for i, (x, y) in enumerate(((-.75, -.65), (-.75, .65), (.75, -.65), (.75, .65))):
            position = frame.Transform(Gf.Vec3d(x, y, .85))
            aim = frame.Transform(Gf.Vec3d(x * .45, y * .45, .30))
            light = UsdLux.RectLight.Define(stage, ROOT + f"/CoffeeTable_{i}")
            light.CreateWidthAttr(.4)
            light.CreateHeightAttr(.3)
            light.CreateIntensityAttr(6000.0)
            light.CreateNormalizeAttr(False)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(temperature)
            light.CreateColorAttr(Gf.Vec3f(1))
            UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(True)
            xf = UsdGeom.Xformable(light)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(position)
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0, 0, -1), (aim-position).GetNormalized()).GetQuat())
            authored.append(str(light.GetPath()))
    return {"schema": SCHEMA, "palette": palette, "light_paths": authored,
            "drawer_materials": materials, "task_dependent": False,
            "trajectory_dependent": False, "physics_effect": "none"}
