"""Task-local opaque door window. No geometry or physics edits."""


def apply_opaque_door(stage):
    from pxr import Sdf, UsdShade

    root = "/World/MicrowaveDoorRefinementLooks"
    glass = UsdShade.Material(stage.GetPrimAtPath(root + "/ReferenceDoorGlass"))
    original = UsdShade.Shader(stage.GetPrimAtPath(root + "/ReferenceDoorGlass/OmniGlass"))
    screen = UsdShade.Shader(stage.GetPrimAtPath(root + "/BlackPerforatedScreen/pbr"))
    if not glass or not original or not screen:
        raise ValueError("Expected approved microwave door materials")

    # OmniGlass transmits light even without an opacity input. Use its existing
    # tint/roughness/IOR in an opaque surface; retain the original shader for
    # provenance. This removes transmission, not the door's physical window.
    opaque = UsdShade.Shader.Define(stage, root + "/ReferenceDoorGlass/OpaqueSurface")
    opaque.CreateIdAttr("UsdPreviewSurface")
    opaque.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(original.GetInput("glass_color").Get())
    opaque.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(original.GetInput("frosting_roughness").Get())
    opaque.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(original.GetInput("glass_ior").Get())
    opaque.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    opaque.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(original.GetInput("reflection_color").Get())
    opaque.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    opaque.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    for name in ("surface", "volume", "displacement"):
        attr = glass.GetPrim().GetAttribute("outputs:mdl:" + name)
        if attr:
            attr.SetConnections([])
    glass.CreateSurfaceOutput().ConnectToSource(opaque.ConnectableAPI(), "surface")
    # Explicitly disconnect the perforation alpha: a scalar fallback alone
    # would leave the old opacity texture active.
    screen.CreateInput("opacity", Sdf.ValueTypeNames.Float).GetAttr().SetConnections([])
    screen.GetInput("opacity").Set(1.0)
    return {"opacity": 1.0, "scope": "microwave_put", "physics_changed": False}
