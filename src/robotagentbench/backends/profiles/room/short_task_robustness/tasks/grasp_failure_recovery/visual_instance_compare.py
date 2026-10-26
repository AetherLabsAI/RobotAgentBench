"""One user-approved visual-only instance A/B; no source file is edited."""
import hashlib


def apply(stage):
    from pxr import Usd, UsdPhysics
    root=stage.GetPrimAtPath('/World/Robot')
    if not root or stage.GetEditTarget().GetLayer()!=stage.GetSessionLayer():
        raise ValueError('Visual comparison must stay in this task session layer')

    def physical_properties():
        # Traverse proxies on BOTH sides so de-instancing cannot make hidden
        # existing properties appear as fictitious physical additions.
        return {str(prop.GetPath()):repr(prop.Get() if hasattr(prop,'Get') else prop.GetTargets())
            for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies())
            for prop in prim.GetProperties()
            if prop.GetName().startswith(('physics:','physx','drive:','material:binding'))}

    before=physical_properties()
    before_session=stage.GetSessionLayer().ExportToString()
    targets=[]
    for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        path=str(p.GetPath())
        if p.IsInstance() and (path.endswith('/visuals') or '/visuals/' in path):
            if p.IsInstanceProxy():
                raise ValueError('Nested instance proxy cannot be directly authored')
            for child in Usd.PrimRange(p,Usd.TraverseInstanceProxies()):
                if any(child.HasAPI(api) for api in (UsdPhysics.CollisionAPI,UsdPhysics.RigidBodyAPI,
                                                     UsdPhysics.ArticulationRootAPI)) or child.IsA(UsdPhysics.Joint):
                    raise ValueError('Visual-only subtree unexpectedly contains physics: '+str(child.GetPath()))
            targets.append(path)
    if not targets:
        raise ValueError('No independent visual instances found; no comparison performed')
    for path in targets:
        stage.GetPrimAtPath(path).SetInstanceable(False)
    after=physical_properties()
    if before!=after:
        raise ValueError('Physical/material properties changed during visual comparison')
    session=stage.GetSessionLayer().ExportToString()
    return dict(schema='grasp_failure_recovery.visual_instance_comparison.v1',
        authorized_once_by_user=True,adopted_as_default=False,
        instanceable_disabled_paths=targets,source_assets_written=False,
        session_layer_only=True,physics_and_material_properties_unchanged=True,
        camera_renderer_parameters_changed=False,
        before_session_sha256=hashlib.sha256(before_session.encode()).hexdigest(),
        after_session_sha256=hashlib.sha256(session.encode()).hexdigest(),
        authored_session_layer=session)
