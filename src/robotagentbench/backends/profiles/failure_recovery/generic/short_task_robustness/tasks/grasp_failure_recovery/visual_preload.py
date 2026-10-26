"""Approved task-layer visual deinstancing, before any renderer first loads.

Only shelf/bag procedural resets call this. Source assets and physical settings
stay untouched. The old one-off comparison and its receipts remain unchanged.
"""
from pathlib import Path
from .common import owned, digest, record_layers, verify_sources, write_json


def _protected_properties(stage):
    from pxr import Usd
    return {str(prop.GetPath()): repr(prop.Get() if hasattr(prop, 'Get') else prop.GetTargets())
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        for prop in prim.GetProperties()
        if prop.GetName().startswith(('physics:', 'physx', 'drive:', 'material:binding'))}


def apply(stage, output, record):
    """Promote the reviewed metadata into this generated root layer only."""
    from .visual_instance_compare import apply as checked_session_apply
    if record.get('variant') not in ('shelf_drink', 'coffee_bag'):
        raise ValueError('Visual preload is approved only for shelf/bag resets')
    if record.get('preparation_accepted') is not True:
        raise ValueError('CPU-rejected initial cannot receive an accepted visual receipt')
    out = owned(output)
    root, session = stage.GetRootLayer(), stage.GetSessionLayer()
    scene = out / 'initial.usda'
    if (Path(root.realPath).resolve() != scene.resolve() or
            stage.GetEditTarget().GetLayer() != root or not session.empty):
        raise ValueError('Fresh task-owned root and empty session required')
    before = _protected_properties(stage)
    try:
        stage.SetEditTarget(session)
        checked = checked_session_apply(stage)
        paths = checked['instanceable_disabled_paths']
        stage.SetEditTarget(root)
        for path in paths:
            stage.OverridePrim(path).SetInstanceable(False)
        session.Clear()
        if any(stage.GetPrimAtPath(path).IsInstance() for path in paths):
            raise ValueError('Visual preload opinions did not compose')
        if before != _protected_properties(stage):
            raise ValueError('Physical/material properties changed on promotion')
    finally:
        stage.SetEditTarget(root)
        session.Clear()
    receipt = dict(schema='grasp_failure_recovery.visual_preload.v1',
        adopted_as_default=True, user_approved=True, before_renderer_first_load=True,
        task_root_layer_only=True, instanceable_disabled_paths=paths,
        source_assets_written=False, physics_and_material_properties_unchanged=True,
        camera_renderer_parameters_changed=False, visual_acceptance=False)
    # This records composition only. RGB/native acceptance belongs to the worker.
    root.Save()
    sources = record_layers(stage, scene)
    verify_sources(sources)
    record.update(scene_sha256=digest(scene), source_layers=sources, visual_preload=receipt)
    write_json(out/'visual_preload.json', receipt)
    write_json(out/'initial.json', record)
    return record
