"""Copy the original dynamic umbrella and its material, never edit physics."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = '/World/Assets/Umbrella'


def copy_target(stage, destination, source_hashes, sha):
    from pxr import Sdf, Usd
    audit_path = HERE/'source.json'
    audit = json.loads(audit_path.read_text())
    entry = HERE/'umbrella.usda'
    if sha(entry) != audit['prototype_sha256']:
        raise ValueError('Umbrella source prototype changed')
    source = Usd.Stage.Open(str(entry))
    flat = source.Flatten()
    Sdf.CreatePrimInLayer(stage.GetRootLayer(), destination)
    if not Sdf.CopySpec(flat, TARGET, stage.GetRootLayer(), destination):
        raise ValueError('Umbrella copy failed')
    # Copy the source physics material into this object's namespace. Do not
    # replace the room's identically named GripMat or change its coefficients.
    old_material = audit['physics_material']
    new_material = destination+'/PhysicsMaterials/GripMat'
    Sdf.CreatePrimInLayer(stage.GetRootLayer(), new_material)
    if not Sdf.CopySpec(flat, old_material, stage.GetRootLayer(), new_material):
        raise ValueError('Umbrella material copy failed')
    for prim in Usd.PrimRange(stage.GetPrimAtPath(destination)):
        for relationship in prim.GetRelationships():
            paths = relationship.GetTargets()
            mapped = [Sdf.Path(new_material) if str(p)==old_material else
                      p.ReplacePrefix(Sdf.Path(TARGET), Sdf.Path(destination))
                      for p in paths]
            if mapped != paths:
                relationship.SetTargets(mapped)
        for attr in prim.GetAttributes():
            paths = attr.GetConnections()
            mapped = [p.ReplacePrefix(Sdf.Path(TARGET), Sdf.Path(destination)) for p in paths]
            if mapped != paths:
                attr.SetConnections(mapped)
    source_hashes[str(entry)] = sha(entry)
    source_hashes[str(audit_path)] = sha(audit_path)
    return dict(id='umbrella', path=destination, entry=str(entry),
                source_prim=TARGET, source_asset='existing_13-1_folded_umbrella',
                destination_role='fixed_cpu_bag', physics_preserved=True,
                native_physics_verified=False)
