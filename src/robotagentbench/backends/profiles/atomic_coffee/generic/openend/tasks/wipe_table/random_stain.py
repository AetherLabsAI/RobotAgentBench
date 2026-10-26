"""Episode-local OpenEnd stains from the unchanged source shape generator.

No robot, physics, contact, opacity-cutout or scoring settings are changed.
The appearance stream is independent of the source generator's shape stream.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

VERSION='openend.wipe_stain.randomization.v1'
GENERATOR_SHA256='2720bf0d784c6349811f34ba82458e93840c0b515809639bb819f0a0f772dbe5'
# User-approved four equally likely natural stain-colour families. These are
# texture base colours, not opacity, cleaning difficulty or reward parameters.
PALETTE=(('coffee_brown',(0.30,0.22,0.14)),
         ('tea_yellow',(0.48,0.32,0.08)),
         ('juice_red',(0.44,0.12,0.10)),
         ('grey_brown',(0.27,0.25,0.22)))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def derived_seed(seed,stream):
    if type(seed) is not int or not 1<=seed<=4294967295:
        raise ValueError('Existing positive room-seed range required')
    # Source USD metadata uses a signed int32; avoid overflow for valid room
    # seeds above 2**31 while retaining deterministic independent streams.
    return int.from_bytes(hashlib.sha256((VERSION+':'+stream+':'+str(seed)).encode()).digest()[:4],'little')&0x7fffffff

def source_generator(root):
    path=Path(root)/'api-protocol/all_assets/table_stain_grid/sources/generate_stain.py'
    if sha(path)!=GENERATOR_SHA256:
        raise ValueError('Original stain generator version changed; re-audit required')
    spec=importlib.util.spec_from_file_location('openend_original_stain_shape',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path

def colour_choice(seed):
    colour_seed=derived_seed(seed,'colour')
    index=int(np.random.default_rng(colour_seed).integers(len(PALETTE)))
    name,rgb=PALETTE[index]
    return dict(family=name,base_rgb=list(rgb),colour_seed=colour_seed,
                palette_probability=1/len(PALETTE))

def generate(root,output,room_seed):
    """Create a fresh asset/layout/texture together; never edit a source asset."""
    output=Path(output)
    if output.resolve()!=output or not output.parent.is_dir():
        raise ValueError('Real existing episode parent required')
    if output.exists():raise ValueError('Episode stain output is immutable')
    source,path=source_generator(root)
    shape_seed=derived_seed(room_seed,'shape')
    appearance=colour_choice(room_seed)
    layout=source.make_layout(shape_seed)
    base,x,y,_=source.base_opacity(layout)
    mask,_,grain,dark=source.fields(layout)
    # Exact original shading/grain rule, applied to the selected base colour.
    shade=(0.75+0.5*(grain-0.5))*(1.25-0.35*dark)
    rgb=np.clip(np.asarray(appearance['base_rgb'])[None,None,:]*shade[...,None],0,1)
    rgba=np.round(np.dstack([rgb,mask])*255).astype(np.uint8)
    vertices=(source.SUB+1)**2
    cells=[]
    for k in range(source.NX*source.NY):
        fraction=float((base[k*vertices:(k+1)*vertices]>0).mean())
        cells.append(dict(k=k,ix=k%source.NX,iy=k//source.NX,
            stain_fraction=round(fraction,4),stained=fraction>=source.STAINED_FRACTION_MIN))
    stained=[c for c in cells if c['stained']]
    if not stained:raise ValueError('Source sampler generated no wipeable stain')
    # Copy only immutable grid/clearing definitions from the original asset.
    reference=Path(root)/'api-protocol/all_assets/table_stain_grid/asset_layout.json'
    original=json.loads(reference.read_text())
    grid=original['grid']
    if (grid['nx'],grid['ny'],grid['vertices_per_cell'],grid['wipes_to_clear'],grid['cutout_threshold'])!=(source.NX,source.NY,vertices,source.WIPES_TO_CLEAR,source.T):
        raise ValueError('Source generator and wipe runtime grid contract disagree')
    layout_doc=dict(schema=original['schema'],layout=layout,grid=grid,
        coverage_definition=original['coverage_definition'],cells=cells,
        summary=dict(cells=len(cells),stained_cells=len(stained),
            stain_area_m2=round(sum(c['stain_fraction'] for c in stained)*source.REGION_X*source.REGION_Y/len(cells),6)),
        randomization=dict(version=VERSION,room_seed=room_seed,shape_seed=shape_seed,appearance=appearance))
    output.mkdir()
    (output/'textures').mkdir()
    texture=output/'textures/stain.png'
    Image.fromarray(rgba).save(texture,optimize=False,compress_level=6)
    asset=output/'asset.usda';asset.write_text(source.build_usda(layout,base,x,y,'./textures/stain.png'))
    layout_path=output/'asset_layout.json'
    layout_path.write_text(json.dumps(layout_doc,sort_keys=True,indent=1)+'\n')
    record=dict(version=VERSION,room_seed=room_seed,shape_seed=shape_seed,appearance=appearance,
        source_hashes={str(path):GENERATOR_SHA256,str(reference):sha(reference),str(Path(__file__)):sha(__file__)},
        files={str(p):sha(p) for p in (asset,layout_path,texture)},
        source_shape_ranges_unchanged=True,scoring_changed=False,physics_changed=False)
    receipt=output/'randomization.json';receipt.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return dict(asset=asset,layout=layout_path,texture=texture,receipt=receipt,record=record)

def verify_generated(group):
    """Bind the actual appearance and cell table to the generated episode."""
    receipt=Path(group['randomization_file'])
    if sha(receipt)!=group['randomization_sha256']:
        raise ValueError('Generated stain receipt changed')
    record=json.loads(receipt.read_text())
    if record!=group['randomization'] or record.get('version')!=VERSION:
        raise ValueError('Generated stain provenance mismatch')
    if (record['appearance']!=colour_choice(record['room_seed']) or
            record['shape_seed']!=derived_seed(record['room_seed'],'shape')):
        raise ValueError('Generated stain seed mapping changed')
    for path,digest in {**record['source_hashes'],**record['files']}.items():
        if sha(path)!=digest:raise ValueError('Stain dependency changed: '+path)
    asset=receipt.parent/'asset.usda';layout=receipt.parent/'asset_layout.json'
    texture=receipt.parent/'textures/stain.png'
    if set(record['files'])!={str(asset),str(layout),str(texture)}:
        raise ValueError('Stain output closure is incomplete')
    if str(layout)!=group['layout_file'] or sha(layout)!=group['layout_sha256']:
        raise ValueError('Rendered stain is not the bound scoring layout')
    return record
