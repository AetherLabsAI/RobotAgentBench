"""Install missing byte-identical preparation modules in an owned source view.

Existing files are never replaced. Fridge identities are recorded in
acceptance_v16/reviewed_initial_inputs.json; the other original initializer
identities retain their individually reviewed canonical source bytes.
No evaluator, runtime, physics module or component-source record is changed.
"""
import hashlib
import json
from pathlib import Path

ORIGINAL_PREPARATION = {
    'short_tasks/tasks/sofa_to_drawer/initial_station.py':
        '0c9c465e34d8f1f3ecf444645d5f6d13d8e7fced09682a4eb9b96b717400bdfe',
    'short_tasks/tasks/sofa_to_drawer/prepare_scene.py':
        '79f075dbfb5fd67390e0f0b95416c8b4e6ab81362be12c98bb986ee612a8e9ec',
    'short_tasks/tasks/sofa_to_drawer/prepare_initial.py':
        '7b91601fc4b6af33b76660b5b3947249f3ffa7aa3388714ddaa47926299b8c9c',
    'short_tasks/tasks/sofa_to_coffee_table/prepare_initial.py':
        '12218ed5e81e7b2c288dfeb4853e55a4bac7e23e6f33ce2fffe4fa165c6fccba',
    'short_tasks/tasks/fridge_to_heated_box/scene.py':
        '9571e8b0610e29e19b6e296728d321a8c59c5cf41088bab5b53e7b0911c77f2e',
    'short_tasks/tasks/fridge_to_heated_box/sources.py':
        '514820a911cfeae7f88db6faf06d7d1aba432125f90e9aa4d7b27fa0640ee6d2',
}
ORIGINAL_LAYERS = {
    'assets/house_scene_l1/demo_sim/scene_room_livingroom_cleanup.usda': (
        'scene_room_livingroom_cleanup.usda',
        '1b29b947952881bf4894f56e71cbe16b4c2c9de462ae88aef381cf24bc7ad26e'),
}
# The small original USD wrapper is embedded for wheel installations whose
# package-data policy includes Python sources only. Bytes remain hash checked.
LAYER_BYTES = b'''#usda 1.0
(
    defaultPrim = ""
    doc = "Livingroom-cleanup task scene: the receipt-validated scene_room_baseline.usda plus the public coffee-table drawer slide layer (livingroom_drawer_slide.usda, user-approved 2026-09-15). No prim opinions of its own."
    metersPerUnit = 1
    subLayers = [
        @./livingroom_drawer_slide.usda@,
        @./scene_room_baseline.usda@
    ]
    timeCodesPerSecond = 60
    upAxis = "Z"
)

'''


def _destination(root,relative):
    target=root/relative
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError('Original resource target escaped the owned source')
    return target


def install(runtime, *, owned_parent):
    """Explicit preparation-only install; invoke before entering SourceTree."""
    runtime = Path(runtime).resolve(strict=True)
    owned_parent = Path(owned_parent).resolve(strict=True)
    if not runtime.is_relative_to(owned_parent):
        raise ValueError('Source installation must remain inside owned task output')
    marker = json.loads((runtime.parent / 'OWNER.json').read_text())
    if marker != {'owner': 'RobotAgentBench-short-instance-bank-v31'}:
        raise ValueError('Independent bank view required')
    config_path = runtime / 'runtime.json'
    config = json.loads(config_path.read_text())
    root = (runtime / Path(config['source_root']).relative_to(config['runtime'])).resolve(strict=True)
    if not root.is_relative_to(runtime)or root==runtime:raise ValueError('Preparation source escaped owned Runtime')
    pins = dict(config['source_files'])
    resources = Path(__file__).resolve().parent / 'original_sources'
    rows = []
    selected = {key: (key, value) for key, value in ORIGINAL_PREPARATION.items()}
    selected.update(ORIGINAL_LAYERS)
    from .cloth_wrapper_resources import PINS,LAYERS
    selected.update({key:(key,value)for key,value in PINS.items()})
    # The original room sampler loads these only when paper is selected. They
    # are shared exact-source preparation data, not an evaluator replacement.
    from robotagentbench.simulation.composite_instance_bank.source_manifest import MANIFEST,EMBEDDED_DATA
    paper={r['target_relative']:r for r in MANIFEST['sources']
           if r['target_relative'].startswith('openend/tasks/trash_transfer/')}
    selected.update({key:(row['bundled_relative'],row['sha256'])for key,row in paper.items()})
    for relative, (resource, expected) in selected.items():
        raw = (EMBEDDED_DATA[resource]if relative in paper else LAYERS[relative]if relative in LAYERS
               else LAYER_BYTES if relative in ORIGINAL_LAYERS else (resources / resource).read_bytes())
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Packaged original preparation source changed: ' + relative)
        target = _destination(root,relative)
        if relative in pins and pins[relative] != expected:
            raise ValueError('Existing source identity conflict; never replace: ' + relative)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError('Existing source bytes differ; never replace: ' + relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        pins[relative] = expected
        rows.append(dict(relative=relative, sha256=expected,
                         provenance='original_task_preparation_byte_identity_no_scoring_replacement'))
    if pins != config['source_files']:
        config['source_files'] = pins
        temporary = config_path.with_name(config_path.name + '.source-install-new')
        with temporary.open('x') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
        temporary.replace(config_path)
    return dict(installed=rows, score_or_physics_modified=False)


def install_fridge_resources(runtime, provider, *, owned_parent):
    """Restore the previously pinned 22 original preparation assets, no rewrite.

    Provider is an explicitly supplied data directory, never a remote URL or
    model input. Asset bytes are not put under a code-license assumption.
    """
    runtime=Path(runtime).resolve(strict=True)
    owned_parent=Path(owned_parent).resolve(strict=True)
    if not runtime.is_relative_to(owned_parent):raise ValueError('Wrong owned Runtime')
    if json.loads((runtime.parent/'OWNER.json').read_text())!={'owner':'RobotAgentBench-short-instance-bank-v31'}:
        raise ValueError('Wrong owned Runtime marker')
    provider=Path(provider).resolve(strict=True)
    raw=(provider/'manifest.json').read_bytes()
    from .resource_manifest import FRIDGE_INTERIOR_MANIFEST_SHA256,FRIDGE_INTERIOR_FILES
    expected_manifest=FRIDGE_INTERIOR_MANIFEST_SHA256
    if hashlib.sha256(raw).hexdigest()!=expected_manifest:raise ValueError('Original fridge resource manifest differs')
    manifest=json.loads(raw)
    if manifest['files_sha256']!=FRIDGE_INTERIOR_FILES:raise ValueError('Original audited resource member set differs')
    config_path=runtime/'runtime.json';config=json.loads(config_path.read_text())
    root=(runtime/Path(config['source_root']).relative_to(config['runtime'])).resolve(strict=True)
    if not root.is_relative_to(runtime)or root==runtime:raise ValueError('Preparation source escaped owned Runtime')
    relative=Path('short_tasks/tasks/fridge_to_heated_box/interior_resources_v1')
    pins=dict(config['source_files'])
    rows=[]
    for name,digest in manifest['files_sha256'].items():
        name_path=Path(name)
        if name_path.is_absolute()or '..'in name_path.parts:raise ValueError('Resource escaped its original directory')
        data=(provider/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Original fridge resource changed: '+name)
        target=_destination(root,relative/name)
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Existing fridge resource differs; never replace: '+name)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb')as stream:stream.write(data)
        key=str(relative/name)
        if key in pins and pins[key]!=digest:raise ValueError('Existing resource pin differs: '+name)
        pins[key]=digest;rows.append(dict(relative=key,sha256=digest))
    if pins!=config['source_files']:
        config['source_files']=pins
        temporary=config_path.with_name(config_path.name+'.resource-install-new')
        with temporary.open('x')as stream:json.dump(config,stream,indent=2);stream.write('\n')
        temporary.replace(config_path)
    return dict(manifest_sha256=expected_manifest,files=rows,physics_or_scoring_modified=False)


def install_sofa_inputs(runtime,provider,*,owned_parent):
    """Two original initializer inputs; never replace an existing data record."""
    from .resource_manifest import SOFA_INITIAL_FILES
    runtime=Path(runtime).resolve(strict=True);owned_parent=Path(owned_parent).resolve(strict=True)
    if not runtime.is_relative_to(owned_parent):raise ValueError('Wrong owned Runtime')
    if json.loads((runtime.parent/'OWNER.json').read_text())!={'owner':'RobotAgentBench-short-instance-bank-v31'}:
        raise ValueError('Wrong owned Runtime marker')
    config_path=runtime/'runtime.json';config=json.loads(config_path.read_text())
    root=(runtime/Path(config['source_root']).relative_to(config['runtime'])).resolve(strict=True)
    if not root.is_relative_to(runtime)or root==runtime:raise ValueError('Preparation source escaped owned Runtime')
    pins=dict(config['source_files'])
    for name,expected in SOFA_INITIAL_FILES.items():
        raw=(Path(provider)/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Original sofa input differs')
        relative='short_tasks/tasks/sofa_to_drawer/'+name;target=_destination(root,relative)
        if target.exists():
            if target.read_bytes()!=raw:raise ValueError('Existing sofa input differs; never replace')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb')as stream:stream.write(raw)
        if relative in pins and pins[relative]!=expected:raise ValueError('Existing sofa input pin differs')
        pins[relative]=expected
    if pins!=config['source_files']:
        config['source_files']=pins;temporary=config_path.with_name(config_path.name+'.sofa-install-new')
        with temporary.open('x')as stream:json.dump(config,stream,indent=2);stream.write('\n')
        temporary.replace(config_path)
    return dict(files=SOFA_INITIAL_FILES,physics_or_scoring_modified=False)
