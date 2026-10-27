"""Operator-only v18 input admission; no simulator imports or score implementation.

The existing outer launch envelope is retained for NativeClothBinding routing.
The inner schema intentionally distinguishes assets, immutable original code,
derived collision geometry, prepared state, owner code and Python geometry libs.
"""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
OUTER='robotagentbench.native_cloth.launch.v1'
INNER='robotagentbench.original_cloth_owner.v1'
TASKS=frozenset(('spread_tablecloth','collect_tablecloth','long_task_18'))
DIRS=frozenset(('asset_root','original_root','collision_geometry_root','prepared_root',
    'owner_root','formal_root','public_root','geometry_python_root','sdk_site'))
FILES=frozenset(('initial','setting','policy','protocol','trial','source_manifest',
    'asset_manifest','geometry_receipt','geometry','reference_action','reference_state'))
FORMAL_FILES=('__init__.py','config.py','binding.py','launcher.py','supervisor.py','physical.py','bootstrap.py',
    'randomization.py','variation_prepare.py','variation_native.py')
OWNER_FILES=('source_hooks.py','source_owner.py','source_controls.py','source_rpc.py',
    'source_visual_client.py','control_session.py','approved_rules.py','composition.py',
    'facts_reader.py','native_reader.py','native_geometry.py','mesh.py','folding.py',
    'placement.py','grasp.py','contact_geometry.py','solids.py','POLICY.json',
    'SCORING_POLICY.json','RENDER_CPU_AUTHORIZATION.json','visual_only_v18/worker.py')
PUBLIC_FILES=('interfaces/robot_profiles.py','interfaces/action_contract.py',
    'runtime/action_host.py','runtime/completion.py','simulation/gpu.py','simulation/workspace.py')
ENTRY_HASHES={'native_removal_continuous_transport_v54.py':'6071a1f888b6c721d7d5a60b6da07fdeebcd020a4e2ef869e5e62dcc525297ce',
 'native_measured_mirrored_lay_v50.py':'af277146a6797013e0677251fe656e8db1e9fb14adb528580cd828c861284428'}


def module_bootstrap(name):
    """Fixed package child under isolated Python; no caller-selected module."""
    if name not in ('supervisor','physical'):
        raise ValueError('Only the two trusted cloth process boundaries')
    module='robotagentbench.simulation.cloth.'+name
    return ('import runpy,sys;sys.dont_write_bytecode=True;'
        'sys.path.insert(0,'+repr(str(HERE.parents[2]))+');'
        'sys.argv=['+repr(module)+',*sys.argv[1:]];'
        'runpy.run_module('+repr(module)+',run_name="__main__")')


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate configuration key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique)


def selection(task):
    if task not in TASKS:raise ValueError('Exact cloth task required')
    spread=task=='spread_tablecloth'
    return dict(episode='spread_v174'if spread else'remove_v179',
        initial='spread_tablecloth.initial.npz'if spread else'collect_tablecloth.initial.npz',
        entry='native_measured_mirrored_lay_v50.py'if spread else'native_removal_continuous_transport_v54.py',
        reference='reference_clean_local_geometry'if spread else'reference_removal_initial_geometry')


def native_directory(owner):
    return Path(owner['directories']['original_root'])/'code/tablecloth_workflow/native_teleop/lay_20260924'


def alias_recipe(owner):
    roots=owner['directories'];asset=Path(roots['asset_root']);native=native_directory(owner)
    result=[dict(path=str(native/'local_scene_collision_geometry.npz'),target=str(Path(roots['collision_geometry_root'])/'local_scene_collision_geometry.npz'))]
    for reference in ('reference_removal_initial_geometry','reference_clean_local_geometry'):
        for name in ('full_state.npz','action.npz'):
            result.append(dict(path=str(native/reference/name),target=str(asset/'assets/task8_initialization'/reference/name)))
    repo=Path(roots['original_root'])/'code/agent_bench'
    for name in ('robot_native_collision.json','robot_native_collision.npz'):
        result.append(dict(path=str(repo/'.codex-tmp/tablecloth-action-loop-20260905'/name),target=str(asset/'assets/task8_collision_exports'/name)))
    result.append(dict(path=str(repo/'arena_scene'),target=str(asset/'assets/alienware/arena_scene')))
    return result


def compatibility_recipe(owner):
    source=Path(owner['directories']['original_root'])/'code'
    return [dict(source=str(source/'agent_bench'),target='/home/aether/workspace/agent_bench'),
        dict(source=str(source/'tablecloth_workflow'),target='/media/aether/OS/agent_bench_runs/tablecloth_gripper_actuator_20260922')]


def _canonical(path,workspace):
    return workspace.input(path)


def _require(pins,path):
    if str(path)not in pins:raise ValueError('Required cloth input is not pinned: '+str(path))


def _relative(value):
    p=Path(value)
    if p.is_absolute()or '..'in p.parts or p.as_posix()!=value:raise ValueError('Canonical relative source-manifest path required')
    return p


def source_target(owner,path):
    """Only the original explicit alias recipe can map a noncanonical leaf."""
    path=Path(path)
    candidates=[row for row in alias_recipe(owner)if path.is_relative_to(row['path'])]
    if not candidates:return path
    row=max(candidates,key=lambda item:len(Path(item['path']).parts))
    return Path(row['target'])/path.relative_to(row['path'])


def validate(value,workspace,*,mounted=True):
    if (not isinstance(value,dict)or set(value)not in ({'schema','task_name','execution_mode','owner'},
                                                     {'schema','task_name','execution_mode','owner','variation'})
        or value['schema']!=OUTER or value['task_name']not in TASKS
        or value['execution_mode']!='native_cloth_scored.v1'):
        raise ValueError('Exact scored cloth envelope required')
    owner=value['owner']
    if (not isinstance(owner,dict)or set(owner)!={'schema','source_episode','directories','files','hashes','aliases','compatibility_mounts'}
        or owner['schema']!=INNER or set(owner['directories'])!=DIRS or set(owner['files'])!=FILES):
        raise ValueError('Explicit original-source cloth schema required; no fallback')
    selected=selection(value['task_name']);roots={k:Path(v)for k,v in owner['directories'].items()}
    if owner['source_episode']!=selected['episode']:raise ValueError('Wrong frozen source episode')
    if owner['aliases']!=alias_recipe(owner)or owner['compatibility_mounts']!=compatibility_recipe(owner):raise ValueError('Only the audited source alias recipe is permitted')
    for path in roots.values():
        if (not path.is_absolute()or path.resolve()!=path or not path.is_dir()
            or not any(path==r or path.is_relative_to(r)for r in workspace.input_roots)):
            raise ValueError('Canonical operator-bound input directories required')
    if roots['formal_root']!=HERE:raise ValueError('Only the installed trusted cloth entry is allowed')
    pins=owner['hashes']
    if not isinstance(pins,dict)or not pins:raise ValueError('Explicit original hashes required')
    for path,digest in pins.items():
        if not isinstance(digest,str)or not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Exact SHA256 required')
        if sha(_canonical(path,workspace))!=digest:raise ValueError('Pinned original input changed: '+path)
    for path in owner['files'].values():_require(pins,_canonical(path,workspace))
    files={k:Path(v)for k,v in owner['files'].items()}
    expected={'trial':roots['asset_root']/'reports'/selected['episode']/'trial.json',
        'geometry':roots['collision_geometry_root']/'local_scene_collision_geometry.npz',
        'initial':roots['prepared_root']/selected['initial'],
        'reference_action':roots['asset_root']/'assets/task8_initialization'/selected['reference']/'action.npz',
        'reference_state':roots['asset_root']/'assets/task8_initialization'/selected['reference']/'full_state.npz'}
    if any(files[k]!=path for k,path in expected.items()):raise ValueError('Frozen namespace/initial/reference binding changed')
    entry=native_directory(owner)/selected['entry'];_require(pins,entry)
    if pins[str(entry)]!=ENTRY_HASHES[selected['entry']]:raise ValueError('Original source entry identity changed')
    for base,names in ((roots['formal_root'],FORMAL_FILES),(roots['owner_root'],OWNER_FILES),
        (roots['public_root']/'robotagentbench',PUBLIC_FILES)):
        for name in names:_require(pins,base/name)
    for path in (roots['public_root']/'robotagentbench').rglob('*.py'):_require(pins,path)
    if any((roots['public_root']/'robotagentbench').rglob('*.pyc')):raise ValueError('Use a clean pinned public source view, not cached bytecode')
    if files['policy']!=roots['owner_root']/'POLICY.json':raise ValueError('Use the unchanged approved owner policy')
    source_protocol=roots['original_root']/'code/tablecloth_workflow/protocol_v14/config.json'
    _require(pins,source_protocol)
    if pins[str(source_protocol)]!=pins[str(files['protocol'])]:raise ValueError('Original protocol copy differs')
    manifest=read(files['source_manifest'])['files']
    for relative,row in manifest.items():
        _relative(relative)
        if not relative.startswith('code/'):continue
        path=source_target(owner,roots['original_root']/relative);_require(pins,path)
        if pins[str(path)]!=row['sha256']:raise ValueError('Original source manifest differs')
    for relative,row in read(files['asset_manifest'])['files'].items():
        _relative(relative)
        path=roots['asset_root']/relative;_require(pins,path)
        if pins[str(path)]!=row['sha256']:raise ValueError('Original asset manifest differs')
    for relative in ('assets/alienware/arena_scene/v1_2/demo_sim/scene_room_baseline.usda',
        'assets/alienware/arena_scene/v1_2/source/geniesim/assets/robot/G1_omnipicker/robot.usda',
        'code/tablecloth_workflow/native_teleop/lay_20260924/basket_profile_split_colliders.json'):
        _require(pins,roots['asset_root']/relative)
    if read(files['geometry_receipt'])['geometry_sha256']!=pins[str(files['geometry'])]:raise ValueError('Original derived geometry receipt changed')
    for row in owner['aliases']:
        target=Path(row['target']);path=Path(row['path'])
        if target.is_file():_require(pins,target)
        if not target.exists():raise ValueError('Original alias target missing')
        if mounted and(not path.is_symlink()or path.resolve()!=target.resolve()):raise ValueError('Original read-only alias missing or retargeted')
    if mounted:
        for row in owner['compatibility_mounts']:
            source,target=Path(row['source']),Path(row['target'])
            if not target.is_dir()or target.resolve()!=target:raise ValueError('Original compatibility bind mount absent')
            for relative,record in manifest.items():
                p=roots['original_root']/relative
                if p.is_relative_to(source)and sha(target/p.relative_to(source))!=record['sha256']:raise ValueError('Original compatibility view changed')
            for alias in owner['aliases']:
                p=Path(alias['path'])
                if p.is_relative_to(source):
                    view=target/p.relative_to(source)
                    if not view.is_symlink()or view.resolve()!=Path(alias['target']).resolve():raise ValueError('Original compatibility alias retargeted')
    if 'variation' in value:
        from . import randomization
        record,state=randomization.load_variation(value['variation'],workspace)
        if record['task_name']!=value['task_name'] or record['baseline_sha256']!=pins[str(files['initial'])]:
            raise ValueError('Variation belongs to a different task or frozen initial state')
        # This boundary also runs under the isolated, SDK-free supervisor.
        # Exact array/rigid-reset checks remain mandatory in NativeReset before
        # SimulationApp is imported, using the pinned native NumPy environment.
        randomization.verify_scene_choices(record)
        production=roots['asset_root']/'code/agent_bench/real2sim/production'
        if record['layout']['catalog']!=str(production/'house_layouts.json'):
            raise ValueError('Use the original asset-owned layout catalog')
        if record['appearance']['source']!=str(roots['original_root']/'code/agent_bench/real2sim/production/livingroom_environment_randomization.py'):
            raise ValueError('Use the pinned original room colour definitions')
    return value


def prepare(path,value,*,workspace):
    """Fresh CPU preparation only; actual mounted admission remains mandatory."""
    validate(value,workspace,mounted=False)
    path=Path(path)
    if path.resolve()!=path or not any(path.is_relative_to(r)for r in workspace.input_roots):raise ValueError('Prepared launch must remain in trusted input roots')
    with path.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return path
