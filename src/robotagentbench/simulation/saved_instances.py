"""CPU-only catalogue of immutable saved task instances.

No generator, USD/Isaac, model, action or scorer is imported. Bindings locate
unchanged files; they do not rewrite paths inside launch/USD/source manifests.
Only selection['model_input'] may be passed to a model, never the operator part.

Source-plan contract::

    {"schema": "robotagentbench.saved_instances.sources.v1",
     "sources": [{"kind": "composite", "index": "/download/index.json",
                  "sha256": "<exact completed-index SHA256>", "public_inputs": []}],
     "readonly_bindings": [{"logical_root": "/original/bank",
                             "backing_root": "/download/bank"}]}

Optional public_inputs rows are exact task_id/episode/path/sha256/launch_sha256
bindings to separately saved PUBLIC instruction+context (or v4 prompt+context).
They are never inferred from request targets, answers, manifests or score plans.
Missing saved public inputs are explicit pending-worker-public-instruction, not
an invented task prompt. Source indexes must be sealed after preparation stops.

release_plan() yields immutable logical_path/backing_path/SHA/role entries and
unassigned release_relative_path fields. A packager must allocate those paths,
include the original transitive asset/source closure and establish its read-only
execution namespace. This module neither supplies private server defaults nor
claims a downloaded absolute-path launch can execute without that namespace.
"""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

SCHEMA='robotagentbench.saved_instances.sources.v1'
SELECTION='robotagentbench.saved_instances.selection.v1'
RELEASE='robotagentbench.saved_instances.release_plan.v1'
INDEX_SCHEMAS={
    'short':{'robotagentbench.short_instance_bank.v1','robotagentbench.short_instance_catalog.v31'},
    'cloth':{'robotagentbench.cloth_instance_bank.v1'},
    'composite':{'robotagentbench.composite_instance_index.v31'},
    'diagnostic':{'robotagentbench.diagnostic_instance_bank.v1'}}
PREPARED={'short':{'prepared_cpu_not_native_accepted'},'cloth':{'cpu_geometry_prepared'},
    'composite':{'actual_scene_cpu_admitted'},'diagnostic':{'prepared_cpu_not_native'}}


def read(path):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    def nonfinite(value):raise ValueError('Nonfinite JSON is not allowed')
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=nonfinite)


def digest(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value,str)or len(value)!=64 or any(c not in '0123456789abcdef'for c in value):
        raise ValueError('Explicit lowercase SHA256 required')
    return value


def template():
    """Caller pins completed index snapshots; no private deployment is implicit."""
    return dict(schema=SCHEMA,sources=[dict(kind='short',index='/download/bank/index.json',
        sha256='REPLACE_WITH_INDEX_SHA256',public_inputs=[])],readonly_bindings=[])


class _Reader:
    def __init__(self,plan):
        if not isinstance(plan,dict)or set(plan)!={'schema','sources','readonly_bindings'}or plan['schema']!=SCHEMA:
            raise ValueError('Exact saved-instance source plan required')
        self.plan=plan;self.cache={};self.bindings=[]
        for row in plan['readonly_bindings']:
            if set(row)!={'logical_root','backing_root'}:raise ValueError('Exact read-only locator binding required')
            logical=Path(row['logical_root']);backing=Path(row['backing_root'])
            if not logical.is_absolute()or '..'in logical.parts or not backing.is_absolute()or backing.resolve()!=backing or not backing.exists():
                raise ValueError('Canonical logical root and actual local backing required')
            if any(logical==old for old,_ in self.bindings):raise ValueError('Duplicate logical namespace binding')
            self.bindings.append((logical,backing))

    def locate(self,path):
        original=Path(path)
        if not original.is_absolute()or '..'in original.parts:raise ValueError('Absolute immutable path required')
        matches=[(a,b)for a,b in self.bindings if original==a or original.is_relative_to(a)]
        actual=original
        if matches:
            a,b=max(matches,key=lambda pair:len(pair[0].parts));actual=b/original.relative_to(a)
        if actual.resolve(strict=True)!=actual or not actual.is_file():raise ValueError('Missing/symbolic saved file: '+str(original))
        return actual

    def checked(self,path,expected):
        expected=_sha(expected);actual=self.locate(path);key=(str(actual),expected)
        if key not in self.cache:
            if digest(actual)!=expected:raise ValueError('Saved artifact changed: '+str(path))
            self.cache[key]=True
        return actual

    def relative(self,root,path):
        value=Path(path)
        if value.is_absolute():return str(value)
        if '..'in value.parts or value.as_posix()!=path:raise ValueError('Canonical bank-relative path required')
        return str(Path(root)/value)

    def rows(self):
        candidates={}
        for source in self.plan['sources']:
            if set(source)!={'kind','index','sha256','public_inputs'}or source['kind']not in INDEX_SCHEMAS:
                raise ValueError('Unknown saved-bank source or fields')
            kind=source['kind'];index_path=self.checked(source['index'],source['sha256']);index=read(index_path)
            if index.get('schema')not in INDEX_SCHEMAS[kind]:raise ValueError('Wrong original bank schema')
            root=str(Path(source['index']).parent)
            entries=index['entries']if kind=='diagnostic'else index['rows']if kind=='composite'else index['instances']
            for row in entries:
                if row.get('status')not in PREPARED[kind]or row.get('selected_for_bank')is False:continue
                task=index['task_name']if kind=='cloth'else row['task_name']
                scenario=row.get('scenario');task_id=task+('--'+scenario if task=='task_discovery'and scenario else '')
                seed=row.get('actual_seed',row.get('seed'))
                if type(seed)is not int or seed<1:raise ValueError('Original actual seed must be explicit')
                key=(task_id,seed);item=dict(kind=kind,source=source,root=root,row=row,task_name=task,
                    task_id=task_id,scenario=scenario,actual_seed=seed)
                if key in candidates:
                    old=candidates[key]
                    if (old['kind'],old['row'].get('launch'),old['row'].get('launch_sha256',old['row'].get('hash')))!=(kind,row.get('launch'),row.get('launch_sha256',row.get('hash'))):
                        raise ValueError('Conflicting successful task/seed records')
                else:candidates[key]=item
        groups={}
        for item in candidates.values():groups.setdefault(item['task_id'],[]).append(item)
        for task,items in groups.items():
            explicit=[x['row'].get('episode_ordinal')for x in items]
            if any(n is not None for n in explicit):
                if any(type(n)is not int or not 1<=n<=30 for n in explicit)or len(set(explicit))!=len(explicit):
                    raise ValueError('Ambiguous saved episode ordinals')
                items.sort(key=lambda x:x['row']['episode_ordinal'])
            else:items.sort(key=lambda x:x['actual_seed'])
            for ordinal,item in enumerate(items,1):item['episode']=item['row'].get('episode_ordinal',ordinal)
            if len(items)>30:raise ValueError('Use a selected catalogue of at most30 episodes, not raw surplus attempts')
        return groups

    def artifacts(self,item):
        """Verify output seals and return numeric evidence, never a private plan."""
        row=item['row'];kind=item['kind'];root=item['root'];files={}
        def check(path,expected,role='artifact'):
            path=self.relative(root,path);actual=self.checked(path,expected)
            files[path]=dict(path=str(actual),sha256=expected,role=role)
            return actual
        model=None;expected_runtime_task=item['task_name']
        if kind=='short':
            for path,expected in row['output_hashes'].items():check(path,expected)
            launch=self.relative(root,row['launch']);launch_sha=row['launch_sha256'];check(launch,launch_sha,'launch')
            physical=row['physical_configuration'];signature=fingerprint({k:v for k,v in physical.items()if k!='layout_id'})
            if not physical.get('objects'):raise ValueError('No actual saved object-state evidence')
            if row.get('physical_fingerprint',fingerprint(physical))!=fingerprint(physical):raise ValueError('Saved physical fingerprint changed')
            if row.get('semantic_fingerprint',signature)!=signature:raise ValueError('Saved semantic fingerprint changed')
        elif kind=='composite':
            instance=read(check(row['instance'],row['instance_hash'],'instance'))
            if instance.get('schema')!='robotagentbench.composite_instance.v31'or instance['task_name']!=item['task_name']or instance['seed']!=item['actual_seed']or instance['status']!='actual_scene_cpu_admitted':
                raise ValueError('Composite instance identity/status mismatch')
            for name,data in instance['files'].items():check(data['path'],data['sha256'],name)
            launch=instance['launch'];launch_sha=instance['files']['launch']['sha256']
            if row.get('launch',launch)!=launch or row.get('hash',launch_sha)!=launch_sha:
                raise ValueError('Composite index and launch identity differ')
            expected_runtime_task=instance['actual_runtime_task']
            manifest=read(self.locate(instance['files']['manifest']['path']));payload=read(self.locate(launch))
            keys=('id','path','visible_bounds_world_m','polygon','placement','support','position_m','matrix','world_transform')
            state=dict(layout_id=manifest['layout_id'],initial_base_xyyaw=payload['initial_base_xyyaw'],
                objects=[{key:obj[key]for key in keys if key in obj}for obj in manifest.get('objects',[])])
            if not state['objects']:raise ValueError('No composed object geometry')
            original=fingerprint(state)
            if original!=instance['geometry_sha256']or original!=row['geometry_sha256']:raise ValueError('Composed numeric geometry fingerprint changed')
            signature=fingerprint({k:v for k,v in state.items()if k!='layout_id'})
            if 'public_input.json'in instance['files']:
                source=instance['files']['public_input.json'];model=self.public(read(self.locate(source['path'])),source['path'],source['sha256'])
        elif kind=='diagnostic':
            record=read(check(row['record'],row['record_sha256'],'instance'))
            if record.get('schema')!='robotagentbench.diagnostic_instance.v1'or record['task_name']!=item['task_name']or record['seed']!=item['actual_seed']or record.get('scenario')!=item['scenario']or record['status']!='prepared_cpu_not_native':
                raise ValueError('Diagnostic record identity/status mismatch')
            for path,expected in record['files_sha256'].items():check(path,expected)
            launch=self.relative(root,record['launch']);launch_sha=row['hash'];check(launch,launch_sha,'launch')
            if row.get('launch')is not None and self.relative(root,row['launch'])!=launch:
                raise ValueError('Diagnostic index and launch identity differ')
            if record.get('original_launch_sha256',launch_sha)!=launch_sha:
                raise ValueError('Diagnostic original launch seal differs')
            variations=[path for path in record['files_sha256']if Path(path).name=='variation.private.json']
            if len(variations)!=1:raise ValueError('One sealed actual variation record is required')
            variation=read(self.locate(self.relative(root,variations[0])))
            expected=hashlib.sha256(json.dumps(variation,sort_keys=True).encode()).hexdigest()
            if expected!=record['variation_sha256']or expected!=row['variation_sha256']:raise ValueError('Actual variation evidence changed')
            if not variation.get('rigid_body_world_transforms'):raise ValueError('No actual rigid-body variation inventory')
            signature=fingerprint(variation)
            public=record.get('model_input')
            if public:
                if public not in record['files_sha256']:raise ValueError('Public input must be sealed with this instance')
                path=self.relative(root,public);model=self.public(read(self.locate(path)),path,record['files_sha256'][public])
        else:
            launch=self.relative(root,row['launch']);launch_sha=row['launch_sha256'];envelope=read(check(launch,launch_sha,'launch'))
            if envelope.get('task_name')!=item['task_name']:raise ValueError('Cloth task changed')
            variation=envelope['variation'];record=read(check(variation['record'],variation['record_sha256'],'variation'))
            initial=check(variation['initial'],variation['initial_sha256'],'physical_initial')
            if record['task_name']!=item['task_name']or record['seed']!=item['actual_seed']or record['initial_sha256']!=row['initial_sha256']or record['initial_sha256']!=variation['initial_sha256']or record['geometry'].get('accepted')is not True:
                raise ValueError('Cloth variation identity/geometry changed')
            signature=_cloth_positions(initial)
        payload=read(self.locate(launch))
        if payload.get('task_name')!=expected_runtime_task:
            raise ValueError('Saved launch task binding differs')
        overrides=[p for p in item['source']['public_inputs']if p['task_id']==item['task_id']and p['episode']==item['episode']]
        if len(overrides)>1:raise ValueError('Ambiguous public input binding')
        if overrides:
            value=overrides[0]
            if set(value)!={'task_id','episode','path','sha256','launch_sha256'}or value['launch_sha256']!=launch_sha:
                raise ValueError('Public input must explicitly bind this immutable launch SHA')
            model=self.public(read(check(value['path'],value['sha256'],'public_input')),value['path'],value['sha256'])
        return dict(launch=launch,launch_backing=str(self.locate(launch)),launch_sha256=launch_sha,
            signature=signature,files=files,model=model)

    @staticmethod
    def public(value,path,sha):
        # v4 scenario inputs use prompt/context; diagnostic uses instruction/context.
        if set(value)=={'prompt','context'}:value=dict(instruction=value['prompt'],context=value['context'])
        if set(value)!={'instruction','context'}or not all(isinstance(v,str)for v in value.values())or not value['instruction']:
            raise ValueError('Only explicit saved public instruction/context may reach a model')
        return dict(value=value,path=str(path),sha256=sha)


def _cloth_positions(path):
    """Hash numeric material coordinates, not NPZ timestamps/filenames/seeds."""
    with zipfile.ZipFile(path)as archive:
        raw=archive.read('initial_cloth_q.npy')
    stream=io.BytesIO(raw)
    if stream.read(6)!=b'\x93NUMPY':raise ValueError('Original numeric cloth state required')
    version=stream.read(2)
    if version==b'\x01\x00':size=struct.unpack('<H',stream.read(2))[0]
    elif version in (b'\x02\x00',b'\x03\x00'):size=struct.unpack('<I',stream.read(4))[0]
    else:raise ValueError('Unsupported NPY version')
    header=ast.literal_eval(stream.read(size).decode('utf-8'));shape=header['shape'];dtype=header['descr']
    if len(shape)!=2 or shape[0]<1 or shape[1]!=3 or header['fortran_order']or dtype not in ('<f8','<f4'):
        raise ValueError('Finite C-order Nx3 native material positions required')
    payload=stream.read();width=8 if dtype=='<f8'else 4
    if len(payload)!=shape[0]*3*width or any(not math.isfinite(v[0])for v in struct.iter_unpack('<d'if width==8 else'<f',payload)):
        raise ValueError('Invalid native material coordinates')
    return fingerprint(dict(shape=shape,dtype=dtype,numeric_bytes_sha256=hashlib.sha256(payload).hexdigest()))


def _reader(plan):return _Reader(read(plan)if isinstance(plan,(str,Path))else plan)


def list_tasks(plan):
    """Index pins checked; counts are not full output or native verification."""
    reader=_reader(plan);groups=reader.rows()
    return [dict(task_id=task,episodes=[r['episode']for r in rows],saved_count=len(rows),
        complete30=len(rows)==30,kind=rows[0]['kind'],output_hashes_verified=False)
        for task,rows in sorted(groups.items())]


def verify(plan,task_id=None):
    reader=_reader(plan);groups=reader.rows();results=[]
    if task_id is not None and task_id not in groups:raise ValueError('Unknown saved task ID')
    for task,rows in sorted(groups.items()):
        if task_id is not None and task!=task_id:continue
        signatures=set();public=0
        for item in rows:
            data=reader.artifacts(item)
            if data['signature']in signatures:raise ValueError('Duplicate actual saved configurations: '+task)
            signatures.add(data['signature']);public+=data['model']is not None
        results.append(dict(task_id=task,verified=len(rows),distinct=len(signatures),complete30=len(rows)==30,
            public_inputs_saved=public,evidence='sealed_numeric_initial_state_not_new_USD_or_native_validation'))
    return dict(schema='robotagentbench.saved_instances.verification.v1',tasks=results,
        resampled=False,native_started=False,dependency_relocation_verified=False)


def select(plan,task_id,episode):
    if type(episode)is not int or not 1<=episode<=30:raise ValueError('Episode ordinal1..30 required')
    reader=_reader(plan);groups=reader.rows();rows=groups.get(task_id,[])
    selected=[r for r in rows if r['episode']==episode]
    if len(selected)!=1:raise ValueError('No unambiguous saved instance for this task/episode')
    signatures=set();data=None
    for item in rows:
        candidate=reader.artifacts(item)
        if candidate['signature']in signatures:raise ValueError('Duplicate actual saved configurations: '+task_id)
        signatures.add(candidate['signature'])
        if item is selected[0]:data=candidate
    item=selected[0];model=data['model']
    return dict(schema=SELECTION,task_id=task_id,episode=episode,actual_seed=item['actual_seed'],
        status='saved_cpu_prepared_not_native_accepted',
        operator=dict(task_name=item['task_name'],scenario=item['scenario'],launch=data['launch'],
            launch_backing=data['launch_backing'],launch_sha256=data['launch_sha256'],
            readonly_bindings=reader.plan['readonly_bindings'],execution_namespace_required=True,
            dependency_relocation_verified=False),
        model_input=model['value']if model else None,
        model_input_binding=dict(launch_sha256=data['launch_sha256'],source=model['path'],sha256=model['sha256'],
            status='saved_public_input')if model else dict(launch_sha256=data['launch_sha256'],
            status='pending_worker_public_instruction',private_plan_fallback=False),
        distinct_numeric_configurations=len(signatures),complete30=len(rows)==30,resampled=False,native_started=False)


def release_plan(plan):
    """Describe immutable files for a release packager; never copy or relocate."""
    reader=_reader(plan);groups=reader.rows();files={}
    for source in reader.plan['sources']:
        files[source['index']]=dict(path=str(reader.checked(source['index'],source['sha256'])),
            sha256=source['sha256'],role='index')
    selections=[]
    for task,rows in sorted(groups.items()):
        for item in rows:
            data=reader.artifacts(item);files.update(data['files'])
            selections.append(dict(task_id=task,episode=item['episode'],actual_seed=item['actual_seed'],
                launch=data['launch'],launch_sha256=data['launch_sha256'],
                public_input=data['model']['path']if data['model']else None))
    return dict(schema=RELEASE,immutable_files=[dict(logical_path=logical,backing_path=row['path'],
        sha256=row['sha256'],role=row['role'],release_relative_path=None)for logical,row in sorted(files.items())],
        selections=selections,launch_bytes_must_remain_unchanged=True,
        required_execution='original_readonly_namespace_with_logical_to_release_backing_bindings',
        transitive_asset_closure='packager_must_supply_and_verify_original_USD_source_asset_dependencies',
        relocation_complete=False,copied_files=False,native_started=False)
