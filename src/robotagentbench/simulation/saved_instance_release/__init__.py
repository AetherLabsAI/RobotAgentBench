"""Offline, content-addressed saved-instance packaging; no task execution.

Trusted packagers supply exact sealed logical paths, source bindings and SHA256
identities. This module never generates a task, rewrites an original input,
imports USD/Isaac, downloads assets, or supplies private evaluator data to a
model. Existing asset-store objects are referenced, not bundled again.
"""
import hashlib,json,os,shutil
from pathlib import Path,PurePosixPath

SCHEMA='robotagentbench.saved_instance_release.v1'
ROLES={'launch','request','manifest','scene','ready_pose','sensor','initial','variation',
       'dependency','public_input','catalog','provenance','source','asset','material','instance'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def _digest(value):
    if not isinstance(value,str)or len(value)!=64 or any(c not in'0123456789abcdef'for c in value):
        raise ValueError('Exact lowercase SHA256 required')
    return value

def _relative(value):
    p=PurePosixPath(value)
    if not isinstance(value,str)or not value or p.is_absolute()or '..'in p.parts or p.as_posix()!=value or'\\'in value:
        raise ValueError('Canonical relative release path required')
    return value

def _logical(value):
    p=PurePosixPath(value)
    if not isinstance(value,str)or not p.is_absolute()or'..'in p.parts or p.as_posix()!=value or value=='/'or'\\'in value:
        raise ValueError('Original canonical absolute logical file identity required')
    return value

def _file(value):
    p=Path(value)
    if not p.is_absolute()or p.resolve()!=p or not p.is_file():raise ValueError('Existing canonical declared backing file required')
    return p

def _write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:stream.write(data)

def _json(path,value):_write(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())

def asset_objects(manifest):
    """Read only the pinned public asset archive member index, not all assets."""
    from robotagentbench.resources import read_manifest
    p=_file(manifest);value=read_manifest(p);objects={}
    for archive in value['archives'].values():
        for relative,row in archive['members'].items():
            if relative=='objects/'+row['sha256'][:2]+'/'+row['sha256']:
                old=objects.setdefault(row['sha256'],dict(path=relative,bytes=row['bytes']))
                if old!=dict(path=relative,bytes=row['bytes']):raise ValueError('Conflicting asset object identity')
    return value,objects,sha(p)

class Builder:
    """One fresh task-owned bundle, populated only from trusted sealed records."""
    def __init__(self,destination,*,asset_manifest,package_sources=None,sdk_sources=None):
        self.root=Path(destination)
        if not self.root.is_absolute()or self.root.resolve()!=self.root or self.root.exists()or not self.root.parent.is_dir():
            raise ValueError('Fresh canonical task-owned bundle destination required')
        release,self.assets,manifest_sha=asset_objects(asset_manifest)
        self.package=package_sources or {};self.sdk=sdk_sources or {}
        self.root.mkdir(mode=0o750);self.nodes={};self.objects={};self.instances=[];self.cache={};self.groups={}
        self.asset_release=dict(version=release['version'],hosting=release['hosting'],manifest_sha256=manifest_sha)
        _json(self.root/'INTENT.json',dict(schema=SCHEMA,purpose='immutable saved inputs only',native_started=False,model_started=False))

    def add(self,logical,backing,expected,role='dependency',*,anchor=None):
        logical=_logical(logical);expected=_digest(expected)
        if role not in ROLES:raise ValueError('Unreviewed release file role')
        name=PurePosixPath(logical)
        if (name.suffix.lower()in('.log','.stderr','.stdout','.jsonl')
            or name.name in('auth.json','credentials.json','.netrc','.env')
            or any(part in('.ssh','.aws','.codex')for part in name.parts)):
            raise ValueError('Operator/model/dispatch logs are not release inputs')
        path=_file(backing);size=path.stat().st_size
        key=(str(path),expected)
        if key not in self.cache:
            if sha(path)!=expected:raise ValueError('Sealed source/input changed: '+logical)
            self.cache[key]=True
        node=hashlib.sha256((logical+'\0'+expected).encode()).hexdigest()
        if node in self.nodes:
            record=self.nodes[node]
            if role not in record['roles']:record['roles'].append(role)
            return node
        if expected in self.assets:
            item=self.assets[expected]
            if item['bytes']!=size:raise ValueError('Asset content identity/size differs')
            storage=dict(kind='assets',relative=item['path'])
        elif expected in self.package:
            relative=_relative(self.package[expected]);storage=dict(kind='package',relative=relative)
        elif expected in self.sdk:
            relative=_relative(self.sdk[expected]);storage=dict(kind='sdk',relative=relative)
        else:
            relative='objects/'+expected[:2]+'/'+expected
            if expected not in self.objects:
                target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
                if sha(target)!=expected:raise ValueError('Copied immutable input changed')
                self.objects[expected]=dict(relative=relative,bytes=size,sha256=expected)
            storage=dict(kind='bundle',relative=relative)
        self.nodes[node]=dict(logical_path=logical,sha256=expected,bytes=size,roles=[role],storage=storage,
            source_anchor=anchor)
        return node

    def group(self,name,nodes):
        chosen=list(dict.fromkeys(nodes))
        if not isinstance(name,str)or not name or name in self.groups or any(n not in self.nodes for n in chosen):
            raise ValueError('Unique named declared source/resource group required')
        self.groups[name]=chosen

    def instance(self,task_id,episode,actual_seed,launch_node,nodes,*,provenance,public_input_node=None,groups=()):
        if not isinstance(task_id,str)or not task_id or type(episode)is not int or not 1<=episode<=30 or type(actual_seed)is not int or actual_seed<1:
            raise ValueError('Explicit original task/episode/actual seed required')
        if any(v['task_id']==task_id and v['episode']==episode for v in self.instances):raise ValueError('Duplicate selected episode')
        own=list(dict.fromkeys([launch_node,*nodes,*([public_input_node]if public_input_node else[])]))
        if len(groups)!=len(set(groups))or any(g not in self.groups for g in groups):raise ValueError('Explicit declared closure groups required')
        chosen=list(dict.fromkeys([*own,*(n for g in groups for n in self.groups[g])]))
        if any(n not in self.nodes for n in chosen):raise ValueError('Undeclared closure node')
        identities={}
        for n in chosen:
            row=self.nodes[n];logical=row['logical_path']
            if logical in identities and identities[logical]!=row['sha256']:raise ValueError('Conflicting versions in one execution namespace')
            identities[logical]=row['sha256']
        if 'launch'not in self.nodes[launch_node]['roles']:raise ValueError('Explicit launch node required')
        if public_input_node and'public_input'not in self.nodes[public_input_node]['roles']:raise ValueError('Explicit saved public input required')
        self.instances.append(dict(task_id=task_id,episode=episode,actual_seed=actual_seed,launch=launch_node,
            nodes=own,groups=list(groups),public_input=public_input_node,provenance=provenance,
            status='saved_cpu_prepared_not_native_accepted',native_validated=False,model_called=False))

    def finish(self,*,closure_evidence,remaining=None):
        if not self.instances:raise ValueError('No real selected saved instances')
        if not closure_evidence:raise ValueError('Actual transitive closure evidence required')
        value=dict(schema=SCHEMA,asset_release=self.asset_release,nodes=self.nodes,objects=self.objects,groups=self.groups,
            instances=self.instances,closure_evidence=closure_evidence,remaining=remaining or [],
            original_input_bytes_unchanged=True,native_started=False,model_started=False)
        _json(self.root/'release.json',value)
        return value

def read_release(path):
    value=json.loads(_file(path).read_text())
    if value.get('schema')!=SCHEMA:raise ValueError('Saved release schema differs')
    for key,row in value['nodes'].items():
        expected=hashlib.sha256((_logical(row['logical_path'])+'\0'+_digest(row['sha256'])).encode()).hexdigest()
        if key!=expected or type(row['bytes'])is not int or row['bytes']<0:raise ValueError('Corrupt release node identity')
        if row['storage']['kind']not in('bundle','assets','package','sdk'):raise ValueError('Unknown release storage')
        _relative(row['storage']['relative'])
    return value

def materialize(release_path,destination,task_id,episode,*,assets,package,sdk=None):
    """One saved instance's offline read-only namespace, with unchanged bytes.

    SDK/source packages must already be installed. No fetch, sampling, worker,
    USD import or execution occurs. All files are copied into a fresh tree so
    consumers cannot mutate the original asset cache through a hard link.
    """
    release_path=_file(release_path);release=read_release(release_path)
    chosen=[v for v in release['instances']if v['task_id']==task_id and v['episode']==episode]
    if len(chosen)!=1:raise ValueError('One exact saved task/episode required')
    chosen=chosen[0];destination=Path(destination)
    if not destination.is_absolute()or destination.resolve()!=destination or destination.exists()or not destination.parent.is_dir():
        raise ValueError('Fresh canonical dedicated namespace destination required')
    roots=dict(bundle=release_path.parent,assets=Path(assets),package=Path(package),sdk=Path(sdk)if sdk else None)
    providers={};logical={}
    selected_nodes=list(dict.fromkeys([*chosen['nodes'],*(n for g in chosen.get('groups',[])for n in release.get('groups',{})[g])]))
    verified=set()
    for key in selected_nodes:
        row=release['nodes'][key];base=roots[row['storage']['kind']]
        if base is None:raise ValueError('Original SDK must be explicitly provided')
        source=_file(base/row['storage']['relative'])
        identity=(str(source),row['sha256'])
        if source.stat().st_size!=row['bytes']or(identity not in verified and sha(source)!=row['sha256']):raise ValueError('Offline provider changed: '+row['logical_path'])
        verified.add(identity)
        if row['logical_path']in logical and logical[row['logical_path']]!=key:raise ValueError('Conflicting execution namespace')
        logical[row['logical_path']]=key;providers[key]=source
    destination.mkdir(mode=0o750);bindings={};sdk_bindings={};local_content={}
    for key,source in providers.items():
        row=release['nodes'][key];path=PurePosixPath(row['logical_path'])
        if row['storage']['kind']=='sdk':
            relative=row['storage']['relative']
            if not row['logical_path'].endswith('/'+relative):raise ValueError('Explicit original SDK logical-root identity required')
            logical_root=row['logical_path'][:-len(relative)].rstrip('/')
            if not logical_root:raise ValueError('An SDK cannot replace the whole filesystem root')
            sdk_bindings[logical_root]=str(roots['sdk'].resolve(strict=True))
            continue
        target=destination/'namespace'/str(path).lstrip('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        if row['sha256']in local_content:
            # Deduplicate aliases only inside this fresh, private namespace.
            # Never link to a mutable external asset/package cache.
            try:os.link(local_content[row['sha256']],target)
            except OSError:shutil.copyfile(source,target)
        else:
            shutil.copyfile(source,target);local_content[row['sha256']]=target
        target.chmod(0o444)
        top='/'+path.parts[1];bindings[top]=str(destination/'namespace'/path.parts[1])
    # Keep the original complete, separately installed SDK available for MDL
    # includes/native runtime dependencies; only its explicitly declared file
    # pins are evaluated here. SDK bytes are never added to the release archive.
    for logical in sdk_bindings:
        parents=[p for p in bindings if PurePosixPath(p)in PurePosixPath(logical).parents]
        if parents:
            parent=max(parents,key=lambda p:len(PurePosixPath(p).parts))
            placeholder=Path(bindings[parent])/PurePosixPath(logical).relative_to(parent)
            placeholder.mkdir(parents=True,exist_ok=True)
    bindings.update(sdk_bindings)
    if chosen.get('original_owner_kind')=='native_cloth':
        for name,logical in chosen.get('owner_directories',{}).items():
            _logical(logical)
            if name=='sdk_site':
                if roots['sdk']is None:raise ValueError('Original cloth SDK must be explicitly provided')
                bindings[logical]=str(roots['sdk'].resolve(strict=True));continue
            target=destination/'namespace'/logical.lstrip('/')
            target.mkdir(parents=True,exist_ok=True)
            top='/'+PurePosixPath(logical).parts[1]
            bindings[top]=str(destination/'namespace'/top.lstrip('/'))
        for row in chosen.get('owner_aliases',[]):
            logical,target_logical=_logical(row['path']),_logical(row['target'])
            alias=destination/'namespace'/logical.lstrip('/')
            target=destination/'namespace'/target_logical.lstrip('/')
            if not target.exists():raise ValueError('Original cloth alias target was not packaged')
            alias.parent.mkdir(parents=True,exist_ok=True)
            if alias.exists()or alias.is_symlink():
                raise ValueError('Original cloth alias unexpectedly has independent bytes')
            # Absolute logical targets are the unchanged native owner's recipe.
            # They resolve only in the explicit container namespace, including
            # both compatibility mount paths; no host file is followed/written.
            alias.symlink_to(target_logical,target_is_directory=target.is_dir())
        for row in chosen.get('owner_compatibility_mounts',[]):
            logical=_logical(row['target']);source=_logical(row['source'])
            backing=destination/'namespace'/source.lstrip('/')
            if not backing.is_dir():raise ValueError('Original compatibility source absent')
            bindings[logical]=str(backing)
    launch=release['nodes'][chosen['launch']]
    runtime_path=None
    metadata=release.get('closure_evidence',{}).get('runtime_profiles',{}).get(chosen.get('runtime_profile'))
    if metadata is not None:
        source_logical=_logical(metadata['source_root'])
        source_root=destination/'namespace'/source_logical.lstrip('/')
        if not source_root.is_dir():raise ValueError('Saved source profile was not materialized')
        runtime_path=destination/'runtime.json'
        # Runtime metadata belongs to the deployer, unlike saved task inputs.
        # Only its paths change; the complete original source pin set stays exact.
        runtime=dict(schema='robotagentbench.local_installation.v1',version=release['asset_release']['version'],
            profile=metadata['profile'],assets=str(roots['assets'].resolve(strict=True)),runtime=str(destination),
            source_root=str(source_root),source_files=metadata['source_files'],resources={},
            compatibility_namespaces=[],container_only_compatibility_paths=True,
            native_environment='Original separately installed Isaac SDK required.',parameter_overrides={})
        for relative,expected in metadata['source_files'].items():
            path=source_root/_relative(relative)
            if not path.is_file()or sha(path)!=expected:raise ValueError('Materialized runtime source differs: '+relative)
        for location,logical in metadata.get('resources',{}).items():
            path=destination/'namespace'/str(logical).lstrip('/')
            if path.is_file():runtime['resources'][location]=str(path)
        _json(runtime_path,runtime)
    model=None;public_binding=None
    if chosen['public_input']:
        record=json.loads(providers[chosen['public_input']].read_text())
        if set(record)=={'prompt','context'}:record=dict(instruction=record['prompt'],context=record['context'])
        if set(record)!={'instruction','context'}or not all(isinstance(x,str)for x in record.values()):raise ValueError('Only saved public input may reach a model')
        model=record
        public_node=release['nodes'][chosen['public_input']]
        public_binding=destination/'operator/public_input_binding.json'
        _json(public_binding,dict(schema='robotagentbench.public_input_binding.v1',
            source=public_node['logical_path'],sha256=public_node['sha256'],launch_sha256=launch['sha256']))
        public_binding.chmod(0o444)
        # Bind this small, newly derived operator record read-only too. It is
        # not part of an original launch/source file and contains no answers.
        bindings[str(destination/'operator')]=str(destination/'operator')
    result=dict(schema='robotagentbench.saved_instance_release.selection.v1',task_id=task_id,episode=episode,
        actual_seed=chosen['actual_seed'],operator=dict(launch=launch['logical_path'],launch_sha256=launch['sha256'],
            readonly_bindings=[dict(logical_root=a,backing_root=b)for a,b in sorted(bindings.items(),key=lambda item:(len(PurePosixPath(item[0]).parts),item[0]))],
            original_owner_and_source_profile_required=True,
            public_input_binding=str(public_binding)if public_binding else None,
            runtime=str(destination)if runtime_path else None,
            runtime_metadata=str(runtime_path)if runtime_path else None,
            input_roots=sorted(bindings),
            readonly_aliases=dict(sorted(bindings.items()))),model_input=model,
        status='offline_namespace_prepared_not_native_validated',native_started=False,model_started=False)
    _json(destination/'selection.json',result)
    return result
