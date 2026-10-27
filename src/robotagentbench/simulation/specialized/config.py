"""Trusted fixed profile anchors; every deployment path is external configuration.

No original owner is imported during describe/verify/dispatch preparation.
The external supervisor, not this module, creates a read-only source namespace.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDENTITIES_SHA = 'c4e14260202c8cfc3ccdab1d843046b4eaffdf050ac507c10138aabc474fc355'
SCHEMA = 'robotagentbench.specialized_operator.v1'
DISPATCH = 'robotagentbench.specialized_dispatch.v1'
BINDINGS = 'robotagentbench.specialized_bindings.v1'
MODULES = {
    'atomic_coffee': 'short_task_robustness.tasks.coffee_machine_push_interference.public_integration_v1.adapter',
    'failure_recovery': 'short_task_robustness.tasks.grasp_failure_recovery.public_adapter',
}
ENTRY_SUFFIXES = {
    'atomic_coffee': '/coffee_machine_push_interference/public_integration_v1/adapter.py',
    'failure_recovery': '/grasp_failure_recovery/public_adapter.py',
}
OUTER_CONTRACT = {
    'read_only_namespace': True, 'exclusive_gpu': True,
    'no_automatic_retry': True, 'persistent_dispatch_identity': True,
    'owned_cleanup_required': True, 'validation_watchdog_s': 600,
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def identities():
    path = HERE / 'identities.json'
    if sha(path) != IDENTITIES_SHA:
        raise ValueError('Reviewed specialized identity anchors changed; no refresh')
    return read(path)['profiles']


def read_operator(path):
    value = read(path)
    fields = {'schema', 'public_root', 'dependencies', 'environment', 'workspace',
              'gpu', 'families', 'outer_lifecycle'}
    if not isinstance(value, dict) or set(value) != fields or value['schema'] != SCHEMA:
        raise ValueError('Explicit trusted specialized operator required')
    if value['outer_lifecycle'] != OUTER_CONTRACT:
        raise ValueError('External independent-job lifecycle contract is mandatory')
    if type(value['gpu']) is not int or value['gpu'] < 0:
        raise ValueError('Explicit exclusive GPU index required')
    if set(value['workspace']) != {'input_roots', 'lock_root'}:
        raise ValueError('Explicit input and cooperative-lock boundaries required')
    if set(value['families']) != set(MODULES):
        raise ValueError('AtomicCoffee and current Recovery need separate profiles')
    import robotagentbench
    public = Path(value['public_root'])
    if public.resolve() != public or public != Path(robotagentbench.__file__).resolve().parent.parent:
        raise ValueError('Use the installed public package root')
    for family, profile in identities().items():
        supplied = value['families'][family]
        if set(supplied) != {'manifest', 'bindings', 'source_root', 'tasks'}:
            raise ValueError('Explicit external audited manifest and deployment bindings required')
        if set(supplied['tasks']) != set(profile['tasks']):
            raise ValueError('Every approved task needs its own existing prepared-input binding')
        for row in supplied['tasks'].values():
            if set(row) != {'launch', 'inputs', 'required_resources', 'output_parent'}:
                raise ValueError('Fixed prepared input and original-owner output scope required')
            for key in ('inputs', 'required_resources'):
                if (not isinstance(row[key], list) or len(set(row[key])) != len(row[key])
                        or any(not isinstance(item, str) or not item for item in row[key])):
                    raise ValueError('Distinct explicit resource identifiers required')
            if row['launch'] not in row['inputs']:
                raise ValueError('Launch must be an explicitly pinned input')
    if len({row['source_root'] for row in value['families'].values()}) != len(MODULES):
        raise ValueError('Two independent family source roots are mandatory')
    return value


def selection(operator, family, task):
    if family not in MODULES or task not in identities()[family]['tasks']:
        raise ValueError('Exact public task/family pair required')
    supplied = operator['families'][family]
    return supplied, identities()[family], supplied['tasks'][task]


def dependencies(operator):
    from robotagentbench.simulation.dependencies import RuntimeDependencies
    return RuntimeDependencies.from_config(operator['dependencies'])


def workspace_for(operator, task_config):
    from robotagentbench.simulation.workspace import NativeWorkspace
    boundary = operator['workspace']
    return NativeWorkspace(tuple(map(Path, boundary['input_roots'])),
        Path(task_config['output_parent']), Path(boundary['lock_root']))


def require_resource(deps, key, expected, *, kind=None):
    resource = deps.resources.get(key)
    if resource is None or resource.sha256 != expected or (kind and resource.kind != kind):
        raise ValueError('Resource differs from its existing reviewed identity')
    return deps.file(key)


def external_profile(operator, family, deps):
    """Verify the fixed camera-only derivative and retain its original anchor."""
    from . import camera_release
    supplied, anchor = operator['families'][family], identities()[family]
    camera = camera_release.profile(family)
    if camera['base_assembly_sha256'] != anchor['manifest_sha256']:
        raise ValueError('Camera release does not derive from the original owner anchor')
    original = require_resource(deps, supplied['manifest'] + '_upstream', anchor['manifest_sha256'])
    path = require_resource(deps, supplied['manifest'], camera['derived_assembly_sha256'])
    if camera_release.derive_assembly(original, family) != Path(path).read_bytes():
        raise ValueError('Selected assembly is not the exact reviewed camera-only derivative')
    manifest = read(path)
    if (manifest.get('schema') != anchor['manifest_schema']
            or manifest.get('owner') != anchor['manifest_owner']):
        raise ValueError('Wrong external audited assembly identity')
    profile = manifest['profiles'][family]
    if (profile['entry_module'] != MODULES[family]
            or len(profile['files']) != anchor['source_file_count']):
        raise ValueError('Wrong external original-owner source selection')
    files = profile['files']
    pins = {row['original']: row['sha256'] for row in files}
    if len(pins) != len(files):
        raise ValueError('Duplicate external source identity')
    bindings = read(deps.file(supplied['bindings']))
    if (set(bindings) != {'schema', 'family', 'source_resources', 'startup_resources',
                          'namespace_mounts', 'import_roots', 'entry_file'}
            or bindings['schema'] != BINDINGS or bindings['family'] != family):
        raise ValueError('Explicit family-specific read-only deployment bindings required')
    if set(bindings['source_resources']) != set(pins):
        raise ValueError('No required original source may be removed from external bindings')
    if set(bindings['startup_resources']) != set(anchor['required_startup_resources']):
        raise ValueError('Exact atomic startup resource roles required')
    for row in bindings['startup_resources'].values():
        if (not isinstance(row, dict) or set(row) != {'resource', 'namespace'}
                or not isinstance(row['namespace'], str) or not Path(row['namespace']).is_absolute()):
            raise ValueError('Explicit backing and original read-only startup namespace required')
    mounts = []
    marker = '/sources/' + family + '/'
    for row in profile['namespace_binds']:
        if row['source'].count(marker) != 1:
            raise ValueError('Unreviewed original assembly namespace shape')
        mounts.append(dict(target=row['target'], source_relative=row['source'].split(marker, 1)[1]))
    if bindings['namespace_mounts'] != mounts:
        raise ValueError('Original logical namespace recipe differs from audited manifest')
    entries = [p for p in pins if p.endswith(ENTRY_SUFFIXES[family])]
    if len(entries) != 1:
        raise ValueError('Fixed original adapter identity is ambiguous or absent')
    entry = Path(bindings['entry_file'])
    possible_entries = set()
    for original in entries:
        relative = Path(original.lstrip('/'))
        for row in mounts:
            prefix = Path(row['source_relative'])
            if relative.is_relative_to(prefix):
                possible_entries.add(Path(row['target']) / relative.relative_to(prefix))
    if entry not in possible_entries:
        raise ValueError('Entry must be the fixed adapter in its original namespace')
    if bindings['import_roots'] != expected_import_roots(family, bindings, pins):
        raise ValueError('Family-specific audited namespace precedence is mandatory')
    return dict(manifest=manifest, files=files, pins=pins, bindings=bindings,
                entry_sha256=pins[entries[0]], anchor=anchor)


def expected_import_roots(family, bindings, pins):
    """Infer exact roots from the sealed namespace, never accept arbitrary order.

    Current Recovery uses its Generic MOBILE22/native-driver/physical baseline.
    Atomic retains Frozen JOINT20/physics and its separate public robot_api.
    These are intentionally different source contracts, not a global reordering.
    """
    module_file = '/' + MODULES[family].replace('.', '/') + '.py'
    entry = bindings['entry_file']
    if not entry.endswith(module_file):
        raise ValueError('Fixed entry must resolve through its original qualified package alias')
    generic = entry[:-len(module_file)]
    targets = [row['target'] for row in bindings['namespace_mounts']]
    if generic not in targets:
        raise ValueError('The fixed adapter generic root must be an audited namespace target')
    # Preserve first canonical target order in the exact external manifest;
    # later legacy aliases do not select a different implementation.
    frozen = next((target for target in targets if target != generic and
                   target + '/evaluation/tasks/coffee_machine/api_runtime.py' in pins), None)
    if frozen is None:
        raise ValueError('Original frozen scoring fallback root is required')
    if family == 'failure_recovery':
        return [generic, frozen]
    buttons = [target for target in targets
               if target.endswith('/coffee_button_lighting_interference')
               and target + '/component_sources.json' in pins]
    if len(buttons) != 1:
        raise ValueError('Exact original button import namespace is required')
    return [str(Path(buttons[0]).parent), frozen, generic]


def readonly(path):
    """Read Linux mount metadata only; never create mounts or query a GPU."""
    mountinfo = Path('/proc/self/mountinfo')
    if not mountinfo.is_file():
        raise ValueError('Actual Linux read-only namespace must be checked before execute')
    mounts = []
    for line in mountinfo.read_text().splitlines():
        fields = line.split(' - ', 1)[0].split()
        target = fields[4]
        for escaped, actual in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            target = target.replace(escaped, actual)
        if path == Path(target) or path.is_relative_to(target):
            mounts.append((len(Path(target).parts), 'ro' in fields[5].split(',')))
    if not mounts or not max(mounts)[1]:
        raise ValueError('Original source/resource view is not mounted read-only')


def verify_sources(operator, family, *, mounted=False):
    supplied = operator['families'][family]
    deps = dependencies(operator)
    profile = external_profile(operator, family, deps)
    root = deps.directory(supplied['source_root'])
    for other, other_config in operator['families'].items():
        if other == family:
            continue
        other_root = deps.directory(other_config['source_root'])
        if root.is_relative_to(other_root) or other_root.is_relative_to(root):
            raise ValueError('Family source trees cannot overlap or alias one another')
    for row in profile['files']:
        actual = require_resource(deps, profile['bindings']['source_resources'][row['original']],
                                  row['sha256'], kind='source')
        if actual != root / row['original'].lstrip('/'):
            raise ValueError('Independent family backing must retain exact relative source identity')
    for role, expected in profile['anchor']['required_startup_resources'].items():
        require_resource(deps, profile['bindings']['startup_resources'][role]['resource'],
                         expected['sha256'], kind=expected['kind'])
    if mounted:
        for mount in profile['bindings']['namespace_mounts']:
            target, relative = Path(mount['target']), Path(mount['source_relative'])
            for original, expected in profile['pins'].items():
                logical = Path(original.lstrip('/'))
                if logical.is_relative_to(relative):
                    actual = target / logical.relative_to(relative)
                    if actual.resolve() != actual or not actual.is_file() or sha(actual) != expected:
                        raise ValueError('Actual source namespace differs from its pinned backing')
                    readonly(actual)
        # Startup paths are external trusted bindings, not hardcoded deployment
        # paths. They must expose the original JSON/USD/texture byte identities.
        for role, row in profile['bindings']['startup_resources'].items():
            actual = Path(row['namespace'])
            expected = profile['anchor']['required_startup_resources'][role]['sha256']
            if actual.resolve() != actual or not actual.is_file() or sha(actual) != expected:
                raise ValueError('Actual startup resource namespace differs from its exact binding')
            readonly(actual)
    return deps, root, profile


def verify(operator, family, task, *, mounted=False):
    supplied, anchor, task_config = selection(operator, family, task)
    deps, root, profile = verify_sources(operator, family, mounted=mounted)
    from robotagentbench.simulation.environment import NativeEnvironment
    env = NativeEnvironment.from_config(deps, operator['environment'])
    env.verify()  # File verification only; no interpreter/SDK execution.
    workspace = workspace_for(operator, task_config)
    inputs = {workspace.input(deps.file(key)): deps.resources[key].sha256 for key in task_config['inputs']}
    for key in task_config['required_resources']:
        path = deps.file(key)
        if mounted:
            readonly(path)
    launch_path = workspace.input(deps.file(task_config['launch']))
    launch = read(launch_path)
    if launch.get('schema') != 'openend.short_task.launch.v1' or launch.get('task_name') != task:
        raise ValueError('Only an unchanged existing original short-task launch is accepted')
    for key in ('request', 'manifest', 'scene', 'ready_pose', 'sensors', 'spawn'):
        if key in launch and workspace.input(launch[key]) not in inputs:
            raise ValueError('Launch input is outside its reviewed fixed input set')
    request, manifest = read(launch['request']), read(launch['manifest'])
    if request.get('task_name') != task or manifest.get('scene') != launch.get('scene'):
        raise ValueError('Prepared request, scene and task identities disagree')
    if mounted:
        for path in inputs:
            readonly(path)
    return dict(dependencies=deps, environment=env, workspace=workspace, root=root,
                launch_path=launch_path, launch=launch, inputs=inputs, profile=profile)
