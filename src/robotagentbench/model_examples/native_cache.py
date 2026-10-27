"""Original task-private Kit/MDL deployment recipe; no SDK execution or edits.

The native SDK remains read-only. One fresh owned Kit copy and an initially
empty MDL search directory reproduce the original successful owner namespace.
This does not change simulator settings, cameras, physics, scoring or timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

SCHEMA='robotagentbench.native_cache_binding.v1'
PROVENANCE='robotagentbench.native_cache_provenance.v1'
RECIPE='original_cp_a_kit_and_empty_mdl_search.v1'
CORE_FILES=('libcarb.so','kit_app.py','launcher.toml','driver-requirements.json')


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(path):
    path=Path(path)
    if not path.is_absolute()or path.resolve(strict=True)!=path or not path.is_dir():
        raise ValueError('Existing canonical native directory required')
    return path


def targets(sdk_site):
    site=canonical(sdk_site);kit=canonical(site/'isaacsim/kit')
    rows=list((site/'isaacsim/extscache').glob('omni.mdl-*/search_paths'))
    if len(rows)!=1:raise ValueError('One original SDK-owned MDL search path required')
    return site,kit,canonical(rows[0])


def validate(value,*,sdk_site):
    if (not isinstance(value,dict)or set(value)!={'schema','owner_root','sdk_site','kit','mdl_search','provenance'}
        or value['schema']!=SCHEMA):raise ValueError('Exact task-private native cache binding required')
    site,kit_target,mdl_target=targets(sdk_site)
    root=canonical(value['owner_root'])
    if (root==site or root.is_relative_to(site)or value['sdk_site']!=str(site)
        or root in (Path.home(),Path(root.anchor))or root.stat().st_uid!=os.getuid()):
        raise ValueError('Private cache must be outside the unchanged shared SDK')
    expected={'kit':dict(source=str(root/'kit'),target=str(kit_target)),
              'mdl_search':dict(source=str(root/'mdl_search'),target=str(mdl_target))}
    if any(value[name]!=row for name,row in expected.items()):
        raise ValueError('Only the two fixed original Kit/MDL writable targets are permitted')
    proof=value['provenance'];path=root/'provenance.json'
    if not isinstance(proof,dict)or set(proof)!={'path','sha256'}or proof['path']!=str(path)or path.resolve()!=path or sha(path)!=proof['sha256']:
        raise ValueError('Private cache provenance was changed')
    record=json.loads(path.read_text())
    if (record.get('schema')!=PROVENANCE or record.get('recipe')!=RECIPE or record.get('sdk_site')!=str(site)
        or record.get('kit_source')!=str(kit_target)or record.get('mdl_source')!=str(mdl_target)
        or record.get('mdl_initially_empty')is not True or record.get('created_fresh')is not True
        or record.get('owner_root')!=str(root)or set(record.get('core_sha256',{}))!=set(CORE_FILES)):
        raise ValueError('Original cache-copy provenance differs')
    for relative,pin in record['core_sha256'].items():
        if sha(kit_target/relative)!=pin or sha(root/'kit'/relative)!=pin:
            raise ValueError('Original Kit code/configuration bytes differ: '+relative)
    for path in (root/'kit',root/'mdl_search',*(root/'kit'/name for name in ('cache','data','logs'))):
        canonical(path)
        if not os.access(path,os.W_OK):raise ValueError('Only owned cache directories may be writable')
    return [(root/'kit',kit_target),(root/'mdl_search',mdl_target)]


def prepare(sdk_site,destination):
    site,kit,mdl=targets(sdk_site);out=Path(destination)
    if (not out.is_absolute()or out.resolve()!=out or out.exists()or not out.parent.is_dir()
        or out==site or out.is_relative_to(site)or out==Path.home()or out==Path(out.anchor)):
        raise ValueError('Fresh owned cache directory outside the shared SDK required')
    before={relative:sha(kit/relative)for relative in CORE_FILES}
    out.mkdir()
    # The original reviewed deployment uses exactly this CPU file-copy recipe;
    # it does not reuse an old run's mutable cache or modify source permissions.
    subprocess.run(['cp','-a','--reflink=auto',str(kit),str(out/'kit')],check=True,timeout=90)
    for name in ('cache','data','logs'):
        path=out/'kit'/name
        if path.exists()and(path.resolve()!=path or not path.is_dir()):raise ValueError('Cache path escapes its owned Kit copy')
        path.mkdir(exist_ok=True)
    (out/'mdl_search').mkdir()
    if any(sha(kit/relative)!=pin or sha(out/'kit'/relative)!=pin for relative,pin in before.items()):
        raise ValueError('Kit source changed during copy or copied core bytes differ')
    record=dict(schema=PROVENANCE,recipe=RECIPE,sdk_site=str(site),kit_source=str(kit),
        owner_root=str(out),created_fresh=True,
        mdl_source=str(mdl),mdl_initially_empty=True,core_sha256=before,
        shared_sdk_modified=False,old_run_cache_reused=False,simulator_started=False,model_called=False)
    path=out/'provenance.json'
    with path.open('x')as stream:json.dump(record,stream,indent=2);stream.write('\n')
    binding=dict(schema=SCHEMA,owner_root=str(out),sdk_site=str(site),
        kit=dict(source=str(out/'kit'),target=str(kit)),
        mdl_search=dict(source=str(out/'mdl_search'),target=str(mdl)),
        provenance=dict(path=str(path),sha256=sha(path)))
    validate(binding,sdk_site=site)
    with(out/'binding.json').open('x')as stream:json.dump(binding,stream,indent=2);stream.write('\n')
    return binding


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdk-site',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(prepare(a.sdk_site,a.destination),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
