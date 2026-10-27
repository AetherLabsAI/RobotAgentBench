"""Image-owned native SDK baseline and fresh per-episode CPU cache preparation.

No SDK is imported. The host validates portable metadata without installing an
SDK; the immutable image validates its original baseline again before startup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from robotagentbench.model_examples import native_cache, native_graphics

SDK_SITE='/workspace/lukun/arena_software/isaac51-py311/lib/python3.11/site-packages'
SDK_ALIAS='/opt/robotagentbench/native'
IMAGE_BASELINE=Path('/opt/robotagentbench/native-baseline.json')
SCHEMA='robotagentbench.native_image_deployment.v1'
BASELINE='robotagentbench.native_image_baseline.v1'
FILES=frozenset(('cache/binding.json','cache/provenance.json','graphics/binding.json',
    'graphics/source.json','graphics/nvidia_original_icd.json','image_baseline.json'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')


def image_id(value):
    if not isinstance(value,str)or not re.fullmatch('sha256:[0-9a-f]{64}',value):
        raise ValueError('Resolve the actual immutable Docker image ID first')
    return value


def name(value):
    if not isinstance(value,str)or not re.fullmatch('[A-Za-z0-9_-]{1,80}',value):
        raise ValueError('Simple fresh native deployment name required')
    return value


def baseline_record(value):
    expected={'schema','sdk_site','kit_target','mdl_target','core_sha256'}
    if not isinstance(value,dict)or set(value)!=expected or value['schema']!=BASELINE or value['sdk_site']!=SDK_SITE:
        raise ValueError('Require the image-owned canonical SDK baseline')
    site=Path(SDK_SITE);mdl=Path(value['mdl_target'])
    if (value['kit_target']!=str(site/'isaacsim/kit')or mdl.parent.parent!=site/'isaacsim/extscache'
            or mdl.name!='search_paths'or not re.fullmatch(r'omni\.mdl-[A-Za-z0-9+_.-]+',mdl.parent.name)
            or set(value['core_sha256'])!=set(native_cache.CORE_FILES)):
        raise ValueError('Only canonical Kit and one original MDL search target are permitted')
    for pin in value['core_sha256'].values():
        if not isinstance(pin,str)or not re.fullmatch('[0-9a-f]{64}',pin):raise ValueError('Invalid image core SHA')
    return value


def create_baseline(sdk_site,destination):
    """Docker build only: seal original SDK hashes outside writable overlays."""
    site,kit,mdl=native_cache.targets(Path(sdk_site).resolve(strict=True))
    if str(site)!=SDK_SITE:raise ValueError('Docker image SDK differs from its canonical recipe')
    out=Path(destination)
    if not out.is_absolute()or out.resolve()!=out or out.exists()or not out.parent.is_dir():
        raise ValueError('Fresh image-owned baseline destination required')
    pins={relative:sha(kit/relative)for relative in native_cache.CORE_FILES}
    record=baseline_record(dict(schema=BASELINE,sdk_site=str(site),kit_target=str(kit),mdl_target=str(mdl),core_sha256=pins))
    write(out,record);out.chmod(0o444)
    return record


def verify_image_baseline(*,sdk_site,baseline=IMAGE_BASELINE,check_visible_sdk=False):
    if sdk_site not in(SDK_ALIAS,SDK_SITE):raise ValueError('Use the SDK shipped in the pinned image')
    record=baseline_record(read(baseline))
    for relative,pin in record['core_sha256'].items():
        if check_visible_sdk and sha(Path(record['kit_target'])/relative)!=pin:
            raise ValueError('Visible image SDK core differs from its immutable original baseline')
    return record


def prepare(runtime,episode,*,sdk_site,image,baseline=IMAGE_BASELINE):
    """Run in the CPU-only image, writing only a fresh /job deployment folder."""
    image=image_id(image);episode=name(episode)
    base=Path(runtime)
    if base.resolve(strict=True)!=base or not base.is_dir():raise ValueError('Canonical owned Runtime required')
    original=verify_image_baseline(sdk_site=sdk_site,baseline=baseline,check_visible_sdk=True)
    parent=base/'native_deployments'
    if parent.exists()and(parent.resolve()!=parent or not parent.is_dir()):raise ValueError('Owned deployment parent required')
    parent.mkdir(exist_ok=True);out=parent/episode
    out.mkdir(exist_ok=False)
    native_cache.prepare(SDK_SITE,out/'cache')
    native_graphics.prepare(out/'graphics')
    shutil.copyfile(baseline,out/'image_baseline.json')
    record=dict(schema=SCHEMA,image_id=image,name=episode,sdk_site=SDK_SITE,
        kit_target=original['kit_target'],mdl_target=original['mdl_target'],
        baseline_sha256=sha(out/'image_baseline.json'),core_sha256=original['core_sha256'],
        files_sha256={relative:sha(out/relative)for relative in sorted(FILES)})
    write(out/'deployment.json',record)
    return record


def validate(runtime,episode,*,image,sdk_site):
    """Host-side checks: no host SDK, container, GPU, or model is accessed."""
    if sdk_site not in(SDK_ALIAS,SDK_SITE):raise ValueError('Use the SDK shipped in the pinned image')
    base=Path(runtime);out=base/'native_deployments'/name(episode)
    if (base.resolve(strict=True)!=base or out.resolve(strict=True)!=out or not out.is_dir()
            or out.stat().st_uid!=os.getuid()):raise ValueError('Owned canonical prepared native deployment required')
    path=out/'deployment.json'
    if path.resolve(strict=True)!=path:raise ValueError('Canonical deployment metadata required')
    record=read(path)
    if (not isinstance(record,dict)or set(record)!={'schema','image_id','name','sdk_site','kit_target','mdl_target',
            'baseline_sha256','core_sha256','files_sha256'}or record['schema']!=SCHEMA
            or record['image_id']!=image_id(image)or record['name']!=episode
            or set(record['files_sha256'])!=FILES):
        raise ValueError('Native deployment requires its exact original immutable image and file set')
    for relative,pin in record['files_sha256'].items():
        item=out/relative
        if item.resolve(strict=True)!=item or not item.is_file()or sha(item)!=pin:
            raise ValueError('Prepared native deployment file changed: '+relative)
    original=baseline_record(read(out/'image_baseline.json'))
    if (record['baseline_sha256']!=sha(out/'image_baseline.json')
            or any(record[key]!=original[key]for key in('sdk_site','kit_target','mdl_target','core_sha256'))):
        raise ValueError('Prepared deployment differs from original image baseline')
    logical=Path('/job/native_deployments')/episode
    binding=read(out/'cache/binding.json');proof=read(out/'cache/provenance.json')
    if (binding!={ 'schema':native_cache.SCHEMA,'owner_root':str(logical/'cache'),'sdk_site':SDK_SITE,
          'kit':dict(source=str(logical/'cache/kit'),target=record['kit_target']),
          'mdl_search':dict(source=str(logical/'cache/mdl_search'),target=record['mdl_target']),
          'provenance':dict(path=str(logical/'cache/provenance.json'),sha256=sha(out/'cache/provenance.json'))}
            or proof.get('schema')!=native_cache.PROVENANCE or proof.get('recipe')!=native_cache.RECIPE
            or proof.get('owner_root')!=str(logical/'cache')or proof.get('created_fresh')is not True
            or proof.get('mdl_initially_empty')is not True or proof.get('core_sha256')!=record['core_sha256']):
        raise ValueError('Original fresh Kit copy and empty-MDL provenance required')
    for relative,pin in record['core_sha256'].items():
        item=out/'cache/kit'/relative
        if item.resolve(strict=True)!=item or sha(item)!=pin:raise ValueError('Prepared Kit core differs from image baseline')
    for item in(out/'cache/kit',out/'cache/mdl_search',*(out/'cache/kit'/p for p in('cache','data','logs'))):
        if item.resolve(strict=True)!=item or not item.is_dir()or not os.access(item,os.W_OK):
            raise ValueError('Dedicated writable native cache directories required')
    icd=out/'graphics/nvidia_original_icd.json'
    if icd.read_bytes()!=native_graphics.ICD_BYTES:raise ValueError('Original ICD bytes changed')
    graphics=read(out/'graphics/binding.json')
    expected=dict(schema=native_graphics.SCHEMA,source=str(logical/'graphics/nvidia_original_icd.json'),
        sha256=native_graphics.SHA256,target=native_graphics.TARGET,runtime='nvidia',
        environment=dict(VK_DRIVER_FILES=native_graphics.TARGET))
    if graphics!=expected:raise ValueError('Only the original explicit read-only ICD alias is permitted')
    return dict(record=record,root=out,metadata=path,sha256=sha(path),icd=icd,
        mounts=((out/'cache/kit',record['kit_target']),(out/'cache/mdl_search',record['mdl_target'])))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    build=sub.add_parser('image-baseline');build.add_argument('--sdk-site',required=True);build.add_argument('--destination',type=Path,required=True)
    for mode in('prepare','run'):
        p=sub.add_parser(mode);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--name',required=True)
        p.add_argument('--image-id',required=True);p.add_argument('--sdk-site',required=True)
        if mode=='run':
            p.add_argument('--deployment-sha256',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--lock-root',required=True)
    args=parser.parse_args(argv)
    if args.mode=='image-baseline':result=create_baseline(args.sdk_site,args.destination)
    elif args.mode=='prepare':result=prepare(args.runtime,args.name,sdk_site=args.sdk_site,image=args.image_id)
    else:
        checked=validate(args.runtime,args.name,image=args.image_id,sdk_site=args.sdk_site)
        if checked['sha256']!=args.deployment_sha256:raise ValueError('Deployment changed after host validation')
        baseline=verify_image_baseline(sdk_site=args.sdk_site,check_visible_sdk=True)
        if baseline!=read(checked['root']/'image_baseline.json'):raise ValueError('Prepared cache belongs to a different image SDK baseline')
        from robotagentbench.example import main as example
        return example(['--runtime',str(args.runtime),'--run','--name',args.name,'--sdk-site',args.sdk_site,
                        '--gpu',str(args.gpu),'--lock-root',args.lock_root])
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
