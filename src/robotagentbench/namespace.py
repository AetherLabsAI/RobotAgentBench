"""Launch the coffee example with local resources in a container-only namespace."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from .example import runtime

GENERIC='/home/lingjun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source'
CURRENT='/workspace/lukun/robot_agent_arena/short_tasks/agent_probe_v1/deployment_generic_20261010/source'


def command(workspace,image,mode,*,name=None,gpu=None,locks=None,sdk_site='/opt/robotagentbench/native',env=None,
            native_deployment=False):
    env=os.environ if env is None else env
    if mode not in ('plan','prepare','prepare-native','run')or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/@-]*',image):
        raise ValueError('Explicit container image and example mode required')
    if mode!='plan'and(not name or not re.fullmatch('[A-Za-z0-9_-]{1,80}',name)):
        raise ValueError('Simple episode name required')
    base,source,config=runtime(workspace)
    deployment=None
    if mode=='prepare-native':
        from .simulation.native_deployment import image_id
        image_id(image)
        if(base/'native_deployments'/name).exists():raise ValueError('Native deployment already exists; choose a fresh episode name')
    if native_deployment:
        if mode!='run':raise ValueError('Prepared native overlays apply only to native execution')
        from .simulation.native_deployment import validate
        deployment=validate(base,name,image=image,sdk_site=sdk_site)
    mounts={}
    def mount(actual,target,readonly=True):
        actual=Path(actual)
        if actual.resolve()!=actual or not actual.exists()or','in str(actual)or','in target:
            raise ValueError('Canonical container mount required')
        if target in mounts and mounts[target]!=(actual,readonly):raise ValueError('Conflicting task namespace')
        mounts[target]=(actual,readonly)
    mount(base,'/job',False)
    mount(source,GENERIC,False);mount(source,CURRENT,False)
    libraries=base/'libraries'
    for library,targets in (
        ('frozen_assets',('/home/lingjun/robot_agent_arena/api-protocol-st18-20261013-frozen4',
                          '/workspace/lukun/robot_agent_arena/api-protocol-st18-20261013-frozen4')),
        ('current_assets',('/home/lingjun/robot_agent_arena/api-protocol',
                           '/workspace/lukun/robot_agent_arena/api-protocol'))):
        if(libraries/library).is_dir():
            for target in targets:mount(libraries/library,target)
    # The asset index retains original absolute identities *below* the local
    # library directory. Expose those identities only inside this container.
    # configure() assembles legacy_data and recovery_original_layouts into
    # the same pinned original_absolute tree. Reject an older partial Runtime
    # with a hidden library instead of silently mounting only one half.
    from .installation import original_library_alias
    pins=config.get('library_files',{})
    for relative,pin in pins.items():
        alias=original_library_alias(relative)
        if alias is None or not alias.startswith('libraries/original_absolute/mnt/agent-bench-data/lingjun/'):
            continue
        target=base/alias
        if (pins.get(alias)!=pin or not target.is_file()or target.resolve()!=target):
            raise ValueError('Reconfigure this Runtime to assemble the exact original layout/library namespace')
    for library in ('original_absolute', 'legacy_data'):
        legacy=libraries/library/'mnt/agent-bench-data/lingjun'
        if legacy.is_dir():
            mount(legacy,'/mnt/agent-bench-data/lingjun')
            break
    for relative in ('layout_task_activation_v1','openend/tasks/all_objects_integration',
                     'openend/tasks/cloth_clear'):
        if(source/relative).is_dir():mount(source/relative,'/home/lingjun/robot_agent_arena/'+relative)
    # Room preparation includes the complete pinned background layer family.
    # Mount the family as a unit so its relative USD sublayers stay together.
    if(source/'assets').is_dir():
        for target in('/home/lingjun/robot_agent_arena/api-protocol/assets',
                      '/workspace/lukun/robot_agent_arena/api-protocol/assets'):
            mount(source/'assets',target)
    args=['docker','run','--rm','--interactive','--name','rab-example-'+uuid.uuid4().hex[:16],
          '--network','none','--read-only','--user',str(os.getuid())+':'+str(os.getgid()),
          '--cap-drop','ALL','--security-opt','no-new-privileges',
          '--tmpfs','/tmp:rw,nosuid,nodev,size=1073741824',
          '--env','PYTHONDONTWRITEBYTECODE=1','--env','HOME=/tmp/rab-home',
          '--label','org.robotagentbench.job=coffee-example']
    if mode=='run':
        if type(gpu)is not int or gpu<0 or locks is None:raise ValueError('Exclusive GPU and shared lock directory required')
        accepted=env.get('OMNI_KIT_ACCEPT_EULA','')
        if accepted.upper()not in('YES','Y','1'):raise ValueError('Accept the NVIDIA Omniverse EULA explicitly before native execution')
        mount(Path(locks).resolve(strict=True),'/locks',False)
        args+=['--gpus','all','--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics',
               '--env','OMNI_KIT_ACCEPT_EULA='+accepted]
        if deployment is not None:
            from .model_examples.native_graphics import TARGET
            for actual,target in deployment['mounts']:mount(actual,target,False)
            mount(deployment['icd'],TARGET)
            args+=['--runtime','nvidia','--env','VK_DRIVER_FILES='+TARGET]
    else:
        args+=['--runtime','runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=']
    for target,(actual,readonly)in sorted(mounts.items(),key=lambda row:len(Path(row[0]).parts)):
        value='type=bind,src='+str(actual)+',dst='+target+(',readonly'if readonly else'')
        args+=['--mount',value]
    if mode=='prepare-native'or deployment is not None:
        args += [image,'/usr/local/bin/python3.11','-B','-m','robotagentbench.simulation.native_deployment',
                 'prepare'if mode=='prepare-native'else'run','--runtime','/job','--image-id',image]
        if deployment is not None:args+=['--deployment-sha256',deployment['sha256']]
    else:
        args += [image,'/usr/local/bin/python3.11','-B','-m','robotagentbench.example',
                 '--runtime','/job','--'+mode]
    if name:args+=['--name',name]
    if mode!='plan':args+=['--sdk-site',sdk_site]
    if mode=='run':args+=['--gpu',str(gpu),'--lock-root','/locks']
    return args


def resolve_image(image):
    """Read local Docker metadata only; never pull an image or query a GPU."""
    from .simulation.native_deployment import image_id
    result=subprocess.run(['docker','image','inspect','--format','{{.Id}}',image],
                          capture_output=True,text=True,check=True)
    return image_id(result.stdout.strip())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--image',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    for flag in('plan','prepare','prepare-native','run'):mode.add_argument('--'+flag,action='store_true',dest=flag.replace('-','_'))
    parser.add_argument('--name');parser.add_argument('--gpu',type=int)
    parser.add_argument('--lock-root',type=Path)
    parser.add_argument('--sdk-site',default='/opt/robotagentbench/native')
    parser.add_argument('--print-command',action='store_true')
    args=parser.parse_args(argv)
    selected=next(flag for flag in('plan','prepare','prepare-native','run')if getattr(args,flag.replace('-','_')))
    image=resolve_image(args.image)if selected in('prepare-native','run')else args.image
    cmd=command(args.runtime,image,selected,name=args.name,gpu=args.gpu,
                locks=args.lock_root,sdk_site=args.sdk_site,native_deployment=selected=='run')
    if args.print_command:
        print(json.dumps(cmd,indent=2));return 0
    # Preserve live JSONL stdin/stdout, including Docker's -i handoff. There is
    # no automatic retry and no generic process/container termination command.
    return subprocess.run(cmd,check=False).returncode

if __name__=='__main__':raise SystemExit(main())
