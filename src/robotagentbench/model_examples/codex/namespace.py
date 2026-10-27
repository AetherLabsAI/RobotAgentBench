"""Trusted Linux native-owner namespace; describe by default, no model client.

The entry argv is operator configuration, never a robot/model tool argument.
It can select the coding worker or an original raw worker using the same
reviewed Runtime. The policy workspace retains its real host path so Docker's
host daemon sees the same files as the native owner.
"""
import argparse,hashlib,json,os,stat,subprocess
from pathlib import Path

def docker_socket_group(path):
    """Preserve only access the trusted host operator already possesses."""
    info=os.stat(path)
    if not stat.S_ISSOCK(info.st_mode):raise ValueError('Configured Docker endpoint is not a Unix socket')
    if info.st_uid==os.getuid()or info.st_gid==os.getgid():return None
    if info.st_gid not in os.getgroups():raise ValueError('Docker socket group is not already owned by the trusted operator')
    return info.st_gid

def command(value):
    from robotagentbench.namespace import command as original_namespace
    from robotagentbench.example import runtime
    fields={'schema','runtime','image','name','gpu','lock_root','sdk_site',
            'package_root','entry_argv','readonly_roots','docker_cli','docker_host','intent'}
    if set(value)-{'readonly_aliases','graphics_binding','native_cache_binding'}!=fields or value['schema']!='robotagentbench.codex_namespace.v1':
        raise ValueError('Explicit trusted native namespace configuration required')
    entry=value['entry_argv']
    if not isinstance(entry,list)or not entry or any(not isinstance(x,str)or not x or '\x00'in x for x in entry):
        raise ValueError('Trusted entry argv required, not a shell command')
    base,source,_=runtime(value['runtime'])
    args=original_namespace(base,value['image'],'run',name=value['name'],gpu=value['gpu'],
        locks=value['lock_root'],sdk_site=value['sdk_site'])
    image_index=args.index(value['image']);args=args[:image_index]
    targets={}
    for i,arg in enumerate(args):
        if arg=='--mount':
            item=dict(part.split('=',1)for part in args[i+1].split(',')if '='in part)
            targets[item['dst']]=item['src']
    def mount(path,readonly,target=None):
        p=Path(path)
        if not p.is_absolute()or p.resolve()!=p or not p.exists()or ','in str(p):
            raise ValueError('Existing canonical native namespace path required')
        target=str(p)if target is None else target
        t=Path(target)
        if not t.is_absolute()or '..'in t.parts or ','in target or str(t)!=target:
            raise ValueError('Canonical absolute trusted alias target required')
        if target in targets:raise ValueError('Duplicate explicit mount target')
        targets[target]=str(p)
        args.extend(['--mount','type=bind,src='+str(p)+',dst='+target+(',readonly'if readonly else'')])
    # Same-path writable Runtime is required by original sandbox host binds;
    # no source/scorer edits are made by this namespace wrapper.
    mount(base,False)
    mount(value['package_root'],True);mount(value['sdk_site'],True)
    roots=value['readonly_roots']
    if not isinstance(roots,list)or len(roots)!=len(set(roots)):raise ValueError('Unique explicit read-only roots required')
    for root in roots:mount(root,True)
    aliases=value.get('readonly_aliases',{})
    if not isinstance(aliases,dict):raise ValueError('Explicit read-only alias map required')
    for target,source in aliases.items():mount(source,True,target)
    if 'graphics_binding'in value:
        from robotagentbench.model_examples.native_graphics import validate,CAPABILITIES,TARGET
        graphics=value['graphics_binding'];source=validate(graphics)
        caps=[args[i+1].split('=',1)[1]for i,x in enumerate(args[:-1])
              if x in ('-e','--env')and args[i+1].startswith('NVIDIA_DRIVER_CAPABILITIES=')]
        if len(caps)!=1 or set(caps[0].split(','))!=CAPABILITIES:
            raise ValueError('Preserve the original compute/graphics/utility capability set')
        if '--runtime'in args:
            if args[args.index('--runtime')+1]!='nvidia':raise ValueError('Conflicting native graphics runtime')
        else:args+=['--runtime','nvidia']
        mount(source,True,TARGET)
        args+=['--env','VK_DRIVER_FILES='+TARGET]
    if 'native_cache_binding'in value:
        from robotagentbench.model_examples.native_cache import validate
        for source,target in validate(value['native_cache_binding'],sdk_site=value['sdk_site']):
            mount(source,False,str(target))
    endpoint=value['docker_host'];cli=value['docker_cli']
    if (endpoint is None)!=(cli is None):raise ValueError('Docker CLI and endpoint must be configured together')
    if endpoint is not None:
        if not isinstance(endpoint,str)or not endpoint.startswith('unix:///'):raise ValueError('Trusted Unix Docker endpoint required')
        socket_path=endpoint[len('unix://'):]
        mount(cli,True);mount(socket_path,False)
        group=docker_socket_group(socket_path)
        if group is not None:args+=['--group-add',str(group)]
        args+=['--env','DOCKER_HOST='+endpoint]
    args+=['--env','PYTHONPATH='+str(value['package_root']),value['image'],*entry]
    return args

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    value=json.loads(a.config.read_text());cmd=command(value)
    if not a.execute:
        print(json.dumps(dict(argv=cmd,execute=False,native_started=False,model_started=False),indent=2));return 0
    intent=Path(value['intent'])
    if not intent.is_absolute()or intent.resolve()!=intent or not intent.parent.is_dir():raise ValueError('Dedicated existing intent parent required')
    digest=hashlib.sha256(a.config.read_bytes()).hexdigest()
    # Once claimed, even an interrupted dispatch cannot be silently repeated.
    with intent.open('x')as stream:
        json.dump(dict(config_sha256=digest,status='dispatch_intent',automatic_retry=False,
            container_name=cmd[cmd.index('--name')+1],
            argv_sha256=hashlib.sha256(json.dumps(cmd,separators=(',',':')).encode()).hexdigest()),stream)
        stream.flush();os.fsync(stream.fileno())
    # Preserve JSONL stdio and Docker -i; never add a TTY or a shell.
    return subprocess.run(cmd,check=False).returncode

if __name__=='__main__':raise SystemExit(main())
