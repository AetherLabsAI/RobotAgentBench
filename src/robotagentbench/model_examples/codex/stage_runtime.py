"""Explicit independent Runtime copy plus the reviewed server coding delta.

Only the newly created destination is written. The old Runtime, immutable
backend manifest, task source/scoring/physics, and running environment remain
untouched. This does not start Docker, Isaac, a GPU or a model.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
from .server_profile import PREFIX,FILES,payloads

OLD_LAUNCHER='500024e1a0d67afcd64f48e1c653890bb7ab3b0d42db829cc1ed721a39b64a4a'

def stage(source_runtime,destination):
    from robotagentbench.example import runtime
    original,source,config=runtime(source_runtime)
    destination=Path(destination)
    if (not destination.is_absolute()or destination.resolve()!=destination or destination.exists()
        or destination==Path(destination.anchor)or not destination.parent.is_dir()):
        raise ValueError('Explicit fresh canonical output under an existing dedicated parent required')
    if destination.is_relative_to(original)or original.is_relative_to(destination):raise ValueError('Source/destination overlap')
    raw=(original/'runtime.json').read_bytes();delta=payloads();pins=dict(config['source_files'])
    for name,data in delta.items():
        expected=hashlib.sha256(data).hexdigest();old=pins.get(name)
        if old is not None and old!=expected and not(name==PREFIX+'sandbox_launcher.py'and old==OLD_LAUNCHER):
            raise ValueError('Unexpected original source identity conflict: '+name)
        old_path=source/name
        if old_path.exists():
            actual=hashlib.sha256(old_path.read_bytes()).hexdigest()
            if old is None or actual!=old:raise ValueError('Original source is not the reviewed input: '+name)
    destination.mkdir(mode=0o750)
    with (destination/'OWNER.json').open('x')as stream:
        json.dump(dict(owner='RobotAgentBench-Codex-connection-v32',purpose='independent source/resource copy'),stream)
    target=destination/'Runtime'
    # cp -a retains internal Runtime hardlinks but never hardlinks the copy to
    # its provider. Reflinks, when supported, are independent copy-on-write.
    subprocess.run(['cp','-a','--reflink=auto',str(original),str(target)],check=True)
    relative=source.relative_to(original);root=target/relative
    changed=[]
    for name,data in delta.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():path.unlink()  # Only in the new copy, never the provider.
        with path.open('xb')as stream:stream.write(data)
        digest=hashlib.sha256(data).hexdigest()
        changed.append(dict(relative=name,original_sha256=pins.get(name),selected_sha256=digest))
        pins[name]=digest
    def rebase(value):
        path=Path(value)
        return str(target/path.relative_to(original))if path.is_absolute()and path.is_relative_to(original)else value
    selected=dict(config,runtime=str(target),source_root=str(root),source_files=pins,
        resources={key:rebase(value)for key,value in config['resources'].items()})
    record=target/'runtime.json';record.unlink()
    with record.open('x')as stream:json.dump(selected,stream,indent=2);stream.write('\n')
    receipt=dict(schema='robotagentbench.codex_runtime_stage.v32',original_runtime=str(original),
        original_runtime_json_sha256=hashlib.sha256(raw).hexdigest(),runtime=str(target),source_root=str(root),
        source_delta=changed,original_runtime_modified=False,scoring_or_physics_modified=False,
        native_started=False,model_started=False,
        remaining='Use existing audited native namespace/ABI and exact policy image; model runs require explicit operator command.')
    with (destination/'STAGING.json').open('x')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    return receipt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path);p.add_argument('--stage',action='store_true')
    a=p.parse_args(argv)
    if not a.stage:
        print(json.dumps(dict(mode='describe',writes=False,delta_members=len(FILES),native_started=False)));return 0
    result=stage(a.runtime,a.destination)
    print(json.dumps({key:result[key]for key in ('runtime','source_root','original_runtime_modified','native_started','model_started')}));return 0

if __name__=='__main__':raise SystemExit(main())
