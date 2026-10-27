"""Install exact original preparation inputs into a fresh task-owned source view."""
import hashlib
import json
from pathlib import Path
from .source_manifest import MANIFEST, EMBEDDED_DATA
from .resource_contract import CONTRACT

HERE=Path(__file__).resolve().parent
OWNER={'owner':'RobotAgentBench-composite-instance-bank-v31'}


def install_preparers(bank_root, source_root):
    bank_root,source_root=Path(bank_root),Path(source_root)
    if (bank_root.resolve()!=bank_root or source_root.resolve()!=source_root
            or not source_root.is_relative_to(bank_root) or source_root==bank_root
            or json.loads((bank_root/'OWNER.json').read_text())!=OWNER):
        raise ValueError('Only a canonical source view inside this owned instance bank may be extended')
    rows=[]
    for row in MANIFEST['sources']:
        relative=Path(row['target_relative'])
        if relative.is_absolute()or'..'in relative.parts:raise ValueError('Unsafe preparer path')
        raw=EMBEDDED_DATA.get(row['bundled_relative'])
        if raw is None:raw=(HERE/row['bundled_relative']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Bundled original preparer changed')
        target=source_root/relative
        if target.exists():
            if target.read_bytes()!=raw:raise ValueError('Never replace a different existing source')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb')as stream:stream.write(raw)
        rows.append(dict(path=str(target),sha256=row['sha256']))
    return rows


def install_generation_resources(bank_root, unpacked_resources):
    """Copy exact helper inputs, not install packages into any Python/SDK site."""
    root,original=Path(bank_root),Path(unpacked_resources)
    if (root.resolve()!=root or original.resolve()!=original
            or json.loads((root/'OWNER.json').read_text())!=OWNER):
        raise ValueError('Canonical owned bank and explicit extracted resource root required')
    receipt=original/'generation_dependencies/receipt.json'
    if hashlib.sha256(receipt.read_bytes()).hexdigest()!=CONTRACT['source_receipt_sha256']:
        raise ValueError('Original preparation resource receipt changed')
    verified=[]
    for relative,row in CONTRACT['files'].items():
        rel=Path(relative)
        if rel.is_absolute()or'..'in rel.parts:raise ValueError('Unsafe resource member')
        source,target=original/rel,root/rel
        if source.resolve()!=source or target.resolve()!=target:
            raise ValueError('Resource aliases must not escape the supplied roots')
        raw=source.read_bytes()
        if len(raw)!=row['bytes']or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('Original preparation resource changed: '+relative)
        if target.exists()and target.read_bytes()!=raw:
            raise ValueError('Never replace different existing preparation inputs')
        verified.append((target,raw))
    copied=0
    for target,raw in verified:
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb')as stream:stream.write(raw)
            copied+=1
    target=root/'generation_dependencies/receipt.json'
    if target.exists():
        if target.read_bytes()!=receipt.read_bytes():raise ValueError('Existing resource receipt differs')
    else:
        with target.open('xb')as stream:stream.write(receipt.read_bytes())
    return dict(files_verified=len(verified),files_copied=copied,
                read_only_namespace=dict(source=str(root/'generation_dependencies/ziche'),
                    target='/mnt/agent-bench-data/lingjun/ziche'),
                packages_or_sdk_modified=False,third_party_licenses_preserved=True)
