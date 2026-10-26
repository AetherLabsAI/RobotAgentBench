"""Audited mechanical deployment of the five explicitly approved shared files.

Refuses concurrent source changes and unrelated source-record drift. Copies
and JSON hash substitutions are mechanical; every old byte is backed up.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil

from . import adapter

SHARED = ('short_tasks/registry.py','short_tasks/runtime.py','short_tasks/catalog.json',
          'short_tasks/agent_probe_v1/prepare_task.py','short_tasks/agent_probe_v1/native_owner.py')


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare_records():
    here, root = adapter.HERE, adapter.ROOT
    work, before = here/'work', here/'before'
    for relative in SHARED:
        if (root/relative).read_bytes() != (before/relative).read_bytes():
            raise ValueError('Concurrent production edit; refusing deployment: '+relative)
    catalog = json.loads((before/'short_tasks/catalog.json').read_text())
    rows = catalog['tasks']
    # Synchronize the existing registered shelf variant without claiming native acceptance.
    for row in rows:
        if row['name']=='stock_wooden_tray_interference':
            row.update(adapter_status='registered_task_owned_actor',rule_adapter_available=True,
                       native_validation='see_task_owned_receipts_not_full_public_rollout')
            row.pop('unavailable',None)
    if not any(r['name']=='stock_wooden_tray_lighting' for r in rows):
        rows.append(dict(index=None,name='stock_wooden_tray_lighting',
            title_zh='货架取物亮暗扰动',adapter='tasks/stock_wooden_tray_lighting/adapter.py',
            rule_adapter_available=True,adapter_status='registered_task_owned_lighting',
            native_validation='see_task_owned_receipts_not_full_public_rollout'))
    for name,(api,variant,_) in adapter.NAMES.items():
        if any(row['name']==name for row in rows):
            raise ValueError('Duplicate catalog registration: '+name)
        rows.append(dict(index=None,name=name,rule_version=adapter.rule(name),source_api=api,
            variant=variant,adapter=str(here/'adapter.py'),rule_adapter_available=True,
            maximum_for_example=5 if api=='1-3' else 6,
            adapter_status='registered_atomic_source_owner',layout_scope='atomic_source',
            native_validation='public_owner_pending_native_smoke',full_actions_validated=False))
    catalog.update(task_count=len(rows),robustness_task_count=6,
                   integration_20261014=dict(added=list(adapter.NAMES),public_protocol='short_tasks.gpt.robot22.v1',
                       scope='atomic_source_scene_not_50_layouts',physical_parameters_changed=False,
                       historical_summary_counts_not_recomputed=True))
    dump(work/'short_tasks/catalog.json',catalog)
    changes=[]
    for relative in SHARED:
        old,new=(before/relative),(work/relative)
        diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),
                                        fromfile='before/'+relative,tofile='after/'+relative))
        target=here/'diffs'/(relative+'.diff')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(diff)
        changes.append(dict(path=relative,old_sha256=adapter.digest(old),new_sha256=adapter.digest(new)))
    source_records=[]
    for pattern in ('short_tasks/tasks/*/component_sources.json','short_task_ContextReasoning/tasks/*/component_sources.json'):
        for record_path in root.glob(pattern):
            record=json.loads(record_path.read_text())
            field=next((k for k in ('sources','files_sha256','source_files') if isinstance(record.get(k),dict)),None)
            if field is None:
                continue
            value=json.loads(json.dumps(record))
            updated=[]
            for change in changes:
                key=change['path']
                if key in value[field]:
                    if value[field][key]!=change['old_sha256']:
                        raise ValueError('Existing source record already drifted: '+str(record_path)+': '+key)
                    value[field][key]=change['new_sha256']
                    updated.append(change)
            if updated:
                relative=str(record_path.relative_to(root))
                backup=here/'source_record_backups'/relative
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(record_path,backup)
                dump(here/'source_record_updates'/relative,value)
                source_records.append(dict(path=str(record_path),relative=relative,
                    old_sha256=adapter.digest(record_path),changes=updated))
    # Pin unchanged source and task dependencies. The sole substitutions are the
    # approved shared files, whose before/after hashes are recorded above.
    dependencies=set()
    for directory in (adapter.SOURCE/'arena_api',adapter.SOURCE/'arena_randomization'):
        dependencies.update(directory.glob('*.py'))
    dependencies.update((adapter.SOURCE/'evaluation/tasks/coffee_machine').rglob('*.py'))
    dependencies.add(adapter.SOURCE/'evaluation/physics_readers.py')
    for directory in (adapter.BAG,adapter.MACHINE,adapter.BUTTON,
                      root/'short_task_robustness/tasks/stock_wooden_tray_interference'):
        dependencies.update(directory.glob('*.py'))
    dependencies.update(here.glob('*.py'))
    dependencies.update((here/'robot_api').glob('*.py'))
    for name in ('robot_profiles.py','mobile_base.py','mobile_base_native.py','bimanual.py'):
        dependencies.add(root/'arena_api'/name)
    for name in ('host.py','contract.py','completion_stream.py'):
        dependencies.add(root/'short_tasks/agent_probe_v1'/name)
    hashes={str(p.resolve()):adapter.digest(p) for p in dependencies}
    for change in changes:
        hashes[str(root/change['path'])]=change['new_sha256']
    dump(here/'component_sources.json',dict(schema='short_tasks.atomic_coffee.integration_sources.v1',
        absolute_sources=hashes,shared_changes=changes,source_records_updated=len(source_records),
        public_host_copy_import_only_changes=True))
    dump(here/'deployment_manifest.json',dict(shared_changes=changes,source_records=source_records,
        shared_edits_user_authorized=True,source_record_updates_user_authorized=True,
        no_physics_or_existing_rule_changes=True))
    return dict(shared_files=len(changes),source_records=len(source_records),dependencies=len(hashes))


def install():
    here,root=adapter.HERE,adapter.ROOT
    record=json.loads((here/'deployment_manifest.json').read_text())
    # Validate the entire write set before the first production write.
    for row in record['shared_changes']:
        if adapter.digest(root/row['path'])!=row['old_sha256']:
            raise ValueError('Concurrent shared edit: '+row['path'])
        if adapter.digest(here/'work'/row['path'])!=row['new_sha256']:
            raise ValueError('Staged shared file changed: '+row['path'])
    for row in record['source_records']:
        if adapter.digest(row['path'])!=row['old_sha256']:
            raise ValueError('Concurrent source-record edit: '+row['path'])
    for row in record['shared_changes']:
        shutil.copy2(here/'work'/row['path'],root/row['path'])
    for row in record['source_records']:
        shutil.copy2(here/'source_record_updates'/row['relative'],row['path'])
    dump(here/'deployment_receipt.json',dict(installed=True,shared_files=record['shared_changes'],
        source_record_count=len(record['source_records']),full_actions_validated=False))
    return dict(installed=True,shared_files=5,source_records=len(record['source_records']))


def refresh_owned_sources():
    path=adapter.HERE/'component_sources.json'
    record=json.loads(path.read_text())
    changes=[]
    for name,old in list(record['absolute_sources'].items()):
        actual=adapter.digest(name)
        if actual==old:
            continue
        if not Path(name).resolve().is_relative_to(adapter.HERE):
            raise ValueError('Not a new task-owned integration source: '+name)
        changes.append(dict(path=name,old_sha256=old,new_sha256=actual))
        record['absolute_sources'][name]=actual
    record.setdefault('task_owned_revision_history',[]).append(changes)
    dump(path,record)
    return dict(task_owned_sources_refreshed=len(changes),external_sources_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare_records','install','refresh_owned_sources'))
    a=p.parse_args()
    print(json.dumps({'prepare_records':prepare_records,'install':install,
                      'refresh_owned_sources':refresh_owned_sources}[a.action]()))
