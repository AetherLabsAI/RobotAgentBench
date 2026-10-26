"""Only scene-owned targets and explicit moving controls, never backgrounds."""
from .policy import SIMPLE, MULTI


def declarations(plan, manifest, ports, stage=None):
    name=plan['task_name'];bound=plan['target_bindings'];out={}
    def add(key, roots, owner, role):
        roots=tuple(dict.fromkeys(roots))
        if not roots or any(not isinstance(p,str) or not p.startswith('/World/') for p in roots):
            raise ValueError('Exact native interaction roots required')
        if key in out:raise ValueError('Duplicate interaction identity')
        out[key]=dict(roots=roots,owner=owner,role=role)
    if name in SIMPLE|MULTI:
        for row in bound:
            owner=row.get('instance_id')
            if owner is None and len(ports)==1:owner=next(iter(ports))
            if owner not in ports:raise ValueError('Interaction target lacks component owner')
            add(row['target_path'],[row['target_path']],owner,'target')
    else:
        if len(ports)!=1:raise ValueError('Device task must have one composite')
        owner,port=next(iter(ports.items()))
        if name=='prepare_coffee':
            roles=port.owner.service.roles
            for role in ('cup','bag','button'):add(role,[roles[role]],owner,role)
        elif name=='bake_bread':
            add('target',[port.source.paths['target']],owner,'target')
            candidates=[port.source.paths['tray']+'/Collision/'+n for n in
                        ('HandleGripExact','HandleGripLegacyClosure')]
            if stage is None:raise ValueError('Fryer handle requires composed scene verification')
            roots=[p for p in candidates if stage.GetPrimAtPath(p) and
                   stage.GetPrimAtPath(p).GetAttribute('physics:collisionEnabled').Get() is not False]
            if len(roots)!=1:raise ValueError('Exactly one enabled fryer grip collider required')
            add('basket',roots,owner,'basket')
        elif name=='fridge_to_heated_box':
            microwaves=[o['path'] for o in manifest['objects'] if o['id']=='microwave']
            if len(microwaves)!=1:raise ValueError('Unique microwave required')
            add('target',[bound['target']],owner,'target')
            add('fridge_handle',[port.fridge_root+'/right_door/col_handle'],owner,'fridge_handle')
            add('microwave_handle',[microwaves[0]+'/Asset/door/col_handle'],owner,'microwave_handle')
        elif name in ('retrieve_from_drawer','sofa_to_drawer'):
            add('target',[bound[0]['target_path']],owner,'target')
            drawer=port.drawer if name=='retrieve_from_drawer' else port.geometry.destination
            # The drawer asset has one merged wall/handle collider, not a
            # separate handle. Bind the moving drawer, NEVER the coffee table.
            add('drawer',[drawer],owner,'drawer')
        else:raise ValueError('Missing interaction binding for '+name)
    if len(out)!=plan['interaction_policy']['target_count']:
        raise ValueError('Interaction count differs from declared maximum')
    roots=[p for row in out.values() for p in row['roots']]
    if len(roots)!=len(set(roots)):raise ValueError('Same object claimed twice')
    return out
