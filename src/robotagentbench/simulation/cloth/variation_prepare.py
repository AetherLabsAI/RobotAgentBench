"""CPU-only immutable cloth-instance preparation using original geometry.

Nothing here imports Isaac, changes a score rule, or starts a simulator. Native
settling/admission is still required when a saved instance is first loaded.
"""
from pathlib import Path
import json

from . import randomization as reset


def geometry_gate(original, geometry, basket_source, *, original_native, owner_root):
    """Use the original convex basket guard and actual tabletop input mesh.

    Rigid transforms preserve each cloth's self-intersection state and shape.
    Native admission repeats the original self/collision audits before physics.
    Above-rim drape is retained; it is not mistaken for a fully contained cloth.
    """
    import sys
    import numpy as np
    # Paths come from the hash-pinned trusted operator, never an agent request.
    sys.path[:0] = [str(original_native),str(owner_root)]
    from basket_profile_guard_all_cloths_v1 import BasketWallGuard
    from placement import convex_polygon
    with np.load(geometry,allow_pickle=False) as archive:
        metadata = json.loads(str(archive['scene_collision_json']))
        row = next(r for r in metadata['meshes'] if r['prim_path']=='/World/Assets/CoffeeTable/Mesh')
        table = np.asarray(archive['scene_points_'+row['key']],float)
    source = json.loads(Path(basket_source).read_text())
    guard = BasketWallGuard(source['colliders'],original['cloth_id'],original['faces'])
    polygon = np.asarray(convex_polygon(table[table[:,2]==table[:,2].max(),:2]))
    edge = np.roll(polygon,-1,axis=0)-polygon
    ids = original['cloth_id']
    # This is the source's pre-episode below-rim cavity check, not a new score.
    cavity_min = np.array([-2.419,-1.013]);cavity_max = np.array([-2.061,-.737])
    def check(q):
        reasons=[];dirty=q[ids==0];clean=q[ids==1]
        if len(dirty):
            delta=dirty[:,None,:2]-polygon[None,:,:]
            cross=edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]
            if np.any(cross < -1e-12):reasons.append('old_cloth_outside_actual_tabletop')
            if dirty[:,2].min() < table[:,2].max():reasons.append('old_cloth_below_tabletop')
        if len(clean):
            below=clean[clean[:,2]<.45,:2]
            if np.any((below<cavity_min)|(below>cavity_max)):
                reasons.append('clean_cloth_outside_original_below_rim_cavity')
        wall = None
        if not reasons:
            wall=guard.check(q,tolerance=.00001)
            if not wall['valid']:reasons.append('original_basket_guard_rejected')
        return dict(accepted=not reasons,reasons=reasons,
            actual_tabletop_polygon_xy_m=polygon.tolist(),
            original_basket_guard=wall,
            self_shape_preserved_by_rigid_transform=True,
            native_settling_validated=False)
    return check


def prepare_bank(operator, task, destination, *, seed, count):
    """Save distinct geometry-filtered candidates and ready-to-load envelopes."""
    import numpy as np
    from . import config, launcher
    if type(count) is not int or not 1<=count<=30:
        raise ValueError('Prepare 1..30 explicit instances per task')
    reset._seed(seed);reset._seed(seed+count-1)
    value, workspace, _ = launcher.verify(operator,task,mounted=False)
    out=Path(destination)
    if out.resolve()!=out or not any(out.is_relative_to(r) for r in workspace.input_roots):
        raise ValueError('Immutable instance bank must stay in trusted input roots')
    out.mkdir(exist_ok=False)
    owner=value['owner'];roots=owner['directories'];files=owner['files']
    production=Path(roots['asset_root'])/'code/agent_bench/real2sim/production'
    catalog=production/'house_layouts.json'
    appearance=Path(roots['original_root'])/'code/agent_bench/real2sim/production/livingroom_environment_randomization.py'
    basket=Path(roots['asset_root'])/'code/tablecloth_workflow/native_teleop/lay_20260924/basket_profile_split_colliders.json'
    for path in (catalog,appearance,basket):workspace.input(path)
    with np.load(files['initial'],allow_pickle=False) as archive:
        original={k:archive[k] for k in archive.files}
    gate=geometry_gate(original,files['geometry'],basket,
                       original_native=config.native_directory(owner),owner_root=roots['owner_root'])
    rows=[]
    for current in range(seed,seed+count):
        initial,record=reset.choose_initial(original,task,current,geometry_gate=gate)
        folder=out/f'seed-{current:010d}';folder.mkdir()
        state=folder/'initial.npz';np.savez_compressed(state,**initial)
        record.update(baseline_sha256=reset.digest(files['initial']),initial_sha256=reset.digest(state),
                      source_hashes={str(p):reset.digest(p) for p in (catalog,appearance,basket,Path(files['geometry']))})
        record['layout'],record['appearance']=reset.scene_choices(catalog,appearance,task,current)
        reset.verify_initial(original,initial,task,record)
        record_path=folder/'variation.json'
        record_path.write_text(json.dumps(record,indent=2)+'\n')
        from copy import deepcopy
        launch=deepcopy(value)
        launch['variation']=dict(record=str(record_path),record_sha256=reset.digest(record_path),
                                 initial=str(state),initial_sha256=reset.digest(state))
        # The immutable, read-only baseline was fully verified once above.
        # Re-hashing gigabytes of identical assets for every seed would add no
        # preparation evidence. Verify each NEW sidecar here; execution still
        # performs the unchanged complete input admission before starting Isaac.
        reset.load_variation(launch['variation'],workspace)
        with (folder/'launch.json').open('x') as stream:
            json.dump(launch,stream,indent=2);stream.write('\n')
        rows.append(dict(seed=current,launch=str(folder/'launch.json'),launch_sha256=reset.digest(folder/'launch.json'),
                         initial_sha256=reset.digest(state),
                         layout=record['layout']['id'],colour=record['appearance']['style'],
                         attempt=record['attempt'],status='cpu_geometry_prepared'))
    if len({r['initial_sha256'] for r in rows})!=count:
        raise ValueError('Duplicate physical cloth initial states in instance bank')
    report=dict(schema='robotagentbench.cloth_instance_bank.v1',task_name=task,instances=rows,
                native_started=False,native_acceptance_inherited=False,automatic_runtime_resampling=False)
    (out/'index.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
