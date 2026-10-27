"""Fresh original-source physical child. Only explicit execution loads Isaac.

No GPU lease here: the non-physical supervisor holds the one common lease.
The source's default app shutdown stays unchanged, including fast shutdown.
"""
import json
import os
from pathlib import Path
import sys


def main():
    launch,output,boundary_text,expected=sys.argv[1:]
    from . import config as admission
    path=Path(launch)
    if admission.sha(path)!=expected:raise ValueError('Source-child launch changed')
    value=admission.read(path);owner_config=value['owner'];roots=owner_config['directories']
    sys.dont_write_bytecode=True
    sys.path[:0]=[roots['public_root'],roots['sdk_site']]
    from robotagentbench.simulation.workspace import NativeWorkspace
    boundary=json.loads(boundary_text)
    workspace=NativeWorkspace(tuple(map(Path,boundary['input_roots'])),Path(boundary['output_root']),Path(boundary['lock_root']))
    admission.validate(value,workspace,mounted=True)
    out=Path(output)
    if out.resolve()!=out or not out.is_dir()or not out.is_relative_to(workspace.output_root):raise ValueError('Owned created output required')
    # Save the RPC descriptor before any original SDK/code can print.
    rpc=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    original=Path(roots['original_root'])
    sys.path[:0]=[roots['sdk_site'],str(original/'code/agent_bench/experiments/physics_fluid'),
        str(original/'code/agent_bench'),str(admission.native_directory(owner_config))]
    import numpy as np
    selected=admission.selection(value['task_name']);files=owner_config['files']
    with np.load(files['initial'],allow_pickle=False)as archive:prepared={k:archive[k]for k in archive.files}
    cloth_reset=None
    if 'variation' in value:
        from .randomization import load_variation
        from .variation_native import NativeReset
        record,initial=load_variation(value['variation'],workspace)
        baseline=prepared
        with np.load(initial,allow_pickle=False) as archive:prepared={k:archive[k] for k in archive.files}
        cloth_reset=NativeReset(record=record,baseline=baseline,prepared=prepared,output=out)
    with np.load(files['reference_action'],allow_pickle=False)as archive:
        names=list(map(str,archive['joint_names']));first=archive['joint_target'][0].copy()
    protocol=admission.read(files['protocol'])
    for i,name in enumerate(names):
        if name in protocol['joint_targets']:first[i]=protocol['joint_targets'][name]
        if name.endswith(('gripper_l_outer_joint1','gripper_r_outer_joint1')):first[i]=.75
    if list(map(str,prepared['joint_names']))!=names or not np.array_equal(first,prepared['initial_joint_targets']):
        raise ValueError('Original frozen initial command recipe differs from prepared input')
    commands=out/'original_initial_commands.action.npz'
    np.savez_compressed(commands,joint_names=np.asarray(names),joint_target=np.tile(first,(65,1)),
        time_s=np.arange(65)/20,phase=np.full(65,'source_hold_diagnostic'))
    from . import bootstrap as recipe
    argv=recipe.source_argv(owner_config,selected,admission.read(files['trial']),out,commands)
    sys.path[:0]=[roots['owner_root'],roots['public_root'],roots['geometry_python_root']]
    from source_hooks import extend_source
    from source_owner import OriginalOwner
    from source_rpc import serve_original_owner
    def external_stream(owner,controls):
        return serve_original_owner(owner,controls,input_stream=sys.stdin,output_stream=rpc)
    owner=OriginalOwner(output=out,task_name=value['task_name'],prepared=prepared,
        setting=admission.read(files['setting']),policy=admission.read(files['policy']),protocol=protocol,
        source_root=Path(roots['asset_root']),camera_regression=False,
        isolated_camera_regression=True,sdk_site=roots['sdk_site'],session_handler=external_stream)
    entry=admission.native_directory(owner_config)/selected['entry']
    adapted=extend_source(entry.read_bytes(),entry.name,observations=False)
    if cloth_reset is not None:
        from .variation_native import extend_initialization
        adapted=extend_initialization(adapted)
    generated=out/'explicit_original_owner_source.py';generated.write_text(adapted)
    (out/'source_invocation.json').write_text(json.dumps(dict(argv=argv,entry_sha256=admission.sha(entry),
        extended_sha256=admission.sha(generated),original_source_edited=False,
        owner='original_source_v18_isolated',post_admission_action_source='external_robot22_stream_only',
        diagnostic_replay_intercepted_before_loop=True),indent=2)+'\n')
    sys.argv=argv
    try:
        exec(compile(adapted,str(generated),'exec'),dict(__name__='__main__',__file__=str(entry),NATIVE_OWNER=owner,
                                                      CLOTH_RESET=cloth_reset))
    except BaseException as exc:
        owner.report.update(status='error',error=repr(exc),evaluation_valid=False,score=None);owner.save()
        raise
    finally:
        rpc.flush()
    # The original app may fast-exit before reaching here. Supervisor validates
    # the saved owner report AND actual child exit, never this return alone.


if __name__=='__main__':main()
