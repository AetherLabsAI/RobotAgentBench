"""Prepare the fresh feedback run after171 validates terminal release; not a replay splice."""
from pathlib import Path
import json
p=Path(__file__).parent;n='mirrored_measured_lay_0174';source='continuous_natural_lay_0168'
s=(p/'native_same_prefix_contact_v45.py').read_text().replace('from same_prefix_live_lay_extension_v1 import LayExtension','from bimanual_lay_extension_v38 import LayExtension')
s=s.replace("parser.add_argument('--surface-self-filter-distance'","parser.add_argument('--raw-state-storage',type=Path,default=None,help='New native output only, never a checkpoint input')\n    parser.add_argument('--surface-self-filter-distance'",1)
s=s.replace('    args.output.mkdir(parents=True, exist_ok=False)',"""    args.output.mkdir(parents=True, exist_ok=False)
    if args.raw_state_storage is not None:
        assert args.raw_state_storage.is_absolute() and not args.raw_state_storage.exists() and args.raw_state_storage.parent.is_dir()
        (args.output/'full_state.npz').symlink_to(args.raw_state_storage)
        (args.output/'raw_state_storage.json').write_text(json.dumps(dict(kind='newly generated native output, not a checkpoint input',path=str(args.raw_state_storage),episode_state_restores=0),indent=2))""")
(p/'native_measured_mirrored_lay_v50.py').write_text(s)
for old,new in [('continuous168_direct_eye_v2.py','continuous174_direct_eye_v2.py'),('adjacent168_shortedge_pregrasp_v8.py','adjacent174_shortedge_pregrasp_v8.py'),('longedge168_left_longedge_insert_guard_v3.py','longedge174_left_longedge_insert_guard_v3.py'),('longedge168_left_longedge_probe_guard_v3.py','longedge174_left_longedge_probe_guard_v3.py')]:
 (p/new).write_text((p/old).read_text().replace(source,n))
s=(p/'mirrored_lay_transport_live_v1.py').read_text();old="if mode=='contact':centers=[[-1.85,-.4,.90],[-1.88,-.4,.90],[-1.90,-.4,.89]];frames=180";new="""if mode=='contact':
 free=ob['uv'][:,1]>.48;observed_low=float(np.quantile(ob['cloth'][free,2],.05));held_height=float((pp['L'][2]+pp['R'][2])/2);height=float(np.clip(held_height+.4505-.008-observed_low,.82,.94));centers=[[-1.85,-.4,height],[-1.88,-.4,height]];frames=180
 (r.ep/f'freeedge_height_target_{i:03d}.json').write_text(json.dumps(dict(actual_freeedge_z05=observed_low,actual_held_height=held_height,target_held_height=height,object_state_writes=0),indent=2))""";assert old in s;s=s.replace(old,new);(p/'adaptive_freeedge_lay_contact_live_v3.py').write_text(s)
cmd=json.loads((p/(source+'.command.json')).read_text());cmd=[x.replace(source,n).replace('native_nonintersecting_feedback_v42.py','native_measured_mirrored_lay_v50.py')for x in cmd];cmd+=['--surface-self-filter-distance','.001','--surface-max-linear-velocity','-1'];(p/(n+'.command.prepared.json')).write_text(json.dumps(cmd,indent=2))
(p/(n+'.plan.json')).write_text(json.dumps(dict(prepared_only=True,launch_condition='171 actual terminal flat+release gate passes; fresh confirmation still required',fresh_initial_state=True,object_state_writes=0,restored_or_spliced_physics_states=False,source_protocol=source,changes=['native automatic cloth velocity bound and1mm self-filter matching171/173','actual right corner feedback retained','left pregrasp, insertion and closure replanned from this episode actual cloth','go directly from confirmed left lift to mirrored same-height pad planes; skip old diagonal unfold/contact/backtrack','free-edge touchdown from actual hanging height, lower from actual table anchor','short holds and one global quintic motion easing'],production_count=0,owner_approved=False),indent=2))
print('PREPARED174; NOT LAUNCHED',flush=True)
