"""Evidence-only review of the approved eight, including shutdown limitations."""
import html
import io
import json
import unittest
from PIL import Image,ImageDraw
from ..common import HERE,owned,digest,write_json
from .exact8 import checked_bank,DEFAULT_BANK,IDS
from .all13_gallery import font


def main():
    out=owned(HERE/'frozen_initials_v2/verification/exact8_v6_visual_review2',fresh=True);out.mkdir(parents=True)
    bank=checked_bank();old=HERE/'frozen_initials_v2/all13_review_v1'
    plan=json.loads((old/'plan.json').read_text());where={r['profile']:HERE/r['path'] for r in plan['reuse_completed']}
    for lane,items in plan['lanes'].items():
        for r in items:where[r['profile']]=old/lane/r['key']
    checks=[]
    for p in bank['cases']['table_waste_empty_grasp']['profiles']:
        directory=where[p['id']];rp=directory/'native_report.json';r=json.loads(rp.read_text())
        if r['status']!='completed' or r['cleanup_errors'] or not r['original_initial_acceptance']['accepted'] or r['hold_control_steps']!=20:
            raise ValueError('Per-case native checks incomplete: '+p['id'])
        checks.append(dict(profile=p['id'],seed=r['seed'],report=str(rp.relative_to(HERE)),
            head=str((directory/'after_hold_head.png').relative_to(HERE)),wrist=str((directory/'after_hold_left.png').relative_to(HERE)),
            native_initial_accepted=True,hold_steps=20,scores=r['final_result']['scores'],
            cup_pose_unchanged=r['measurements']['settled']['target_transforms']==r['measurements']['after_hold']['target_transforms'],
            initial_sha256=r['initial_sha256'],manual_image_review=dict(
                head_target_complete=p['id']!='train_ep1491_f70_l',active_wrist_target_complete=True,
                observation=('Cup clipped at bottom of native head image; full cup visible in left wrist. No camera changes.'
                    if p['id']=='train_ep1491_f70_l' else 'Cup and empty closed gripper visible in original head and active wrist images.'))))
    app_exits={lane:json.loads((old/'commands'/lane/'exit.json').read_text()) for lane in plan['lanes']}
    names=['frozen_initials_v2.test_exact8','frozen_initials_v2.test_bank','frozen_initials_v1.test_bank',
        'expansion_v2.test_catalog','expansion_v2.test_ik_branches','expansion_v2.test_operating_station',
        'expansion_v2.test_empty_grasp','expansion_v2.test_arm_first_cup','expansion_v2.test_geometric_direction']
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['grasp_failure_recovery.'+n for n in names])
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(out/'unittest.log').write_text(stream.getvalue())
    frozen=json.loads((HERE/'public_component_sources.json').read_text())['task_sources']
    drift=[name for name,sha in frozen.items() if digest(HERE/name)!=sha]
    cpu=HERE/'frozen_initials_v2/batch/exact8_cpu16_v1/summary.json';cpu_data=json.loads(cpu.read_text())
    if cpu_data['accepted']!=16 or cpu_data['optimizer_calls']:raise ValueError('16 fresh CPU generations required')
    if {r['profile'] for r in cpu_data['rows']}!=set(IDS):raise ValueError('CPU profile coverage incomplete')
    smoke=HERE/'expansion_v2/candidates/exact8_default_smoke_v1/initial.json';sv=json.loads(smoke.read_text())
    if sv['frozen_profile_id'] not in IDS or 'exact8' not in sv['initialization_policy']:raise ValueError('Default route wrong')
    pages=[]
    for offset in (0,4):
        group=checks[offset:offset+4];sheet=Image.new('RGB',(1144,1920),'#f4f6f9');draw=ImageDraw.Draw(sheet)
        draw.text((20,16),'EXACT 8 TRAINING ARM / BODY POSTURES',font=font(29),fill='#142234')
        draw.text((20,58),'Actual head + left wrist images | Per-case native initial + hold checks passed',font=font(19),fill='#526071')
        for i,r in enumerate(group):
            y=100+i*455;draw.text((20,y),f"{offset+i+1:02} {r['profile']} | seed {r['seed']} | LEFT | empty closed gripper",font=font(22),fill='#142234')
            draw.text((20,y+29),('HEAD CROP: cup partly out of bottom; full cup in wrist' if not r['manual_image_review']['head_target_complete']
                else 'Cup supported / unheld / stable; 20 hold actions'),font=font(19),fill='#a56316' if not r['manual_image_review']['head_target_complete'] else '#237444')
            for x,key,size in ((20,'head',(640,400)),(680,'wrist',(444,251))):
                image=Image.open(HERE/r[key]).convert('RGB');sheet.paste(image.resize(size,Image.Resampling.LANCZOS),(x,y+54))
        path=out/f'exact8_page{offset//4+1}.png';sheet.save(path);pages.append(str(path.relative_to(HERE)))
    result=dict(selected_count=8,hands={'l':8,'r':0},bank_sha256=digest(DEFAULT_BANK),per_case_native_checks_passed=len(checks),
        native_app_shutdown_clean=False,application_exit_records=app_exits,manual_image_review_completed=8,
        visual_limitation='Profile train_ep1491_f70_l seed20 has a partially clipped cup in native head image; full cup visible in left wrist. CPU frustum check did not capture this discrepancy; no camera/geometry overrides or further profile filtering performed.',
        application_shutdown_limitation='Both multi-stage diagnostic applications segfaulted during final shutdown after all case checks and image captures; no clean-process acceptance claimed.',
        cpu_generations=dict(requested=16,accepted=16,summary_sha256=digest(cpu),online_ik_calls=0),
        default_smoke=dict(profile=sv['frozen_profile_id'],initial_sha256=digest(smoke),preparation_accepted=sv['preparation_accepted']),
        tests=dict(run=tests.testsRun,successful=tests.wasSuccessful()),protected_source_drift=drift,
        physics_parameters_changed=False,shared_registration_changed=False,model_calls=0,rows=checks,pages=pages)
    write_json(out/'review.json',result)
    rel=lambda key:'../../../'+key
    cards=[]
    for r in checks:
        cards.append(f'''<article><h2>{html.escape(r['profile'])}</h2><p>LEFT · seed {r['seed']} · per-case initial/hold checks passed</p>
<p>{html.escape(r['manual_image_review']['observation'])}</p><div><a href="{rel(r['head'])}"><img src="{rel(r['head'])}"></a><a href="{rel(r['wrist'])}"><img src="{rel(r['wrist'])}"></a></div><a href="{rel(r['report'])}">Raw native report</a></article>''')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Eight exact training cup initial states</title>
<style>body{font:16px system-ui;background:#f4f6f9;color:#142234;max-width:1400px;margin:30px auto;padding:20px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}article div{display:grid;grid-template-columns:1.5fr 1fr;gap:15px}img{width:100%}aside{padding:20px;background:#fff0d9}a{color:#175aab}</style>
<h1>Eight exact training arm/body postures — cup empty grasp</h1><p>8 left-hand profiles; no mirrored or IK-adapted arm configurations. Actual unchanged camera frames. Cup placement remains random within approved checks.</p>
<aside>All eight per-case native initial/hold checks passed. Profile 04 has a partly cropped cup in the head image (full cup visible in left wrist); CPU frustum validation missed this discrepancy. The two multi-stage diagnostic apps crashed during final shutdown; clean-process acceptance is NOT claimed. No recovery policy or model was tested.</aside>'''+''.join(cards))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','application_exit_records')},indent=2))
    if drift or not tests.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
