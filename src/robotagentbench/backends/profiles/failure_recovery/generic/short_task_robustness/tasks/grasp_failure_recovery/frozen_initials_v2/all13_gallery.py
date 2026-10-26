"""Organize unmodified real camera frames; never synthesize missing evidence."""
import hashlib
import html
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'frozen_initials_v2/all13_review_v1'


def font(size):
    for p in ('/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).is_file():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def main():
    summary=json.loads((ROOT/'frozen_initials_v2/batch/cup_profiles13_first_v2/summary.json').read_text())
    plan=json.loads((OUT/'plan.json').read_text());lookup={}
    for r in plan['reuse_completed']:lookup[r['profile']]=(ROOT/r['path'],'native',True)
    for lane,items in plan['lanes'].items():
        br=json.loads((OUT/lane/'batch_report.json').read_text())
        exitcode=json.loads((OUT/'commands'/lane/'exit.json').read_text())['returncode']
        if br['status']!='completed' or br['cleanup_errors'] or exitcode:raise ValueError('Batch not cleanly completed: '+lane)
        for r in items:lookup[r['profile']]=(OUT/lane/r['key'],r['mode'],False)
    rows=[]
    for ordinal,row in enumerate(summary['rows'],1):
        directory,mode,reused=lookup[row['profile']]
        report=directory/('native_report.json' if mode=='native' else 'static_report.json')
        value=json.loads(report.read_text())
        if value['cleanup_errors']:raise ValueError('Unclean case '+str(directory))
        valid=mode=='native' and value['status']=='completed' and value['original_initial_acceptance']['accepted'] and value['hold_control_steps']==20
        label='NATIVE PASS' if valid else 'REJECTED / STATIC ONLY' if mode=='static' else 'NATIVE INVALID'
        prefix='after_hold' if valid else 'static' if mode=='static' else 'settled'
        head=directory/(prefix+'_head.png');wrist=directory/(prefix+('_left.png' if row['hand']=='l' else '_right.png'))
        if not head.is_file() or not wrist.is_file():raise ValueError('Missing actual evidence '+row['profile'])
        rows.append(dict(index=ordinal,profile=row['profile'],seed=row['seed'],hand=row['hand'],gap_cm=row['gap_m']*100,
            label=label,native_pass=valid,reused=reused,report=str(report.relative_to(ROOT)),
            head=str(head.relative_to(ROOT)),wrist=str(wrist.relative_to(ROOT)),
            error=value.get('error'),images_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (head,wrist)}))
    pages=[]
    for offset in range(0,13,4):
        group=rows[offset:offset+4];sheet=Image.new('RGB',(1144,100+455*len(group)),'#f4f6f9');draw=ImageDraw.Draw(sheet)
        draw.text((20,14),'13 FROZEN CUP EMPTY-GRASP INITIALS',font=font(29),fill='#172334')
        draw.text((20,54),'Original head + active wrist cameras | Red = rejected, static diagnosis only',font=font(19),fill='#526071')
        for i,r in enumerate(group):
            y=100+i*455;color='#237444' if r['native_pass'] else '#a52a2a'
            draw.text((20,y),f"{r['index']:02}  {r['profile']}  |  {r['hand'].upper()}  seed {r['seed']}  gap {r['gap_cm']:.2f} cm",font=font(22),fill='#172334')
            draw.text((20,y+29),r['label']+(' (reused completed probe)' if r['reused'] else ''),font=font(19),fill=color)
            for column,key,size in ((20,'head',(640,400)),(680,'wrist',(444,251))):
                im=Image.open(ROOT/r[key]).convert('RGB');sheet.paste(im.resize(size,Image.Resampling.LANCZOS),(column,y+54))
        target=OUT/f'profiles_{offset+1:02}_{offset+len(group):02}.png'
        if target.exists():raise ValueError('Refusing to replace existing gallery')
        sheet.save(target);pages.append(target.name)
    result=dict(profile_count=len(rows),native_pass=sum(r['native_pass'] for r in rows),rejected_static=sum('STATIC' in r['label'] for r in rows),
        source='full_only',physics='cpu',model_calls=0,bank_changed=False,default_pool_filtered=False,rows=rows,pages=pages)
    (OUT/'review_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    cards=[]
    for r in rows:
        rel=lambda key:'../../'+r[key]
        cards.append(f'''<article><h2>{r['index']:02} {html.escape(r['profile'])}</h2>
<p class="{'pass' if r['native_pass'] else 'bad'}">{r['label']} · {r['hand'].upper()} · seed {r['seed']} · gap {r['gap_cm']:.2f} cm</p>
<div><a href="{rel('head')}"><img src="{rel('head')}" alt="Original head camera"></a><a href="{rel('wrist')}"><img src="{rel('wrist')}" alt="Original active wrist camera"></a></div>
<p><a href="{rel('report')}">Raw report</a> · Images are genuine renderer outputs. Static rejection is not physical acceptance.</p></article>''')
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>13 cup initial-state review</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#142234;max-width:1400px;margin:32px auto;padding:20px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}article div{display:grid;grid-template-columns:1.5fr 1fr;gap:15px}img{width:100%}h2{font-size:20px}.pass{color:#237444}.bad{color:#a52a2a}a{color:#175aab}</style>
<h1>13 cup empty-grasp initial states</h1><p>'''+f"{result['native_pass']} native passes · {result['rejected_static']} protected-overlap rejections (static images only). Full source; CPU physics; unchanged cameras and scoring. Recovery policy not tested."+'</p>'+''.join(cards))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
