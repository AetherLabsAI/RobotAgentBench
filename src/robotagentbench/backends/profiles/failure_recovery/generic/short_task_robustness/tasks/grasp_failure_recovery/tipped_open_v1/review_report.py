"""Evidence-only summary and original-image review page, no image editing."""
import hashlib
import html
import json
from pathlib import Path
from ..common import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review_remaining_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    plan=json.loads((OUT/'plan.json').read_text());rows=[]
    visual_path=OUT/'visual_review.json'
    visuals=json.loads(visual_path.read_text()).get('cases',{}) if visual_path.exists() else {}
    for row in plan['cases']:
        reassignment=OUT/'flex_queue'/f"dispatch{row['number']}.json"
        if reassignment.exists():row=dict(row,**{k:v for k,v in json.loads(reassignment.read_text()).items() if k in ('native_key','host','gpu')})
        record=dict(row);initial=ROOT/'samples'/row['sample_key']/'initial.json'
        report=ROOT/'native'/row['native_key']/'native_report.json';exit_path=ROOT/'native_commands'/row['native_key']/'exit.json'
        value=json.loads(report.read_text()) if report.exists() else {}
        native_exit=json.loads(exit_path.read_text()) if exit_path.exists() else None
        if initial.exists():
            cpu=json.loads(initial.read_text())
            if cpu['frozen_profile_id']!=row['profile']:raise ValueError('Actual scene is not requested distinct profile')
            record['initial_sha256']=digest(initial)
        score_initial=value.get('initial_result',{}).get('scores',{}).get('total')
        score_final=value.get('final_result',{}).get('scores',{}).get('total')
        head=report.parent/'after_hold_head.png';wrist=report.parent/'after_hold_left.png'
        passed=bool(native_exit and native_exit['returncode']==0 and value.get('status')=='completed'
            and not value.get('cleanup_errors') and value.get('original_native_initial_accepted')
            and value.get('hold_control_steps')==20 and score_initial==0 and score_final==0
            and value.get('authored_start_measurements',{}).get('after_hold',{}).get('hand_above_cup') is True
            and head.exists() and wrist.exists())
        record.update(status='native_initial_hold_pass_visual_pending' if passed else value.get('status','pending'),
            native_initial_hold_pass=passed,exit=native_exit,error=value.get('error'),cleanup_errors=value.get('cleanup_errors'),
            initial_score=score_initial,final_score=score_final,model_calls=value.get('model_calls'),
            measured=value.get('authored_start_measurements'),visual_reviewed=False,
            report=str(report.relative_to(ROOT)),head=str(head.relative_to(ROOT)),wrist=str(wrist.relative_to(ROOT)),
            image_sha256={p.name:digest(p) for p in (head,wrist) if p.exists()})
        visual=visuals.get(str(row['number']))
        if visual and visual['native_key']==row['native_key'] and visual['image_sha256']==record['image_sha256']:
            record['visual_reviewed']=True;record['visual_observations']=visual['observations']
            if passed:record['status']='native_initial_hold_pass_visually_reviewed'
        rows.append(record)
    result=dict(requested_count=7,distinct_profile_count=len({r['profile'] for r in rows}),
        native_initial_hold_pass=sum(r['native_initial_hold_pass'] for r in rows),
        normal_exit_count=sum(bool(r['exit'] and r['exit']['returncode']==0) for r in rows),
        visual_review_pending=not all(r['visual_reviewed'] for r in rows),source_tree='full_only',physics='cpu',model_recovery_tested=False,
        original_room_seed=plan['original_room_seed'],fresh_rooms_generated=False,rows=rows)
    write_json(OUT/'summary.json',result)
    if all(r['native_initial_hold_pass'] for r in rows):
        cards=[]
        for row in rows:
            m=row['measured']['after_hold'];gap=(m['hand_lowest_z_m']-m['cup_highest_z_m'])*100
            cards.append(f'''<section><h2>{row['number']:02d} · {html.escape(row['profile'])}</h2>
<p>原生初态 + hold 通过 · 手在杯子上方 {gap:.1f} cm · 开局/结束 0 分</p>
<a href="../{row['head']}"><img src="../{row['head']}" alt="第 {row['number']} 套原始头部相机渲染"></a>
<details><summary>左腕原图与原始报告</summary><a href="../{row['wrist']}"><img src="../{row['wrist']}" alt="第 {row['number']} 套原始左腕相机渲染"></a><p><a href="../{row['report']}">原始报告</a></p></details></section>''')
        page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>侧倒杯剩余七套初态</title>
<style>body{font:16px system-ui;margin:24px auto;padding:0 18px;max-width:1280px;background:#f5f6f7;color:#20252b}section{margin:30px 0}h1{font-size:26px}h2{font-size:18px;overflow-wrap:anywhere}img{display:block;width:100%;height:auto}summary{padding:12px 0;cursor:pointer}a{color:#165da8}p{line-height:1.6}</style>
<h1>侧倒杯：剩余七套初态</h1><p>CPU 物理 · 原相机原图 · 同一 401 房间中的七套不同配置。未做模型恢复动作测试。点图可看原图；杯子可能被手指局部遮挡，几何可见不等于无遮挡。</p>'''+''.join(cards)+'</html>'
        (OUT/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))


if __name__=='__main__':collect()
