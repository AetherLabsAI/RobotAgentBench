"""Build the review-site hub served at http://127.0.0.1:8764/ (site root: evaluation/tasks/preview/).

One page per data_requirement.csv family ("Task N"), hash-routed in a single index.html (#task6 …). Each family page
lists its subtasks as 【Task N 家族名 · 子任务 x.y · API id · 任务编号 x.y · 任务描述】 and links to the API detail page
when one exists (shelf_pick 4-x, coffee_table_place 5-x, tidy_sofa 6.*). Family preview directories are exposed as
symlinks next to index.html; nothing is copied. CPU only; rerun after editing any family page.
"""
import html, json, os, re, time
from pathlib import Path
from task_catalog import families, api_page_href, runtime_id, api_index, SUBTASK_RUNTIME_ID, FAMILY_DIR, REPO, RUNNER_TO_SERVER

TASKS = Path(__file__).resolve().parent
SITE = TASKS / 'preview'
LINKS = {'shelf_pick': 'shelf_pick', 'coffee_table_place': 'coffee_table_place', 'tidy_sofa': 'tidy_sofa', 'microwave_put': 'microwave_put',
         'task6_tidy_sofa': 'tidy_sofa', 'livingroom_cleanup': 'tidy_sofa',   # old names kept as aliases so stale/bookmarked links keep working
         'air_fryer': 'air_fryer', 'basket_place': 'basket_place', 'personal_item_pick': 'personal_item_pick', 'small_table_pick': 'small_table_pick',
         'tabletop_pick': 'tabletop_pick', 'trash_disposal': 'trash_disposal', 'refrigerator_take': 'refrigerator_take', 'dining_place': 'dining_place',
         'microwave_take': 'microwave_take', 'hang_hat': 'tidy_sofa/hang_hat', 'book_cloth': 'tabletop_pick/book_cloth'}
STATUS = json.loads((SITE / 'status.json').read_text()) if (SITE / 'status.json').exists() else dict(registry={}, runner_configs={}, runs=[], summary={}, families={})
RUNS_BY_TASK = {}
for _r in STATUS['runs']: RUNS_BY_TASK.setdefault(_r['task_id'], []).append(_r)


def reg_cell(rt):
    """登记状态 cell for a runtime id (server registration + runner config)."""
    if not rt: return "<span class='none'>—</span>"
    srv = RUNNER_TO_SERVER.get(rt, rt); reg = STATUS['registry'].get(srv)
    if not reg: return "<span class='pill pend'>未登记</span>"
    cfg = rt in STATUS['runner_configs'] or any(v['environment_task_id'] == srv for v in STATUS['runner_configs'].values())
    return (f"<span class='pill {'ok' if reg['ready'] else 'blk'}'>{'ready' if reg['ready'] else 'ready=False'}</span>"
            f"<div class='note'><code>{html.escape(str(reg['evaluator_id']))}</code>{'' if cfg else ' · 无 runner 配置'}</div>")


def result_cell(rt):
    """评测结果 cell: best valid run + counts, with an expandable list of every run and its videos."""
    srv = RUNNER_TO_SERVER.get(rt, rt) if rt else None
    runs = RUNS_BY_TASK.get(srv, []) if srv else []
    if not runs: return "<span class='none'>无评测记录</span>"
    s = STATUS['summary'].get(srv) or {}
    b = s.get('best')
    head = (f"<span class='pill {'ok' if b['success'] else 'blk'}'>{'成功' if b['success'] else '未成功'} {b['score']}/{b['max_score']}</span> "
            f"<span class='note'>{html.escape(b['kind'])} · {b['when']}</span>" if b else "<span class='pill err'>只有无效记录</span>")
    rows = []
    for r in sorted(runs, key=lambda r: -r['mtime']):
        vids = ' '.join(f"<a href='{m['href']}'>{'三视角' if v == 'compare3' else v}</a>" for v, m in sorted(r['media'].items()))
        st = "<span class='pill err'>无效</span>" if not r['valid'] else ("<span class='pill ok'>成功</span>" if r['success'] else "<span class='pill blk'>未成功</span>")
        rows.append(f"<tr><td>{html.escape(r['kind'])}</td><td class='id'>{r['when']}</td><td>{st} {r['score']}/{r['max_score']}</td><td>{html.escape(str(r['reason']))} · {r['time_s']} s</td>"
                    f"<td>{vids or '<span class=none>无</span>'}</td><td class='note'><code>{html.escape(r['path'])}</code></td></tr>")
    return (f"{head}<div class='note'>记录 {s.get('runs', len(runs))} · 有效 {s.get('valid', 0)} · 成功 {s.get('successes', 0)} · GPT/agent {s.get('agent_trials', 0)} · 有视频 {s.get('with_video', 0)}</div>"
            f"<details><summary class='note'>全部记录与视频</summary><div class='tbl'><table><thead><tr><th>类型</th><th>时间</th><th>结果</th><th>原因 · 仿真时长</th><th>视频</th><th>作业目录</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>")

CSS = """
:root{--primary:#7FA6E0;--on-primary:#fff;--primary-container:#EBF2FC;--on-primary-container:#2E4A73;--secondary-container:#E6F1FB;--on-secondary-container:#2B4562;--tertiary-container:#FFDDB3;--on-tertiary-container:#2A1800;
--error:#B3261E;--error-container:#F9DEDC;--success:#146C2E;--success-container:#C4EED0;--info:#5A86C8;--info-container:#D3E3FD;--surface:#FAFBFD;--surface-low:#F5F7FB;--surface-mid:#EFF3F9;--surface-high:#E8EDF5;
--on-surface:#1A1C1E;--on-surface-variant:#44474E;--outline:#74777F;--outline-variant:#C4C6D0;--shadow:0 1px 2px rgba(0,0,0,.08),0 2px 8px rgba(127,166,224,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--on-surface);font-family:"Google Sans","Roboto","Inter",system-ui,-apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;font-size:15px;line-height:1.7}
a{color:var(--info);text-decoration:none}a:hover{text-decoration:underline}
header.brand{background:#DCE9FA;color:#1F3B5E;padding:16px 28px;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;box-shadow:0 2px 6px rgba(127,166,224,.2)}
header.brand h1{margin:0;font-size:20px;font-weight:600;letter-spacing:.01em}header.brand .sub{color:#4A6790;font-size:13px}
header.brand .ext a{color:#1F3B5E;font-size:13px;margin-left:14px;background:rgba(255,255,255,.6);padding:4px 12px;border-radius:999px}
nav.fam-nav,nav.tabs{position:sticky;top:0;z-index:5;background:rgba(254,247,255,.95);backdrop-filter:blur(8px);border-bottom:1px solid var(--outline-variant);padding:10px 28px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
nav.fam-nav a,nav.tabs a,nav.tabs b{padding:5px 14px;border:1px solid var(--outline-variant);background:#fff;border-radius:999px;font-size:13px;color:var(--on-surface-variant);font-weight:500;transition:.15s}
nav.fam-nav a:hover,nav.tabs a:hover{background:var(--surface-high);color:var(--on-surface);text-decoration:none}
nav.fam-nav a.on,nav.tabs b{background:var(--secondary-container);border-color:var(--secondary-container);color:var(--on-secondary-container);font-weight:600}
nav.tabs .sum{margin-left:auto;color:var(--on-surface-variant);font-size:12.5px;border:0;background:none;padding:0}
main{max-width:1200px;margin:0 auto;padding:22px 28px 60px}
.page{display:none}.page.on{display:block}
h2.title{font-size:24px;font-weight:600;margin:6px 0 2px}h2.title small,h2 small{font-size:13px;color:var(--on-surface-variant);font-weight:500;margin-left:10px}
h2{font-size:24px;font-weight:600;margin:4px 0 2px}
h3{font-size:16px;font-weight:600;margin:26px 0 8px;padding-left:10px;border-left:4px solid var(--primary)}
.lead,.meta{color:var(--on-surface-variant);font-size:13.5px;margin:0 0 16px;overflow-wrap:anywhere}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 18px}.chip{background:var(--surface-mid);border:0;border-radius:10px;padding:8px 14px;font-size:13px;color:var(--on-surface-variant)}
.chip b{display:block;font-size:18px;line-height:1.2;color:#2E4A73}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.fam{background:#fff;border:1px solid var(--outline-variant);border-radius:16px;padding:18px 20px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:6px;color:var(--on-surface);transition:.15s}
.fam:hover{text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.12),0 6px 18px rgba(127,166,224,.12);transform:translateY(-1px)}
.fam h3{margin:0;padding:0;border:0;font-size:17px;color:#2E4A73}.fam .k{color:var(--on-surface-variant);font-size:12.5px}.fam .stats{display:flex;gap:14px;font-size:13px;margin-top:4px}
.fam .stats b{font-size:16px;color:var(--on-surface)}.fam .dir{font-family:"Roboto Mono",ui-monospace,Menlo,monospace;font-size:12px;color:var(--on-surface-variant)}
.fam .bar,.bar{height:8px;background:var(--surface-high);border-radius:6px;overflow:hidden;margin-top:6px}.fam .bar i,.bar i{display:block;height:100%;background:var(--primary);transition:width .6s}
.card,.tbl{background:#fff;border:1px solid var(--outline-variant);border-radius:16px;box-shadow:var(--shadow);overflow:hidden}
.tbl{overflow:auto}
.grid .card{padding:12px 16px;border-radius:14px}.card .note{font-size:12.5px}.card b{font-size:24px;display:block;line-height:1.25;color:#2E4A73}.card.alert b{color:var(--error)}.card.dim b{color:var(--outline)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}
th{background:var(--surface-mid);font-weight:600;font-size:13px;color:var(--on-surface-variant);text-align:left;padding:10px 12px;border-bottom:1px solid var(--outline-variant);white-space:nowrap}
td{padding:10px 12px;border-bottom:1px solid var(--surface-high);vertical-align:top}tbody tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:var(--surface-low)}tr.shared td{background:var(--tertiary-container)!important}
td.id{white-space:nowrap;font-family:"Roboto Mono",ui-monospace,Menlo,monospace;font-size:13px}
.api{display:inline-block;background:var(--primary-container);color:var(--on-primary-container);border:0;border-radius:8px;padding:1px 9px;font-weight:600;font-size:13px;font-family:"Roboto Mono",ui-monospace,Menlo,monospace}
.desc{font-weight:600}.identity{display:block;font-family:"Roboto Mono",ui-monospace,Menlo,"Noto Sans Mono CJK SC",monospace;font-size:12px;color:var(--on-surface-variant);margin-top:3px}
.note{color:var(--on-surface-variant);font-size:12.5px}.none{color:var(--outline)}.tag{display:inline-block;font-size:11.5px;padding:0 8px;border-radius:999px;background:var(--secondary-container);color:var(--on-secondary-container);margin-left:6px}
a.btn{display:inline-block;background:#9DBDE8;color:#14304F;border-radius:999px;padding:6px 14px;font-size:13px;font-weight:500;white-space:nowrap}a.btn:hover{background:#5A86C8;text-decoration:none;box-shadow:0 1px 3px rgba(0,0,0,.2)}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.ok{background:var(--success-container);color:var(--success)}.err{background:var(--error-container);color:var(--error)}.run{background:var(--info-container);color:var(--info)}.blk{background:var(--tertiary-container);color:var(--on-tertiary-container)}.pend{background:var(--surface-high);color:var(--on-surface-variant)}
.acc{font-family:"Roboto Mono",ui-monospace,Menlo,monospace;font-size:12px;white-space:nowrap}
code{font-family:"Roboto Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:var(--surface-mid);padding:1px 6px;border-radius:6px}
pre#log{font-size:12px;background:#1D1B20;color:#E6E0E9;border-radius:16px;padding:14px 16px;max-height:280px;overflow:auto;line-height:1.5}
footer{color:var(--on-surface-variant);font-size:12.5px;margin-top:30px}
"""


JS = """
function show(){var h=(location.hash||'#home').slice(1).split('/')[0];document.querySelectorAll('.page').forEach(p=>p.classList.toggle('on',p.id==='page-'+h));
document.querySelectorAll('.fam-nav a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+h));window.scrollTo(0,0)}
window.addEventListener('hashchange',show);show();
"""


def family_page(fam, detail_dirs):
    rows = []
    rts = [runtime_id(s['sub'], s['api']) for s in fam['subtasks']]
    for s in fam['subtasks']:
        api = s['api']; rt = runtime_id(s['sub'], api); href = api_page_href(rt) or (api_page_href(api) if api else None)
        shared = [f for f, x in _api_index.get(api, []) if f['id'] != fam['id']] if api else []
        ident = f"【Task {fam['id']} {fam['name']} · 子任务 {s['sub']} · API {api or '—'} · 任务编号 {s['sub']} · {s['description']}】"
        link = f"<a class='btn' href='{href}'>打开任务页</a>" if href else "<span class='none'>尚无任务页</span>"
        rt_cell = ('<span class=api>' + html.escape(api) + '</span>') if api else '<span class=none>—</span>'
        if rt and rt != api: rt_cell += f"<div class='note'>运行时 <code>{html.escape(rt)}</code></div>"
        share = ('<span class="tag">共用 API · ' + '、'.join(f"Task {f['id']}" for f in shared) + '</span>') if shared else ''
        note = f"<div class='note'>{html.escape(s['note'])}</div>" if s['note'] else ''
        rows.append(f"<tr class='{'shared' if shared else ''}'><td class='id'>{html.escape(s['sub'])}</td><td class='id'>{rt_cell}</td>"
                    f"<td><span class='desc'>{html.escape(s['description'])}</span>{share}<span class='identity'>{html.escape(ident)}</span>{note}</td>"
                    f"<td>{html.escape(s['target'])}</td><td>{reg_cell(rt)}</td><td>{result_cell(rt)}</td><td>{link}</td></tr>")
    folder = fam['dir']; exists = folder is not None
    dirs = sorted({d for d in re.findall(r'[a-z_]+', folder or '') if (TASKS / d).is_dir()} |
                  {STATUS['registry'][RUNNER_TO_SERVER.get(rt, rt)]['family'] for rt in rts if rt and STATUS['registry'].get(RUNNER_TO_SERVER.get(rt, rt), {}).get('family')})
    fam_links = ' '.join(f"<a class='btn' href='{d}/{'family.html' if (TASKS / d / 'preview/family.html').exists() else 'index.html'}'>{d} 家族说明与评测记录</a>" for d in dirs if (TASKS / d / 'preview').is_dir())
    if fam['id'] == 6: fam_links += " <a class='btn' href='hang_hat/index.html'>hang_hat（17-1）</a>"
    n_api = sum(1 for s in fam['subtasks'] if s['api']); n_page = sum(1 for rt in rts if api_page_href(rt))
    n_reg = sum(1 for rt in rts if rt and STATUS['registry'].get(RUNNER_TO_SERVER.get(rt, rt), {}).get('ready'))
    n_run = sum(1 for rt in rts if rt and RUNS_BY_TASK.get(RUNNER_TO_SERVER.get(rt, rt)))
    n_succ = sum(1 for rt in rts if rt and (STATUS['summary'].get(RUNNER_TO_SERVER.get(rt, rt)) or {}).get('successes'))
    chips = (f"<div class='chips'><div class='chip'>子任务<b>{len(fam['subtasks'])}</b></div><div class='chip'>有 API ID<b>{n_api}</b></div><div class='chip'>已登记 ready<b>{n_reg}</b></div>"
             f"<div class='chip'>有评测记录<b>{n_run}</b></div><div class='chip'>有成功回合<b>{n_succ}</b></div><div class='chip'>有任务页<b>{n_page}</b></div>"
             f"<div class='chip'>需求条数<b>{sum(int(s['need'] or 0) for s in fam['subtasks'])}</b></div><div class='chip'>仓库目录<b style='font-size:13px;font-family:ui-monospace,Menlo,monospace'>{('evaluation/tasks/' + folder) if exists else '尚未建立（接入时新建）'}</b></div></div>")
    return (f"<section class='page' id='page-task{fam['id']}'><h2 class='title'>Task {fam['id']} · {html.escape(fam['name'])}"
            f"<small>family {fam['id']} {html.escape(fam['slug'])}{' · 负责人 ' + html.escape(fam['owner']) if fam['owner'] else ''}</small></h2>"
            f"<p class='lead'>来源：data_requirement.csv 的 Task {fam['id']} 行；API ID 与任务描述来自 VLA_task.csv（任务编号 = 子任务ID）。黄底行表示该 API 与其他家族共用。"
            f"「登记」= 服务端 evaluation/registration.py 的登记与 ready 标志；「评测结果」= 目录内实际 task_evaluation.json 记录（服务端评分），展开可看每条记录与视频；定义与状态原文在各任务页。</p>{chips}<p>{fam_links}</p>"
            f"<div class='card'><table><thead><tr><th>子任务ID</th><th>API / 运行时 ID</th><th>任务</th><th>对象 / 目标位置</th><th>登记</th><th>评测结果（最高分）</th><th>详情</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>")


def catalog_json():
    """运行时任务 ID -> [{n, name, sub, desc}]，给 feedback.js 按 Task 分类导出反馈用。"""
    cat = {}
    for api, items in api_index().items():
        for fam, sub in items:
            cat.setdefault(api, []).append(dict(n=fam['id'], name=fam['name'], sub=sub['sub'], desc=sub['description']))
    fams = families()
    for sub_no, rt in SUBTASK_RUNTIME_ID.items():
        n = int(sub_no.split('.')[0]); fam = fams.get(n)
        if not fam: continue
        desc = next((x['description'] for x in fam['subtasks'] if x['sub'] == sub_no), '')
        rows = cat.setdefault(rt, [])
        if not any(r['sub'] == sub_no and r['n'] == n for r in rows):
            rows.append(dict(n=n, name=fam['name'], sub=sub_no, desc=desc))
    return json.dumps(cat, ensure_ascii=False)


def results_page():
    """#results: every server-registered task id (all families), registration, best/latest evaluation, links."""
    from task_catalog import identity_items
    rows = []
    def key(t):
        return [(0, int(x)) if x.isdigit() else (1, x) for x in re.split(r'[-._]', t)]
    for tid in sorted(STATUS['registry'], key=lambda t: (STATUS['registry'][t].get('family') or 'zzz', key(t))):
        reg = STATUS['registry'][tid]
        if tid in ('microwave_open_001', 'isaac_policy_execution', 'microwave_put.beef_initial_probe.v1'): continue
        ident = identity_items(tid); ident_s = ' '.join(f'【{x}】' for x in ident) if ident else ''
        subs = [k for k, v in __import__('task_catalog').SUBTASK_RUNTIME_ID.items() if RUNNER_TO_SERVER.get(v, v) == tid]
        if subs and not ident: ident_s = '【data_requirement 子任务 ' + '、'.join(subs) + '】'
        href = api_page_href(tid) or next((api_page_href(v) for v in RUNNER_TO_SERVER if RUNNER_TO_SERVER[v] == tid and api_page_href(v)), None)
        rows.append(f"<tr><td class='id'>{html.escape(reg.get('family') or '—')}</td><td class='id'><span class='api'>{html.escape(tid)}</span></td>"
                    f"<td><span class='desc'>{html.escape(str(reg['instruction']))}</span><span class='identity'>{html.escape(ident_s)}</span></td>"
                    f"<td>{reg_cell(tid)}</td><td>{result_cell(tid)}</td><td>{f'<a class=btn href={href}>任务页</a>' if href else '<span class=none>—</span>'}</td></tr>")
    return (f"<section class='page' id='page-results'><h2 class='title'>评测总表<small>服务端登记的全部任务 ID × 实际评测记录 · 扫描 {html.escape(STATUS.get('generated_at', '—'))}</small></h2>"
            f"<p class='lead'>数据来自 evaluation/tasks/scan_status.py：登记信息取自 arena_api.task_registry.default_registry()，评测记录取自 evaluation/tasks 下每个 task_evaluation.json（跟随批次目录到冻结快照）。"
            f"「最高分」只在有效记录里取；成功/未成功是服务端评分器的结论，是否合理请看视频。评测类型：GPT/agent 试验 = 真实模型驱动的回合；冻结批次回放 / API 回放 = 现有轨迹回放；初态短测 = 只做初态验收的短回合（分数通常为 0，不代表任务失败）。</p>"
            f"<div class='card'><table><thead><tr><th>家族目录</th><th>任务 ID</th><th>指令 · 身份</th><th>登记</th><th>评测结果（最高分）</th><th>详情</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>")


def main():
    global _api_index
    from task_catalog import api_index
    _api_index = api_index()
    fams = families()
    SITE.mkdir(exist_ok=True)
    for name, target in LINKS.items():
        link = SITE / name
        if (TASKS / target / 'preview').is_dir() and not link.exists(): link.symlink_to(f'../{target}/preview')
    n_reg_all = sum(1 for r in STATUS['registry'].values() if r['ready']); n_run_all = len(RUNS_BY_TASK); n_succ_all = sum(1 for v in STATUS['summary'].values() if v['successes'])
    nav = "<a href='#home'>总览</a><a href='#results'>评测总表</a>" + ''.join(f"<a href='#task{f['id']}'>Task {f['id']} {html.escape(f['name'])}</a>" for f in fams.values())
    cards = []
    for f in fams.values():
        rts = [runtime_id(s['sub'], s['api']) for s in f['subtasks']]
        n_api = sum(1 for s in f['subtasks'] if s['api']); n_page = sum(1 for rt in rts if api_page_href(rt))
        n_run = sum(1 for rt in rts if rt and RUNS_BY_TASK.get(RUNNER_TO_SERVER.get(rt, rt)))
        n_succ = sum(1 for rt in rts if rt and (STATUS['summary'].get(RUNNER_TO_SERVER.get(rt, rt)) or {}).get('successes'))
        pct = int(100 * n_run / max(1, len(f['subtasks'])))
        cards.append(f"<a class='fam' href='#task{f['id']}'><h3>Task {f['id']} · {html.escape(f['name'])}</h3><span class='k'>family {f['id']} {html.escape(f['slug'])}{' · ' + html.escape(f['owner']) if f['owner'] else ''}</span>"
                     f"<div class='stats'><span>子任务 <b>{len(f['subtasks'])}</b></span><span>有评测记录 <b>{n_run}</b></span><span>有成功回合 <b>{n_succ}</b></span><span>任务页 <b>{n_page}</b></span></div>"
                     f"<div class='bar'><i style='width:{pct}%'></i></div><span class='dir'>{('evaluation/tasks/' + f['dir']) if f['dir'] else '目录：尚未建立'}</span></a>")
    home = (f"<section class='page' id='page-home'><h2 class='title'>总览<small>按 data_requirement.csv 的一类任务分页</small></h2>"
            f"<p class='lead'>每个子任务标注【Task · 子任务ID · API ID · 任务编号 · 任务描述】，API ID 来自 VLA_task.csv。任务定义表、真机视频与 202 评测结果在各 API 的任务页。"
            f"状态扫描 {html.escape(STATUS.get('generated_at', '—'))}：服务端登记 ready 的任务 <b>{n_reg_all}</b> 个，有评测记录的任务 ID <b>{n_run_all}</b> 个（其中有成功回合 <b>{n_succ_all}</b> 个），评测记录共 <b>{len(STATUS['runs'])}</b> 条。进度条 = 有评测记录的子任务占比；逐任务明细见「评测总表」。<br><b>写反馈：</b>各任务页「202 评测视频」区每条评测记录下、以及每页的「任务整体反馈」都有输入框（保存在你自己的浏览器里，不写入仓库）；写完回到本页底部「审阅反馈」按 Task 分类导出成一份文档。</p>"
            f"<div class='grid'>{''.join(cards)}</div><div id='fb-export-host'></div></section>")
    pages = results_page() + ''.join(family_page(f, LINKS) for f in fams.values())
    (SITE / 'index.html').write_text(f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>任务审阅站</title><style>{CSS}</style></head><body>"
                                     f"<header class='brand'><h1>Robot Agent Arena · 任务审阅站</h1><span class='sub'>data_requirement.csv × VLA_task.csv · 生成 {time.strftime('%Y-%m-%d %H:%M')}</span>"
                                     f"<span class='ext'><a href='http://127.0.0.1:8765/'>批次进度页 8765 ↗</a></span></header>"
                                     f"<nav class='fam-nav'>{nav}</nav><main>{home}{pages}<footer>重建顺序：<code>../.venv-api/bin/python evaluation/tasks/scan_status.py</code>（扫描登记与评测记录、补编码视频）→ <code>python3 evaluation/tasks/build_family_pages.py</code>（通用家族页）→ 各家族自己的 <code>tools/build_preview.py</code> → <code>python3 evaluation/tasks/build_site.py</code>。</footer></main><script>window.ARENA_CATALOG={catalog_json()};</script><script>{JS}</script><script src='feedback.js'></script></body></html>")
    print('site written', SITE / 'index.html', 'families', len(fams))


if __name__ == '__main__':
    main()
