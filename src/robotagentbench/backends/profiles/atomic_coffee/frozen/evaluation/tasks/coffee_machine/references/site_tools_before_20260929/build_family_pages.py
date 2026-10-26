"""Generic review pages for task families that have no hand-built preview (air_fryer, basket_place, personal_item_pick,
small_table_pick, tabletop_pick, trash_disposal, refrigerator_take, dining_place, microwave_take, tidy_sofa/hang_hat).

For each family: evaluation/tasks/<fam>/preview/index.html with
  1. identity lines of every registered task id that belongs to the family (task_catalog.identity_html + registry status),
  2. the family README.md rendered as HTML (the README is the family's single source of definitions and status — not rewritten here),
  3. per task id: every evaluation run found by scan_status.py (kind, date, score, success/reason, sim time, videos, run directory),
  4. the other *.md documents of the family (rendered inline, collapsed) and mp4 evidence found under artifacts/.
Pure CPU; reads evaluation/tasks/preview/status.json (run scan_status.py first). Media are exposed via symlinks under preview/media/.
"""
import html, json, os, re, time
from pathlib import Path
from task_catalog import identity_html, api_index, vla_rows, SUBTASK_RUNTIME_ID

TASKS = Path(__file__).resolve().parent
SITE = TASKS / 'preview'
STATUS = json.loads((SITE / 'status.json').read_text())
# family dir -> (page title, sub path relative to evaluation/tasks, registered ids belonging to it)
GENERIC = {
    'air_fryer': ('Task 2 · 加热面包（空气炸锅）', 'air_fryer'),
    'basket_place': ('Task 5 · 放入收纳篮（14-x）', 'basket_place'),
    'personal_item_pick': ('Task 5 · 抓取随身物品（13-x）', 'personal_item_pick'),
    'small_table_pick': ('Task 4/7 · 小茶几抓取（11-x）', 'small_table_pick'),
    'tabletop_pick': ('Task 4/7 · 茶几桌面抓取（7-x）', 'tabletop_pick'),
    'trash_disposal': ('Task 7 · 垃圾投放（18-x）', 'trash_disposal'),
    'refrigerator_take': ('Task 9 · 从冰箱取物（26-1）', 'refrigerator_take'),
    'dining_place': ('Task 10 · 放到餐桌（10.2 / 29-x）', 'dining_place'),
    'microwave_take': ('Task 10 · 从微波炉取物（10.1）', 'microwave_take'),
    'hang_hat': ('Task 6 · 将帽子挂在衣架（17-1）', 'tidy_sofa/hang_hat'),
    'book_cloth': ('Task 8 · 把桌布上的书拿起放到旁边（20-2）', 'tabletop_pick/book_cloth'),
}
# 子项目目录：单列成页的（键）与其所属家族页应排除的目录
SUBPROJECT_DIRS = {'tabletop_pick': ['book_cloth', 'soft_cloth', 'clear_objects'], 'tidy_sofa': ['hang_hat']}
# 子项目页额外纳入的兄弟目录（没有独立任务 ID 的组件目录，其视频/文档并到这一页）
SUBPROJECT_EXTRA = {'book_cloth': ['tabletop_pick/soft_cloth', 'tabletop_pick/clear_objects']}
# families with their own hand-built preview/index.html: only a companion family.html (README + docs + runs + media) is generated
DOCS_ONLY = {
    'shelf_pick': ('Task 3 · 零食架抓取（4-x）家族说明与评测记录', 'shelf_pick'),
    'coffee_table_place': ('Task 3/10 · 茶几放置（5-x / 30-x）家族说明与评测记录', 'coffee_table_place'),
    'microwave_put': ('Task 9 · 放入微波炉（9.2）家族说明与评测记录', 'microwave_put'),
    'tidy_sofa': ('Task 6 · 整理沙发家族说明与评测记录', 'tidy_sofa'),
}


# ---------- minimal Markdown -> HTML (headings, tables, lists, fenced code, paragraphs, inline code/bold/links) ----------
def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', lambda m: f"<a href='{m.group(2)}'>{m.group(1)}</a>", s)
    return s


def md_to_html(text, base_href=''):
    out, i, lines = [], 0, text.splitlines()
    def fix_link(m):  # only http links and the family's own *.md (exposed as preview/docs/<name>) stay clickable; other repo paths become text
        href = m.group(2)
        if re.match(r'^(https?:|#)', href): return m.group(0)
        if re.match(r'^[A-Za-z0-9_.-]+\.md$', href): return f"[{m.group(1)}](docs/{href})"
        return f"{m.group(1)}（`{href}`）"
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith('```'): buf.append(lines[j]); j += 1
            out.append('<pre><code>' + html.escape('\n'.join(buf)) + '</code></pre>'); i = j + 1; continue
        m = re.match(r'^(#{1,6})\s+(.*)', ln)
        if m:
            lvl = min(len(m.group(1)) + 1, 6); out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>"); i += 1; continue
        if re.match(r'^\s*[-*_]{3,}\s*$', ln): out.append('<hr>'); i += 1; continue
        if ln.startswith('>'):
            q = []
            while i < len(lines) and lines[i].startswith('>'): q.append(lines[i].lstrip('> ').rstrip()); i += 1
            out.append(f"<blockquote>{inline(re.sub(fix_link_re, fix_link, ' '.join(q)))}</blockquote>"); continue
        if ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'): rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
            body = [r for r in cells if not all(re.match(r'^:?-{2,}:?$', c) for c in r if c) or not r]
            if not body: continue
            head, rest = body[0], body[1:]
            out.append("<div class='tbl'><table><thead><tr>" + ''.join(f"<th>{inline(re.sub(fix_link_re, fix_link, c))}</th>" for c in head) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f"<td>{inline(re.sub(fix_link_re, fix_link, c)).replace('&lt;br&gt;', '<br>')}</td>" for c in r) + '</tr>' for r in rest) + '</tbody></table></div>')
            continue
        if re.match(r'^\s*([-*+]|\d+[.)])\s+', ln):
            items = []
            while i < len(lines) and re.match(r'^\s*([-*+]|\d+[.)])\s+', lines[i]):
                item = re.sub(r'^\s*([-*+]|\d+[.)])\s+', '', lines[i]); i += 1
                while i < len(lines) and lines[i].startswith('  ') and not re.match(r'^\s*([-*+]|\d+[.)])\s+', lines[i]): item += ' ' + lines[i].strip(); i += 1
                items.append(f"<li>{inline(re.sub(fix_link_re, fix_link, item))}</li>")
            out.append('<ul>' + ''.join(items) + '</ul>'); continue
        if not ln.strip(): i += 1; continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '```')) and not re.match(r'^\s*([-*+]|\d+[.)])\s+', lines[i]):
            para.append(lines[i].strip()); i += 1
        out.append(f"<p>{inline(re.sub(fix_link_re, fix_link, ' '.join(para)))}</p>")
    return '\n'.join(out)


fix_link_re = re.compile(r'\[([^\]]+)\]\(([^)\s]+)\)')


def pill(run):
    if not run.get('valid'): return "<span class='pill err'>无效</span>"
    if run.get('success'): return "<span class='pill ok'>成功</span>"
    return f"<span class='pill blk'>未成功</span>"


def run_row(r, site_prefix):
    vids = ' '.join(f"<a class='btn' href='{site_prefix}{m['href']}'>{'三视角' if v == 'compare3' else v}</a>" for v, m in sorted(r['media'].items()))
    if r.get('video_source') == 'encoded_from_frames': vids += " <span class='note'>（由逐帧 PNG 编码）</span>"
    step0 = f"<a href='{site_prefix}{r['step0']}'>初态图</a>" if r.get('step0') else ''
    rnd = r.get('randomization'); rnd_s = ''
    if isinstance(rnd, dict): on = [k for k, v in rnd.items() if v]; rnd_s = ('随机：' + '、'.join(on)) if on else '固定初态'
    notes = ' '.join(f"<code>{html.escape(Path(n).name)}</code>" for n in r.get('notes', [])[:3])
    return (f"<tr><td>{html.escape(r['kind'])}</td><td class='id'>{r['when']}</td><td>{pill(r)} {r['score']}/{r['max_score']}</td>"
            f"<td>{html.escape(str(r['reason']))}<div class='note'>{r['time_s']} s · {r.get('steps') or '—'} 步 · {rnd_s}</div></td>"
            f"<td>{vids or '<span class=none>无视频</span>'} {step0}</td><td class='note'><code>{html.escape(r['path'])}</code> {notes}</td></tr>")


def runs_table(runs, site_prefix='../'):
    if not runs: return "<p class='none'>未找到该任务的 task_evaluation.json 评测记录。</p>"
    return ("<div class='card'><table><thead><tr><th>类型</th><th>时间</th><th>结果</th><th>结束原因</th><th>视频</th><th>作业目录</th></tr></thead><tbody>"
            + ''.join(run_row(r, site_prefix) for r in sorted(runs, key=lambda r: -r['mtime'])) + '</tbody></table></div>')


def reg_line(tid):
    reg = STATUS['registry'].get(tid)
    if not reg: return "<span class='pill pend'>未登记</span>"
    cfg = [k for k, v in STATUS['runner_configs'].items() if v['environment_task_id'] == tid or k == tid]
    return (f"<span class='pill {'ok' if reg['ready'] else 'blk'}'>{'已登记 ready' if reg['ready'] else '已登记 ready=False'}</span> "
            f"<span class='note'>evaluator <code>{html.escape(str(reg['evaluator_id']))}</code> · {reg['max_steps']} 步 · runner 配置 {', '.join(cfg) or '无'} · 指令：{html.escape(str(reg['instruction']))}</span>")


def family_media(fam_dir, out_dir):
    """mp4 files under the family (artifacts etc., not references/preview) exposed via preview/media/ symlinks."""
    media = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(fam_dir, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen: dirnames[:] = []; continue
        seen.add(real)
        dirnames[:] = [d for d in dirnames if d not in ('references', 'preview', '__pycache__', 'runtime', 'cache', 'nv', 'pkg')]
        for f in filenames:
            if f.endswith('.mp4'): media.append(Path(dirpath) / f)
    media.sort(key=lambda p: -p.stat().st_mtime)
    rows = []
    mdir = out_dir / 'media'; mdir.mkdir(exist_ok=True)
    for p in media[:60]:
        rel = p.relative_to(fam_dir).as_posix()
        link = mdir / rel.replace('/', '__')
        if not link.exists(): link.symlink_to(os.path.relpath(p, mdir))
        rows.append(f"<tr><td class='id'>{time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime))}</td><td><code>{html.escape(rel)}</code></td>"
                    f"<td>{p.stat().st_size // 1024} KB</td><td><a class='btn' href='media/{html.escape(link.name)}'>播放</a></td></tr>")
    return rows


def build_docs_page(fam, title, sub, filename='index.html'):
    fam_dir = TASKS / sub; out_dir = fam_dir / 'preview'; out_dir.mkdir(exist_ok=True)
    ids = [tid for tid, r in STATUS['registry'].items() if (r['family'] == fam) or (fam == 'hang_hat' and r.get('sub_dir') == 'hang_hat')]
    if fam == 'hang_hat': ids = ['17-1']
    if fam == 'microwave_take': ids = ['10.1']
    # subtask numbers (data_requirement) mapped to runtime ids without a CSV API id (air fryer / trash / 9.2 / 10.x)
    extra = {tid: sub_no for sub_no, tid in SUBTASK_RUNTIME_ID.items()}
    runs = [r for r in STATUS['runs'] if r['family'] == sub.split('/')[0] and (fam != 'hang_hat' or r['task_id'] == '17-1') and (fam != 'tidy_sofa' or r['task_id'] != '17-1')]
    site = '../'   # every family preview dir is served as <site-root>/<fam>/ (symlink), so the site root is one level up in URL space
    docs_dir = out_dir / 'docs'; docs_dir.mkdir(exist_ok=True)
    for md in fam_dir.glob('*.md'):
        lnk = docs_dir / md.name
        if not lnk.exists(): lnk.symlink_to(os.path.relpath(md, docs_dir))
    parts = [f"<h2 class='title'>{html.escape(title)}<small>evaluation/tasks/{html.escape(sub)}</small></h2>",
             "<p class='lead'>本页由 evaluation/tasks/build_family_pages.py 从该家族目录的 README.md、其他说明文档和实际评测记录自动生成，只汇总不改写；"
             "定义、规则与状态以 README 原文为准。评测记录来自目录内每个 <code>task_evaluation.json</code>（服务端评分结果），视频为该回合的三路相机录像（没有 mp4 时由逐帧 PNG 编码）。</p>"]
    # per task id
    parts.append("<h3>任务身份、登记状态与评测记录</h3>")
    for tid in sorted(ids, key=lambda x: [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r'[-._]', x)]):
        ident = identity_html(tid) if tid in api_index() else (f"<p class='identity'>【运行时任务 ID {html.escape(tid)}" + (f" · 对应 data_requirement 子任务 {html.escape(extra[tid])}" if tid in extra else '') + "】</p>")
        trs = [r for r in runs if r['task_id'] == tid]
        s = STATUS['summary'].get(tid)
        summ = (f"共 {s['runs']} 次评测记录（有效 {s['valid']}，成功 {s['successes']}，GPT/agent 试验 {s['agent_trials']}，有视频 {s['with_video']}）；最高分 {s['best']['score']}/{s['best']['max_score']}（{s['best']['kind']}，{s['best']['when']}）" if s and s['best'] else '尚无带评分的评测记录')
        parts.append(f"<div class='card' style='padding:14px 18px;margin:10px 0' id='{html.escape(tid)}'><h4 style='margin:0'>任务 <span class='api'>{html.escape(tid)}</span></h4>{ident}<p>{reg_line(tid)}</p><p class='note'>{summ}</p>{runs_table(trs, site)}</div>")
    other = [r for r in runs if r['task_id'] not in ids]
    if other:
        parts.append("<h3>目录内其他任务 ID 的评测记录</h3>" + runs_table(other, site))
    # README
    readme = fam_dir / 'README.md'
    if readme.exists():
        parts.append(f"<h3>README.md（家族说明、定义与状态原文）<small>{time.strftime('%Y-%m-%d %H:%M', time.localtime(readme.stat().st_mtime))}</small></h3><div class='card doc'>{md_to_html(readme.read_text(), '../')}</div>")
    else:
        parts.append("<h3>README.md</h3><p class='none'>该目录没有 README.md；规则见 docs/TASK_EVALUATION.md 与 docs/MICROWAVE_PHYSICS.md（微波炉取物 10.1）。</p>")
    # other docs
    docs = sorted([p for p in fam_dir.glob('*.md') if p.name != 'README.md'], key=lambda p: -p.stat().st_mtime)
    if docs:
        parts.append(f"<h3>其他说明文档（{len(docs)}，按修改时间倒序，点开阅读）</h3>")
        for p in docs:
            parts.append(f"<details class='card doc' style='margin:8px 0;padding:8px 16px'><summary><b>{html.escape(p.name)}</b> <span class='note'>{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))} · {p.stat().st_size // 1024} KB</span></summary>{md_to_html(p.read_text(), '../')}</details>")
    # media
    rows = family_media(fam_dir, out_dir)
    if rows:
        parts.append(f"<h3>目录内的视频文件（{len(rows)} 条，最新在前；含来源/诊断/试验视频，性质以所在目录为准）</h3><div class='card'><table><thead><tr><th>时间</th><th>路径（相对家族目录）</th><th>大小</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    page = (f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<link rel='stylesheet' href='{site}arena.css'><style>main{{max-width:1200px;margin:0 auto;padding:20px 28px 60px}}.doc{{padding:12px 20px;overflow-x:auto}}.doc h2{{font-size:20px}}.doc h3{{font-size:17px;border:0;padding:0}}"
            f".doc table{{font-size:13px}}.doc pre{{background:#1D1B20;color:#E6E0E9;padding:10px 12px;border-radius:10px;overflow:auto;font-size:12px}}h4 .api{{font-size:15px}}</style></head><body>"
            f"<header class='brand'><h1>Robot Agent Arena · 任务审阅站</h1><span class='sub'>{html.escape(title)} · 生成 {time.strftime('%Y-%m-%d %H:%M')}</span><span class='ext'><a href='{site}index.html'>← 总览</a></span></header>"
            f"<main>{''.join(parts)}<footer>生成器 evaluation/tasks/build_family_pages.py；先跑 scan_status.py 再跑本脚本，最后 build_site.py。</footer></main></body></html>")
    (out_dir / filename).write_text(page)
    print(fam, '->', out_dir / filename, len(ids), 'ids', len(runs), 'runs')


TEMPLATE = TASKS / 'shelf_pick/preview/template.html'   # same page form as shelf_pick 4-x: single file, hash-routed, one task per page


def template_shell():
    """shelf_pick template with its hard-coded 4-1 section removed; keeps CSS, topbar, pager, TOC and lazy-video JS."""
    t = TEMPLATE.read_text()
    a = t.index('<section class="page"'); b = t.index('__PAGES__')
    t = t[:a] + t[b:]
    t = t.replace('零食架抓取任务批量审阅 4-1 … 4-14', '__TITLE__').replace("' · 零食架批量审阅'", "' · __SHORT__'").replace('零食架抓取任务批量审阅', '__TITLE__')
    t = re.sub(r'<p class="note" style="margin:2px 0 0">批次：.*?</p>', '__BATCH_NOTE__', t, flags=re.S)
    return t


def lazy_video(href, label, api, frames=None, fps=None, poster=None):
    meta = f"<p class='meta'>{frames} 帧 · {fps} fps</p>" if frames else ''
    pst = f" data-poster='{poster}'" if poster else ''
    return (f"<div class='episode'><h3>{html.escape(label)}</h3><video class='lazy-video' controls playsinline preload='none'{pst} aria-label='{api} {html.escape(label)}'><source data-src='{href}' type='video/mp4'></video>"
            f"<button class='load-video' type='button'>播放视频</button><p class='note video-status' aria-live='polite'>点击播放后才加载。也可直接打开 <a href='{href}'>MP4</a>。</p>{meta}</div>")


def source_videos(fam_dir, out_dir, keys):
    """来源仿真视频 candidates: mp4 under references/sources/source/resources whose path mentions one of `keys` (or any, when keys is None).
    Grouped per directory = one source episode; head/top_head first. Exposed via preview/media symlinks."""
    groups = {}
    for sub in ('references', 'sources', 'source', 'resources'):
        d = fam_dir / sub
        if not d.is_dir(): continue
        for p in d.rglob('*.mp4'):
            rel = p.relative_to(fam_dir).as_posix()
            if keys is not None and not any(k in rel.split('/') or f'/{k}/' in f'/{rel}' or k in p.parent.name for k in keys): continue
            groups.setdefault(p.parent.as_posix() if p.name != 'review.mp4' else (p.parent / 'videos').as_posix(), []).append(p)
    mdir = out_dir / 'media'; mdir.mkdir(exist_ok=True)
    eps = []
    for gdir, files in sorted(groups.items()):
        views = {}
        for p in files:
            n = p.stem.lower()
            v = 'head' if n in ('head', 'top_head', 'review') or 'head' in n else 'left' if 'left' in n else 'right' if 'right' in n else 'compare3' if 'compare' in n else n
            link = mdir / p.relative_to(fam_dir).as_posix().replace('/', '__')
            if not link.exists(): link.symlink_to(os.path.relpath(p, mdir))
            views.setdefault(v, f'media/{link.name}')
        eps.append(dict(dir=Path(gdir).relative_to(fam_dir).as_posix() if gdir.startswith(str(fam_dir)) else gdir, views=views))
    return eps


def definition_table(path, tid):
    """Render the '## <tid>' section of DEFINITION_TABLE.md as the shelf_pick-style 3-column definition table."""
    if not path.exists(): return ''
    lines = path.read_text().splitlines(); rows = []; on = False
    for ln in lines:
        if ln.startswith('## '): on = ln[3:].strip() == tid; continue
        if on and ln.startswith('|'): rows.append([c.strip() for c in ln.strip().strip('|').split('|')])
    if len(rows) < 3: return ''
    head = "<tr>" + ''.join(f"<th scope='col'>{inline(c)}</th>" for c in rows[0]) + "</tr>"
    body = ''.join("<tr>" + ''.join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[2:])
    return f"<table class='definition'><thead>{head}</thead><tbody>{body}</tbody></table>"


def excl_dirs(base):
    """扫描 base 时要跳过的子项目目录（它们有自己的页面）。"""
    out = set()
    for parent, subs in SUBPROJECT_DIRS.items():
        for d in subs:
            q = TASKS / parent / d
            if os.path.realpath(q) != os.path.realpath(base): out.add(os.path.realpath(q))
    return out


def other_videos(dirs, out_dir, used, limit=60):
    """目录里未被任何评测记录使用的 mp4（诊断 / 组件 / 历史），按目录分组、最新在前。"""
    vids = []
    seen = set()
    for base in dirs:
        if not base.is_dir(): continue
        for dirpath, dirnames, files in os.walk(base, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in seen: dirnames[:] = []; continue
            seen.add(real)
            skip = ('preview', '__pycache__', 'runtime', 'cache', 'nv', 'pkg')
            dirnames[:] = [d for d in dirnames if d not in skip and os.path.realpath(Path(dirpath) / d) not in excl_dirs(base)]
            for f in files:
                if not f.endswith('.mp4'): continue
                q = Path(dirpath) / f
                if os.path.realpath(q) in used: continue
                vids.append((base, q))
    if not vids: return ''
    mdir = out_dir / 'media'; mdir.mkdir(exist_ok=True)
    vids.sort(key=lambda bq: -bq[1].stat().st_mtime)
    groups = {}
    for base, q in vids[:limit]:
        rel = os.path.relpath(q, TASKS)
        link = mdir / rel.replace('/', '__')
        if not link.exists(): link.symlink_to(os.path.relpath(q, mdir))
        groups.setdefault(os.path.relpath(q.parent, TASKS), []).append((q, f'media/{link.name}'))
    rows = ''
    for gdir, items in sorted(groups.items(), key=lambda kv: -max(q.stat().st_mtime for q, _ in kv[1])):
        when = time.strftime('%Y-%m-%d %H:%M', time.localtime(max(q.stat().st_mtime for q, _ in items)))
        links = ' · '.join(f"<a href='{href}'>{html.escape(q.stem)}</a>（{q.stat().st_size // 1024} KB）" for q, href in sorted(items, key=lambda x: x[0].name))
        rows += f"<tr><td class='id'>{when}</td><td><code>{html.escape(gdir)}</code><div class='note'>{links}</div></td></tr>"
    more = f"<p class='note'>共 {len(vids)} 条，这里列出最新 {min(len(vids), limit)} 条。</p>" if len(vids) > limit else ''
    return (f"<div id='othervid'><h2>其他视频（诊断 / 组件 / 历史，未绑定评测记录）</h2>"
            f"<p class='note'>家族目录（含子项目）里存在、但没有对应服务端评分记录的 mp4：初态/接触诊断、来源对照、组件实验等。"
            f"点链接在新标签页播放；性质以所在目录和该目录的说明文档为准，不是 202 评测结论。</p>{more}"
            f"<div class='card'><table><thead><tr><th>最新时间</th><th>目录 · 文件</th></tr></thead><tbody>{rows}</tbody></table></div></div>")


def readme_intro(readme):
    """Leading prose of the README (before the first table / second heading) for the 这个任务具体要做什么 block."""
    if not readme.exists(): return ''
    lines = readme.read_text().splitlines()[1:]
    buf = []
    for ln in lines:
        if ln.startswith('|') or (ln.startswith('## ') and any(x.strip() for x in buf)): break
        if ln.startswith('#'): continue
        buf.append(ln)
    return md_to_html('\n'.join(buf).strip())


def task_page(fam, sub, tid, runs, fam_dir, out_dir, readme, docs, single, others=''):
    reg = STATUS['registry'].get(tid, {})
    title = f"{tid} · {reg.get('instruction') or tid}"
    ident = identity_html(tid) if tid in api_index() else f"<p class='identity'>【运行时任务 ID {html.escape(tid)}" + (f" · data_requirement 子任务 {html.escape('、'.join(k for k, v in SUBTASK_RUNTIME_ID.items() if RUNNER_TO_SERVER_LOCAL.get(v, v) == tid))}" if any(RUNNER_TO_SERVER_LOCAL.get(v, v) == tid for v in SUBTASK_RUNTIME_ID.values()) else '') + "】</p>"
    s = STATUS['summary'].get(tid)
    summ = (f"共 {s['runs']} 条服务端评分记录（有效 {s['valid']}，成功 {s['successes']}，GPT/agent 试验 {s['agent_trials']}，有视频 {s['with_video']}）；最高分 {s['best']['score']}/{s['best']['max_score']}（{s['best']['kind']}，{s['best']['when']}）" if s and s['best'] else '目录内没有该任务的服务端评分记录（task_evaluation.json）')
    status_badge = (f"<span class='status'>登记 ready · {html.escape(str(reg.get('evaluator_id')))} · {reg.get('max_steps')} 步</span>" if reg.get('ready') else
                    f"<span class='status draft'>{'登记 ready=False' if reg else '未登记'}{(' · ' + html.escape(str(reg.get('evaluator_id')))) if reg else ''}</span>")
    keys = None if single else [tid] + [k for k, v in SUBTASK_RUNTIME_ID.items() if v == tid] + [sd['sub'] for fam_, sd in api_index().get(tid, [])]
    srcs = source_videos(fam_dir, out_dir, keys)
    # 1 来源仿真视频
    if srcs:
        cards = ''
        for i, ep in enumerate(srcs[:5]):
            v = ep['views']
            hands = ''.join(f"<div class='hand-view'><h4>{lab}</h4><video class='lazy-video' controls playsinline preload='none'><source data-src='{v[c]}' type='video/mp4'></video><button class='load-video' type='button'>播放{lab}</button><p class='note video-status' aria-live='polite'>点击播放后才加载。</p></div>"
                            for c, lab in (('left', '左手相机'), ('right', '右手相机'), ('compare3', '三视角拼接')) if c in v)
            main_v = v.get('head') or next(iter(v.values()))
            cards += (f"<div class='episode'><h3>来源视频 {i+1}/{len(srcs[:5])} · <code>{html.escape(ep['dir'])}</code></h3>"
                      f"<video class='lazy-video' controls playsinline preload='none'><source data-src='{main_v}' type='video/mp4'></video><button class='load-video' type='button'>播放来源视频</button><p class='note video-status' aria-live='polite'>点击播放后才加载。也可直接打开 <a href='{main_v}'>MP4</a>。</p>"
                      + (f"<details class='hand-views-wrap'><summary>其他视角（可选）</summary><div class='hand-views'>{hands}</div></details>" if hands else '') + "</div>")
        sim_src = (f"<div id='simsrc-{tid}'><h2>来源仿真视频 · 主参考（{min(len(srcs), 5)}/5）</h2><p class='note'>这是家族目录 references/sources 里保存的来源方视频（来源方渲染或来源仿真/真机拼接，性质以目录名为准），不是 202 回放，也不代表 202 验收。"
                   f"{'不足 5 条：来源方只交付了这些；缺口如实保留，不用 202 回放冒充。' if len(srcs) < 5 else ''}</p><div class='episodes'>{cards}</div></div>")
    else:
        sim_src = f"<div id='simsrc-{tid}'><h2>来源仿真视频 · 主参考（0/5）</h2><p class='status draft'>家族目录里没有找到该任务的来源渲染视频（references/sources 下无 mp4）；来源轨迹/状态文件见 README 的来源节。缺口如实保留。</p></div>"
    # 2 真机视频 (none of these families fetched real videos)
    real = (f"<div id='media-{tid}'><details class='real-fold'><summary><h2 style='display:inline'>真机视频 · 头部相机（0/3，默认折叠）</h2></summary>"
            f"<p class='note'>本家族尚未按 L2 record task_id 取真机视频；如需补齐按 review-pages.md 的真机取法处理。</p></details></div>")
    # 3 202 评测视频 (合成视频检查节): success -> head only; otherwise reason + three views
    jobs = ''
    for i, r in enumerate(sorted(runs, key=lambda r: -r['mtime'])):
        ok = bool(r['valid']) and bool(r['success'])
        verdict = ('无效回合' if not r['valid'] else ('成功' if ok else '未成功')) + f"，{r['score']}/{r['max_score']} 分"
        rnd = r.get('randomization'); on = [k for k, v in rnd.items() if v] if isinstance(rnd, dict) else []
        detail = (f"{html.escape(r['kind'])} · {r['when']} · {r['time_s']} s 仿真 / {r.get('steps') or '—'} 控制步 · 终止原因 <code>{html.escape(str(r['reason']))}</code> · "
                  f"{'随机：' + '、'.join(on) if on else '固定初态'}{' · 初态验收通过' if r.get('initial_accepted') else ''}{' · 视频由逐帧 PNG 编码' if r.get('video_source') == 'encoded_from_frames' else ''}")
        m = r['media']
        if ok:
            media = lazy_video('../' + m['head']['href'], '主视角（头部相机）', tid, m['head']['frames'], m['head']['fps'], ('../' + r['step0']) if r.get('step0') else None) if m.get('head') else "<p class='note'>该回合没有视频文件。</p>"
            cls = 'ok'
        else:
            media = ''.join(lazy_video('../' + m[c]['href'], l, tid, m[c]['frames'], m[c]['fps'], ('../' + r['step0']) if (c == 'head' and r.get('step0')) else None) for c, l in (('head', '头部相机'), ('left', '左腕相机'), ('right', '右腕相机')) if c in m)
            if not media and r.get('frame_sheet'):
                fc = r.get('frame_counts') or {}
                media = (f"<div class='episode'><h3>关键帧（三路相机，回合太短未编码视频）</h3>"
                         f"<img class='lazy-img' decoding='async' style='width:100%' data-src='../{r['frame_sheet']}' alt='{tid} 关键帧'>"
                         f"<p class='meta'>逐帧 PNG 共 头部 {fc.get('head', 0)} / 左腕 {fc.get('left', 0)} / 右腕 {fc.get('right', 0)} 帧，按时间均匀取 6 帧拼图。</p></div>")
            media = media or "<p class='note'>该回合没有视频文件，也没有可用的逐帧 PNG。</p>"
            cls = 'draft'
            if r['kind'].startswith('初态短测'): detail += "<br><b>说明：</b>初态短测只验证初态与几步保持，0 分是预期结果，不是任务失败。"
            elif r['valid']: detail += f"<br><b>推测原因：</b>评分器终止原因 <code>{html.escape(str(r['reason']))}</code>；具体过程请看三视角视频与作业目录的 worker.log / score_events.json（本页不替评分器下结论）。"
        step0 = f" · <a href='../{r['step0']}' target='_blank'>初态头部图</a>" if r.get('step0') else ''
        jobs += (f"<div class='w120-job' id='w120-{tid}-run-{i}' data-toc='{html.escape(r['when'][5:])} {html.escape(r['kind'])} · {verdict}'><h3>{html.escape(r['kind'])} · {r['when']} · <span class='status {cls}'>{verdict}</span></h3>"
                 f"<p class='note'>{detail}<br>作业目录 <code>evaluation/tasks/{html.escape(r['path'])}</code>{step0}</p><div class='episodes'>{media}</div></div>")
    w120 = (f"<div id='w120-{tid}'><h2>202 评测视频 · 合成视频检查（{len(runs)} 条记录）</h2><p class='note'>以下全部是本项目在 202 上的评测回合结果（服务端评分器的 task_evaluation.json），不是来源方仿真视频，也不是真机。"
            f"成功的只放主视角；未成功的附三视角供人工复核。{summ}</p>{jobs or '<p class=note>无。</p>'}</div>")
    # 4 当前任务定义表：DEFINITION_TABLE.md 的 "## <tid>" 节（与 shelf_pick 相同的 9 行 3 列表）；README / 其他文档只作折叠参考
    dt = definition_table(fam_dir / 'DEFINITION_TABLE.md', tid)
    refs = ((f"<details><summary>家族 README.md 原文（规则来源，{time.strftime('%Y-%m-%d %H:%M', time.localtime(readme.stat().st_mtime))}）</summary><div class='doc'>{md_to_html(readme.read_text())}</div></details>" if readme.exists() else '')
            + ''.join(f"<details><summary>{html.escape(p.name)} <span class='note'>{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}</span></summary><div class='doc'>{md_to_html(p.read_text())}</div></details>" for p in docs))
    definition = (f"<div id='definition-{tid}'><h2>当前任务定义表</h2><p class='status{'' if reg.get('ready') else ' draft'}'>{status_badge}</p>"
                  f"<p class='note'>按家族 README / 评分说明整理成与 shelf_pick 相同的表格；规则以 README 及其引用文档为准（折叠在表下），本表不新增规则。</p>"
                  + (dt or "<p class='status draft'>该任务尚无整理好的定义表（DEFINITION_TABLE.md 缺少此 ID 的小节）。</p>")
                  + f"<p class='note'>共同机器人、相机、动作协议与控制频率沿用项目配置。</p><h3>规则来源文档（折叠）</h3>{refs}</div>")
    review = (f"<div id='review-{tid}'><h2>看完后如何反馈</h2><p>在对话里按任务 ID 与表格项目提出修改；负责该家族的 agent 更新 <code>evaluation/tasks/{html.escape(sub)}/README.md</code> 后重跑 <code>build_family_pages.py</code> 与 <code>build_site.py</code>。</p>"
              f"<p class='note'>评测记录与视频由 <code>evaluation/tasks/scan_status.py</code> 扫描目录得到，不手工挑选。</p></div>")
    # intro from the definition table itself (目标 / 起点 / 动作过程), like the shelf_pick pages; no README dump
    rows = {}
    dt_path = fam_dir / 'DEFINITION_TABLE.md'
    if dt_path.exists():
        on = False
        for ln in dt_path.read_text().splitlines():
            if ln.startswith('## '): on = ln[3:].strip() == tid; continue
            if on and ln.startswith('|'):
                c = [x.strip() for x in ln.strip().strip('|').split('|')]
                if len(c) == 3 and c[0] not in ('项目',) and not c[0].startswith('---'): rows[c[0]] = c[1]
    intro = ''.join(f"<p><b>{k}：</b>{inline(rows[src])}</p>" for k, src in (('目标', '目标与源位置'), ('起点', '初态'), ('动作过程', '动作边界与结束位置')) if src in rows)
    return (f"<section class='page' data-api='{html.escape(tid)}' id='page-{html.escape(tid)}'><h1>{html.escape(title)}</h1>{ident}"
            f"<p>任务说明：{html.escape(str(reg.get('instruction') or ''))}<br><span class='note'>家族目录 <code>evaluation/tasks/{html.escape(sub)}</code> · {status_badge}</span></p>"
            f"<nav class='local'><a href='#{tid}/simsrc'>来源仿真视频</a><a href='#{tid}/media'>真机视频（折叠）</a><a href='#{tid}/w120'>202 评测视频</a>{'<a href=\'#' + tid + '/othervid\'>其他视频</a>' if others else ''}<a href='#{tid}/definition'>任务定义表</a><a href='#{tid}/review'>如何反馈</a></nav>"
            f"<div id='introduction'><h2>这个任务具体要做什么</h2>{intro or '<p class=note>见下方 README 原文。</p>'}</div>{sim_src}{real}{w120}{others}{definition}{review}</section>")


RUNNER_TO_SERVER_LOCAL = {'air_fryer_load_no_sugar': 'air_fryer_load', 'air_fryer_load_sugar': 'air_fryer_load', 'air_fryer_pull_sugar': 'air_fryer_pull', 'air_fryer_pull_no_sugar': 'air_fryer_pull'}


def build_task_pages(fam, title, sub):
    """shelf_pick-style paginated page for one family: evaluation/tasks/<sub>/preview/index.html."""
    fam_dir = TASKS / sub; out_dir = fam_dir / 'preview'; out_dir.mkdir(exist_ok=True)
    docs_dir = out_dir / 'docs'; docs_dir.mkdir(exist_ok=True)
    for md in fam_dir.glob('*.md'):
        lnk = docs_dir / md.name
        if not lnk.exists(): lnk.symlink_to(os.path.relpath(md, docs_dir))
    ids = [tid for tid, r in STATUS['registry'].items() if r['family'] == fam]
    if fam == 'hang_hat': ids = ['17-1']
    if fam == 'book_cloth': ids = ['20-2']
    if fam == 'microwave_take': ids = ['10.1']
    if fam == 'tabletop_pick': ids = [t for t in ids if t not in ('20-2',)]
    ids = sorted(ids, key=lambda x: [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r'[-._]', x)])
    readme = fam_dir / 'README.md'
    docs = sorted([p for p in fam_dir.glob('*.md') if p.name != 'README.md'], key=lambda p: -p.stat().st_mtime)
    used = {os.path.realpath(TASKS / v['source']) for r in STATUS['runs'] for v in r['media'].values()}
    scan_dirs = [fam_dir] + [TASKS / x for x in SUBPROJECT_EXTRA.get(fam, [])]
    others_html = other_videos(scan_dirs, out_dir, used)
    pages, titles = [], {}
    for tid in ids:
        runs = [r for r in STATUS['runs'] if r['task_id'] == tid]
        pages.append(task_page(fam, sub, tid, runs, fam_dir, out_dir, readme, docs, single=len(ids) == 1, others=others_html.replace("id='othervid'", f"id='othervid-{tid}'")))
        titles[tid] = f"{tid} {STATUS['registry'].get(tid, {}).get('instruction') or ''}"[:28]
    task_list = ''.join(f"<a href='#{html.escape(t)}' data-api='{html.escape(t)}'>{html.escape(titles[t])}</a>" for t in ids)
    note = (f"<p class=\"note\" style=\"margin:2px 0 0\">{html.escape(title)}：每页一个任务 ID；视频点击播放后才加载。页面形式与 shelf_pick 4-x 相同，内容来自该家族目录（README、评测记录、来源视频）。"
            f"<span class=\"family-nav\"><a href=\"../\">← 任务审阅站总览</a> · <a href=\"../#results\">评测总表</a></span></p>")
    page = template_shell().replace('__TITLE__', html.escape(title)).replace('__SHORT__', html.escape(title)).replace('__BATCH_NOTE__', note).replace('__TASK_LIST__', task_list).replace('__PAGES__', '\n'.join(pages))
    (out_dir / 'index.html').write_text(page)
    print(fam, '->', out_dir / 'index.html', len(ids), 'ids')


def main():
    for fam, (title, sub) in GENERIC.items(): build_task_pages(fam, title, sub)
    for fam, (title, sub) in DOCS_ONLY.items(): build_docs_page(fam, title, sub, 'family.html')


if __name__ == '__main__':
    main()
