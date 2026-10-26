"""Build the paginated static review page for Task 1 (coffee machine): preview/index.html. CPU only, no simulator or browser.

Pages are keyed by API id 1-1 … 1-4 (user decision 2026-09-29: four independent APIs), which map one-to-one to
任务编号 / data_list.json subtasks 1.1 … 1.4 (放杯子 / 倾倒咖啡豆 / 按按钮 / 取杯子). Old 1.x anchors and directories stay as aliases.

Single sources (the page never edits rules):
  ../apis/<key>/README.md                                   definitions, randomization two-column table, implementation binding
  ../apis/api_index.json, ../apis/observations.json         titles / facts shown in the identity table, video observations
  ../references/source_sim_videos/<key>/sim-provenance.json  5 source-rendered simulation videos (data producer's renders)
  ../references/l2_real_batch/<key>/media-provenance.json    real-robot videos (n/3, folded)
"""
from pathlib import Path
import html
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from task_catalog import identity_html   # 【Task · 子任务ID · API ID · 任务编号 · 任务描述】 from data_requirement.csv + VLA_task.csv

KEYS = ['1-1', '1-2', '1-3', '1-4']   # API ids, in 任务编号 order
EVALUATOR_ID = re.search(r"EVALUATOR_ID = '([^']+)'", (ROOT / 'specs.py').read_text()).group(1)
SCENE = 'kitchen'   # 合成视频检查节的场景标签（preview/kitchen-results.json，测试后由发布工具写入）


def inline(value, media_prefix=''):
    value = html.escape(value, quote=False)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = value.replace('&lt;br&gt;', '<br>')
    def link(m):
        text, href = m.group(1), m.group(2)
        if re.match(r'^(https?:|#)', href): return f"<a href='{href}'>{text}</a>"
        if href.startswith('preview/'): return f"<a href='{media_prefix}{href[len('preview/'):]}' target='_blank' rel='noopener'>{text}</a>"
        return f"{text}（<code>{href}</code>）"   # repo paths are not served by the preview server
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, value)


def readme_sections(path):
    sections, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith('## '):
            current = line[3:].strip(); sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def table_html(rows, cls):
    assert rows and all(len(r) == len(rows[0]) for r in rows), rows[0]
    if len(rows) >= 2 and all(re.fullmatch(r":?-+:?", c) for c in rows[1]):
        head = "<thead><tr>" + "".join(f"<th scope='col'>{inline(c)}</th>" for c in rows[0]) + "</tr></thead>"
        rows = rows[2:]
    else:   # continuation rows written without a header line
        head = ''
    body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'>{head}<tbody>{body}</tbody></table>"


def blocks(lines, table_cls='definition'):
    """Minimal Markdown: pipe tables, bullet lists, ### headings, paragraphs."""
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
            out.append(table_html(rows, table_cls)); continue
        if re.match(r'^\s*[-*]\s+', ln):
            items = []
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                items.append(f"<li>{inline(re.sub(r'^\s*[-*]\s+', '', lines[i]))}</li>"); i += 1
            out.append('<ul>' + ''.join(items) + '</ul>'); continue
        if ln.startswith('### '):
            out.append(f"<h3>{inline(ln[4:])}</h3>"); i += 1; continue
        if ln.strip(): out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return ''.join(out)


def identity_line(key, index):
    return identity_html(key)


def lazy(base, cam, label, poster=None):
    pst = f" data-poster='{poster}'" if poster else ''
    return (f"<video class='lazy-video' controls playsinline preload='none'{pst} aria-label='{html.escape(label)}'><source data-src='{base}-{cam}.mp4' type='video/mp4'></video>"
            f"<button class='load-video' type='button'>播放{html.escape(label)}</button><p class='note video-status' aria-live='polite'>点击播放后才加载。也可直接打开 <a href='{base}-{cam}.mp4'>MP4</a>。</p>")


def sim_section(key):
    path = ROOT / 'references/source_sim_videos' / key / 'sim-provenance.json'
    prov = json.loads(path.read_text()) if path.is_file() else dict(episodes=[])
    eps = prov['episodes'][:5]
    cards = []
    for i, ep in enumerate(eps):
        base = f"{key}/sim-{ep['label']}"
        for suffix in ('-head.mp4', '-poster.jpg', '-keyframes.jpg'):
            assert (ROOT / 'preview' / (base + suffix)).is_file(), base + suffix
        hands = ''.join(f"<div class='hand-view'><h4>{lab}</h4>{lazy(base, cam, lab)}</div>" for cam, lab in (('left', '左腕相机'), ('right', '右腕相机')) if (ROOT / 'preview' / f'{base}-{cam}.mp4').is_file())
        hands = f"<details class='hand-views-wrap'><summary>左右腕相机（可选）</summary><div class='hand-views'>{hands}</div></details>" if hands else ''
        feats = '，'.join(f"{k} {v}" for k, v in ep['features'].items())
        cats = '，'.join(f"{k}={v}" for k, v in ep['categories'].items())
        dl = ep['data_list']
        head_sha = next(f['source_sha256'] for f in ep['files'] if f['cam'] == 'head')
        cards.append(f"<div class='episode'><h3>仿真视频 {i + 1}/{len(eps)} · {html.escape(ep['label'])}</h3>"
                     f"{lazy(base, 'head', '头部相机仿真视频', base + '-poster.jpg')}"
                     f"<img class='lazy-img' decoding='async' width='2400' height='300' style='width:100%' data-src='{base}-keyframes.jpg' alt='首帧、25%、50%、75%、末帧'>"
                     f"<p class='meta'>{ep['head_frames']} 帧 · {ep['head_fps']} fps · {ep['head_duration_s']} s · 原分辨率 {ep['head_resolution'][0]}×{ep['head_resolution'][1]} · {html.escape(cats)}</p>"
                     f"<p class='meta'>本条采样值：{html.escape(feats)}</p>"
                     f"<p class='meta'>数据清单标记（原样）：<b>{html.escape(dl['source_status'])}</b> · 清单条目 <code>{html.escape(dl['title'])}</code>（id {html.escape(dl['id'])}）</p>"
                     f"<p class='meta'>202 目录 <code>{html.escape(ep['episode_dir'])}</code><br>来源登记（不是 202 下载地址）<code>{html.escape(dl['source_path'])}</code> · 头部视频 SHA-256 <code>{head_sha[:16]}…</code></p>{hands}</div>")
    n = len(eps)
    gap = '' if n >= 5 else f"<p class='status'>来源仿真视频 {n}/5：缺口如实保留，不用 202 回放冒充。</p>"
    return (f"<div id='sim-{key}'><h2>来源仿真视频 · 主参考（{n}/5，数据方渲染的现有轨迹）</h2>{gap}"
            f"<p class='note'>这是数据方（lukun 咖啡机工程）对其现有轨迹的渲染视频，从该子任务 50 条里按覆盖度选出 5 条不同 episode（{html.escape(prov.get('selection_method', ''))}）。"
            f"它们不是 202 的物理回放，也不代表 202 验收；数据清单只核对了目录与文件存在，没有重新做物理或人工验收。来源路径与三路视频哈希见仓库文件 "
            f"<code>evaluation/tasks/coffee_machine/references/source_sim_videos/{key}/sim-provenance.json</code>。</p><div class='episodes'>{''.join(cards)}</div></div>")


def real_section(key, observation):
    path = ROOT / 'references/l2_real_batch' / key / 'media-provenance.json'
    prov = json.loads(path.read_text())
    eps = [e for e in prov['episodes'] if e.get('kind') == 'real'][:3]
    assert len({e['episode_uuid'] for e in eps}) == len(eps)
    cards = []
    for i, ep in enumerate(eps):
        short = ep['episode_uuid'][:8]; base = f'{key}/real-head-{short}'
        for suffix in ('.mp4', '-poster.jpg', '-keyframes.jpg'):
            assert (ROOT / 'preview' / (base + suffix)).is_file(), base + suffix
        cards.append(f"<div class='episode'><h3>真机视频 {i + 1}/{len(eps)} · episode {short} · L2 task {ep['small_task_id']}</h3>"
                     f"<video class='lazy-video' controls playsinline preload='none' data-poster='{base}-poster.jpg' aria-label='{key} 真机头部相机视频 {short}'><source data-src='{base}.mp4' type='video/mp4'></video>"
                     f"<button class='load-video' type='button'>播放真机视频</button><p class='note video-status' aria-live='polite'>点击播放后才加载。也可直接打开 <a href='{base}.mp4'>MP4</a>。</p>"
                     f"<img class='lazy-img' decoding='async' width='2400' height='270' style='width:100%' data-src='{base}-keyframes.jpg' alt='首帧、25%、50%、75%、末帧'>"
                     f"<p class='meta'>所见：{html.escape(ep.get('note') or '')}</p>"
                     f"<p class='meta'>{ep['decoded_frames']} 帧 · 原始时间戳跨度 {ep['source_timestamp_span_s']:.2f} s · 录制 {html.escape(ep['create_time'])} · 原分辨率 {ep['original_resolution'][0]}×{ep['original_resolution'][1]} · meta 校验 {'通过' if ep['validate'] else '未通过'}</p>"
                     f"<p class='meta'>来源：<code>arena-l2:{html.escape(ep['remote_video'])}</code></p></div>")
    rejected = [r for g in prov.get('groups', []) for r in g.get('rejected', [])]
    rej = ('<p class=\'note\'>未选候选：' + '；'.join(f"<code>{html.escape(r['uuid'][:8])}</code> {html.escape(r['reason'])}" for r in rejected) + '</p>') if rejected else ''
    gap = '' if len(eps) == 3 else (f"<p class='status'>真机视频 {len(eps)}/3：L2 没有这个子任务的真机录制。已查范围：{html.escape(prov['searched'])}。"
                                   f"缺口保留到用户定稿时决定是否接受现有资料；不用仿真冒充真机。</p>")
    return (f"<details id='media-{key}' class='real-wrap'><summary><h2 style='display:inline'>真机视频 · 头部相机（{len(eps)}/3，真实机器人录制，默认折叠）</h2></summary>{gap}"
            f"<p><b>观察：</b>{html.escape(observation)}</p>"
            f"<p class='note'>来源 L2 <code>/data/admin/work/livingroom_data/record/&lt;uuid&gt;/camera/head_color/head_color.h265</code>，按 record_index.tsv / meta_info.json 的 task_id 选取不同 episode 并逐条看首/中/末帧核对后转码为 960 px H.264 MP4；"
            f"原时间戳与哈希见仓库文件 <code>evaluation/tasks/coffee_machine/references/l2_real_batch/{key}/media-provenance.json</code>。真机画面只作动作参考，不代表 202 初态或成功验收。</p>{rej}"
            f"<div class='episodes'>{''.join(cards)}</div></details>")


def random_section(key, sections):
    lines = sections.get('随机化（实际覆盖 / 拟采用）')
    assert lines, (key, '随机化节缺失')
    frag = ROOT / 'preview' / key / 'coverage.html'   # inline SVG + table view written by tools/build_coverage_figures.py
    fig = frag.read_text() if frag.is_file() else ''
    return (f"<div id='random-{key}'><h2>随机化：(a) 这批数据实际覆盖 / (b) 评测拟采用</h2>"
            f"<p class='note'>(a) 逐条读取 50 条 episode 的记录文件得到的实测值；(b) 必须来自可复现的采样代码 + 种子，以数据方生成器为准补齐。指垫、摩擦、阻尼、驱动等物理参数固定，不纳入随机化。</p>"
            f"{blocks(lines, 'definition rand')}{fig}</div>")


def eval_section(key):
    """合成视频检查节：满分成功只放头部视角；未成功或有扣分标原因并放三视角（preview/kitchen-results.json）。"""
    intro = ("<p class='note'><b>以下全部是 202 仿真结果，不是真机。</b>评测场景用公共共享咖啡资产（<code>resources/shared_assets/coffee_v1</code>）和公共统一参数，"
             "只从来源 episode 取布局（物体位姿、杯款、1-4 的杯内液体）；动作只回放来源轨迹，不生成、不修改。标准作业三项（用户 2026-09-29 定）："
             "固定初态短测、来源轨迹回放（每条在自己的来源布局上）、种子 7 随机初态短测；不做随机初态回放。"
             "GPU 只占 202 的空闲卡，GPU 0、1 不占用。左侧目录里“合成视频检查”下列出每条作业，点一条就跳到它并自动加载它的头部相机视频。</p>")
    path = ROOT / f'preview/{SCENE}-results.json'
    data = json.loads(path.read_text()) if path.is_file() else {}
    batch_of = (data.get('batches') or {}).get(key, data.get('batch'))
    rows = data.get('apis', {}).get(key, {})
    blocked = data.get('blocked', {}).get(key)
    not_run = {k.split(':', 1)[1]: v for k, v in data.get('not_run', {}).items() if k.startswith(key + ':')}
    if not rows and not blocked and not not_run:
        return (f"<div id='{SCENE}-{key}'><h2>合成视频检查 · 202 评测结果</h2><p class='status draft'>尚无已发布的作业结果</p>{intro}</div>")

    def video(v, label):
        return (f"<div class='episode'><h3>{label}</h3><video class='lazy-video' controls playsinline preload='none' aria-label='{key} {label}'><source data-src='{v['href']}' type='video/mp4'></video>"
                f"<button class='load-video' type='button'>播放仿真视频</button><p class='note video-status' aria-live='polite'>点击播放后才加载。也可直接打开 <a href='{v['href']}'>MP4</a>。</p><p class='meta'>{v['frames']} 帧 · {v['fps']} fps</p></div>")

    def measured(m):
        if not m: return ''
        names = dict(cup_tilt_deg='杯子倾斜 °', bag_tilt_deg='豆袋倾斜 °', cup_supports='杯子承托', bag_supports='豆袋承托', cup_xy_error_m='杯子离放杯点 m',
                     button_displacement_m='按钮位移 m', beans_in_bag='袋内豆数', beans_in_hopper='豆仓豆数', liquid_particles='液体粒子数', liquid_in_cup='杯内粒子数', cup_in_head_frame='杯子完整在头部相机画面内')
        show = lambda v: ('是' if v else '否') if isinstance(v, bool) else ('%.4g' % v) if isinstance(v, float) else ('、'.join(map(str, v)) or '无') if isinstance(v, list) else str(v)
        return '；'.join(f"{names[k]} {html.escape(show(v))}" for k, v in m.items() if k in names)

    body, notes = '', data.get('notes', {}).get(key, {})
    order = data.get('pipeline', {}).get(key, list(rows))
    summary = []
    for name in order:
        if name in not_run and name not in rows:
            shown = '种子 7 随机初态短测' if name == 'random-probe' else name.replace('source-run-', '来源轨迹回放 ').replace('layout-probe-', '来源布局初态验收 ')
            why = not_run[name]
            if 'no reproducible sampler' in why:
                why = '定义表没有给出这个 API 可复现的随机采样器（原始生成器没有取得），不自造采样器，所以不运行。'
            if 'outside the public interface' in why:
                m = re.search(r'(\d+) of (\d+) rows .*?largest ([0-9.]+) rad', why)
                why = ('未运行：来源轨迹有 %s/%s 行的夹爪指令超出公共接口范围 [0, π/4]（最大 %s rad）；按“动作数值禁止任何改动”不截断、不钳制，公共接口拒绝，等轨迹负责方处理。' % m.groups()
                       if m else '未运行：来源轨迹指令超出公共接口范围，按“动作数值禁止任何改动”不放行。' + why)
            body += f"<div class='house-job' id='{SCENE}-{key}-{name}-0' data-toc='{html.escape(shown)} · 未运行' data-ok='bad'><h3>{html.escape(shown)} · <span class='status draft'>未运行</span></h3><p class='note'>{html.escape(why)}</p></div>"
            summary.append(f"<tr><td>{html.escape(shown)}</td><td>—</td><td>未运行</td><td>{html.escape(why)}</td></tr>")
            continue
        row = rows.get(name)
        if not row: continue
        for i, att in enumerate(row['attempts']):
            label = row['label'] + ('' if len(row['attempts']) == 1 else f'（第 {i + 1} 次）')
            episode = f" · 来源 episode <code>{html.escape(row['episode'])}</code>" if row.get('episode') else ''
            out = f"<br>作业目录 <code>{html.escape(att.get('output') or '')}</code>"
            vids = att.get('videos') or {}
            if att.get('kind') == 'replay' and att.get('audit_passed') is not None:
                full = bool(att['success']) and att['score'] == att['maximum_score']
                verdict = ('成功' if att['success'] else '未成功') + f"，{att['score']:g} / {att['maximum_score']} 分"
                stages = '，'.join(f"阶段{s} {'+%d（%.2f s）' % (v['points'], v['time_s']) if v['awarded'] else '未得分'}" for s, v in att['stage_points'].items())
                counts = att.get('final_counts') or {}
                extra = []
                if 'beans_in_hopper' in counts: extra.append(f"最终进仓 {counts['beans_in_hopper']}/{counts['beans_total']} 颗")
                if 'in_cup_ml' in counts: extra.append(f"杯内 {counts['in_cup_ml']:.1f} ml，洒出 {counts['spilled_particles']} 粒，空中 {counts['airborne_particles']} 粒（基数 {counts['denominator_particles']} 粒）")
                if 'valid_button_events' in counts: extra.append(f"有效按钮事件 {counts['valid_button_events']} 次")
                detail = (f"{att['control_steps']} 控制步（{att['end_time_s'] if att.get('end_time_s') is not None else att['control_steps'] * 0.05:.2f} s）；{stages}；"
                          f"{'扣分：' + '、'.join(att['deductions']) if att['deductions'] else '零扣分'}；{'；'.join(extra) + '；' if extra else ''}"
                          f"结束原因：{html.escape(str(att['reason']))}；CPU 复算{'一致' if att['audit_passed'] else '不一致'}")
                reason = notes.get(name, '')
                if full:
                    media, cls = (video(vids['head'], '头部相机') if vids.get('head') else ''), 'ok'
                else:
                    media = ''.join(video(vids[c], l) for c, l in (('head', '头部相机'), ('left', '左腕相机'), ('right', '右腕相机')) if vids.get(c))
                    cls = 'draft'
                    detail += f"<br><b>推测原因：</b>{html.escape(reason) if reason else '待人工看三视角视频后补'}"
                keys_img = f"<img class='lazy-img' decoding='async' style='width:100%' data-src='{att['keyframes']}' alt='关键帧'>" if att.get('keyframes') else ''
                toc = f"{label} · {row.get('episode') or ''} · {verdict}"
                body += (f"<div class='house-job' id='{SCENE}-{key}-{name}-{i}' data-toc='{html.escape(toc)}' data-ok='{'ok' if full else 'bad'}'><h3>{html.escape(label)} · <span class='status {cls}'>{verdict}</span></h3>"
                         f"<p class='note'>{detail}{episode}{out}</p><div class='episodes'>{media}</div>{keys_img}</div>")
                summary.append(f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(row.get('episode') or '')}</code></td><td>{verdict}</td><td>{html.escape(str(att['reason']))}{'；' + '、'.join(att['deductions']) if att['deductions'] else ''}</td></tr>")
            elif att.get('kind') == 'probe' and att.get('audit_passed') is not None:
                verdict = '初态验收通过' if att['audit_passed'] else '初态验收未通过'
                img = f" <a href='{att['initial_head']}' target='_blank'>初态头部图</a>" if att.get('initial_head') else ''
                failed = ('<br>未通过项：' + html.escape('、'.join(att['rejected_checks']))) if att.get('rejected_checks') else ''
                layout = att.get('layout_randomization')
                drawn = ''
                if layout:
                    pr = layout.get('proposal') or {}
                    parts = []
                    if pr: parts.append(f"咖啡机平移 ({pr['machine_dx_m']:.4f}, {pr['machine_dy_m']:.4f}) m，转 {pr['machine_dyaw_deg']:.2f}°")
                    if 'machine_offset_m' in layout: parts.append(f"咖啡机平移 ({layout['machine_offset_m'][0]:.4f}, {layout['machine_offset_m'][1]:.4f}) m，转 {layout['machine_yaw_deg']:.2f}°")
                    if 'cup_offset_m' in layout: parts.append(f"杯子平移 ({layout['cup_offset_m'][0]:.4f}, {layout['cup_offset_m'][1]:.4f}) m，杯柄转 {layout['cup_yaw_delta_deg']:.2f}°")
                    if 'bag_offset_m' in layout: parts.append(f"豆袋平移 ({layout['bag_offset_m'][0]:.4f}, {layout['bag_offset_m'][1]:.4f}) m，转 {layout['bag_yaw_deg']:.2f}°")
                    if layout.get('cup_style'): parts.append(f"杯款 {layout['cup_style']}")
                    if layout.get('episode_seed') not in (None, layout.get('seed')): parts.append(f"采样器前移到种子 {layout['episode_seed']}")
                    drawn = f"<br>种子 {layout['seed']}（{html.escape(str(layout.get('sampler', ''))[:90])}）：" + '；'.join(parts)
                toc = f"{label} · {row.get('episode') or ''} · {verdict}"
                body += (f"<div class='house-job' id='{SCENE}-{key}-{name}-{i}' data-toc='{html.escape(toc)}' data-ok='{'ok' if att['audit_passed'] else 'bad'}'><h3>{html.escape(label)} · <span class='status{'' if att['audit_passed'] else ' draft'}'>{verdict}</span></h3>"
                         f"<p class='note'>静置 {att['settle_steps']} 个物理步后验收；物理配置 <code>{html.escape(str(att.get('physics')))}</code>；{measured(att.get('measured'))}{img}{failed}{drawn}{episode}{out}</p></div>")
                summary.append(f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(row.get('episode') or '')}</code></td><td>{verdict}</td><td>{html.escape('、'.join(att.get('rejected_checks') or []))}</td></tr>")
            else:
                st = att.get('status')
                text = {'running': '进行中', 'preflight_failed': '启动前检查未过（未启动物理）', 'error': '作业出错（不是任务结果）'}.get(st, str(st))
                why = att.get('error') or att.get('follow_up_error') or att.get('note') or ''
                body += (f"<div class='house-job' id='{SCENE}-{key}-{name}-{i}' data-toc='{html.escape(label + ' · ' + (row.get('episode') or '') + ' · ' + text)}' data-ok=''><h3>{html.escape(label)} · <span class='status draft'>{html.escape(text)}</span></h3>"
                         f"<p class='note'>{html.escape(str(why))[:400]}{episode}{out}</p></div>")
                summary.append(f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(row.get('episode') or '')}</code></td><td>{html.escape(text)}</td><td>{html.escape(str(why))[:120]}</td></tr>")
    block_note = f"<p class='status draft'>受阻：{html.escape(str(blocked))}</p>" if blocked else ''
    snap = data.get('snapshot') or {}
    snap_note = (f"<p class='note'>冻结快照 <code>{html.escape(str(snap.get('snapshot')))}</code>，{snap.get('file_count')} 个文件逐个记哈希，清单 <code>{html.escape(str(snap.get('manifest')))}</code>。</p>" if snap else '')
    table = f"<table class='definition'><thead><tr><th scope='col'>作业</th><th scope='col'>来源 episode</th><th scope='col'>结果</th><th scope='col'>结束原因 / 扣分 / 未通过项</th></tr></thead><tbody>{''.join(summary)}</tbody></table>"
    parallel = data.get('max_parallel')
    return (f"<div id='{SCENE}-{key}'><h2>合成视频检查 · 202 评测结果</h2><p class='note'>批次目录 <code>{html.escape(str(batch_of))}</code>{'，同时最多 %s 张 GPU' % parallel if parallel else ''}</p>{block_note}{intro}{snap_note}"
            f"{table}<p class='note'>满分成功的回放只给头部视角；未成功或有扣分的给三视角供人工二次检查。</p>{body}</div>")


def build_page(key, index, observations):
    readme = ROOT / 'apis' / key / 'README.md'
    sections = readme_sections(readme)
    lines = readme.read_text().splitlines(); title = lines[0].lstrip('# ').strip()
    status = next((l[len('状态：'):] for l in lines if l.startswith('状态：')), '')
    approved = '已定稿' in status or '已审完' in status
    # badge = the README's own leading status phrase (the README is the single source), never a hard-coded draft number
    badge = re.split(r'[，。（(]', status, 1)[0].strip() or ('定义已定稿' if approved else '草案 · 待用户审阅')
    ix = index[key]
    sim_path = ROOT / 'references/source_sim_videos' / key / 'sim-provenance.json'
    sim_n = min(5, len(json.loads(sim_path.read_text())['episodes'])) if sim_path.is_file() else 0
    real_n = len([e for e in json.loads((ROOT / 'references/l2_real_batch' / key / 'media-provenance.json').read_text())['episodes'] if e.get('kind') == 'real'][:3])
    identity = ("<table><tbody>"
                f"<tr><th>任务编号（CSV / 数据清单）</th><td><code>{ix['task_no']}</code> {html.escape(ix['title'])}</td><th>API ID（运行接口）</th><td><code>{key}</code>（用户 2026-09-29 裁定；服务端已登记，评测器 <code>{EVALUATOR_ID}</code>）</td></tr>"
                f"<tr><th>数据批次 / 条数</th><td><code>{html.escape(ix['collection'])}</code> / {ix['count']}</td><th>数据清单标记（原样）</th><td>{inline(ix['source_status'])}</td></tr>"
                f"<tr><th>操作对象</th><td>{inline(ix['target'])}</td><th>用手（来源轨迹）</th><td>{inline(ix['hand'])}</td></tr>"
                f"<tr><th>来源仿真视频 / 真机视频</th><td>{sim_n}/5 仿真（主参考）；真机 {real_n}/3（{inline(ix['real'])}）</td><th>来源轨迹时长</th><td>{inline(ix['duration'])}</td></tr>"
                "</tbody></table>")
    binding = sections.get('实现绑定（供实现与验收，不影响上表语义）')
    binding_html = (f"<details id='binding-{key}'><summary><h2 style='display:inline'>实现绑定（技术细节，供实现与验收）</h2></summary>{blocks(binding)}</details>" if binding else '')
    def_lines = sections.get('第一步：任务定义表') or sections.get('已确认任务定义表')
    assert def_lines, (key, '定义表节缺失')
    return (f"<section class='page' data-api='{key}' id='page-{key}'>"
            f"<h1>{html.escape(ix['task_no'])} / {html.escape(key)} · {html.escape(ix['title'])}</h1>{identity_line(key, ix)}"
            f"<p><span class='status{'' if approved else ' draft'}'>{badge}</span> <span class='note'>{inline(status)}</span></p>"
            f"<nav class='local'><a href='#{key}/sim'>来源仿真视频</a><a href='#{key}/media'>真机视频</a><a href='#{key}/random'>随机化</a><a href='#{key}/{SCENE}'>合成视频检查</a><a href='#{key}/definition'>任务定义表</a><a href='#{key}/review'>如何反馈</a></nav>"
            f"<div><h2>这个任务具体要做什么</h2>{blocks(sections['任务具体介绍'])}{identity}</div>"
            f"{sim_section(key)}{real_section(key, observations.get(key, '待观察'))}{random_section(key, sections)}{eval_section(key)}"
            f"<div id='definition-{key}'><h2>当前任务定义表</h2><p class='status{'' if approved else ' draft'}'>{badge}</p>"
            f"<p class='note'>{'已获用户确认；右列为实现备注与实测核定项。' if approved else '中列为 agent 根据来源仿真视频、数据方通过判据与记录文件提出的建议值，右列为本任务差异与待核定项；均未当作已批准规则。四个子任务各按自己的定义评测，不拼接为长任务。'}</p>"
            f"{blocks(def_lines, 'definition')}<p class='note'>共同机器人、相机、动作协议与控制频率沿用项目配置；不因本任务调整任何物理参数。</p></div>{binding_html}"
            f"<div id='review-{key}'><h2>看完后如何反馈</h2><p>在当前对话里按 API ID（或任务编号）与表格项目提出修改，例如“{key}：成功条件改为 …”。agent 更新 <code>apis/{key}/README.md</code> 后重新生成本页；确认定稿（“{key} 通过”）后才进入实现与测试。"
            f"编号裁定记录见 <a href='family.html'>家族说明</a>。</p>"
            f"<p class='note'>定义表源文件：<code>evaluation/tasks/coffee_machine/apis/{key}/README.md</code>（仓库内路径，预览服务不发布）。</p></div>"
            "</section>")


def main():
    template = (ROOT / 'preview/template.html').read_text()
    index = json.loads((ROOT / 'apis/api_index.json').read_text())
    observations = json.loads((ROOT / 'apis/observations.json').read_text())
    titles = {k: (ROOT / 'apis' / k / 'README.md').read_text().splitlines()[0].lstrip('# ').strip() for k in KEYS}
    task_list = ''.join(f"<a href='#{k}' data-api='{k}'>{html.escape(index[k]['task_no'] + ' / ' + k + ' ' + index[k]['title'])}</a>" for k in KEYS)
    pages = '\n'.join(build_page(k, index, observations) for k in KEYS)
    for marker, value in {'__TASK_LIST__': task_list, '__PAGES__': pages}.items():
        assert template.count(marker) == 1, marker
        template = template.replace(marker, value)
    (ROOT / 'preview/index.html').write_text(template)
    (ROOT / 'preview/hub-fragment.html').unlink(missing_ok=True)   # superseded 2026-09-29: the hub lists the four API rows itself
    report = dict(family='coffee_machine (Task 1 refill_beans)', preview='paginated_task_review', pages=KEYS, page_key_meaning='API id (1-1 … 1-4) = 任务编号 1.1 … 1.4', scene_tag=SCENE,
                  media_loading='external_files_on_play; posters/keyframes only for current page', titles=titles,
                  definition_sources={k: f'../apis/{k}/README.md' for k in KEYS},
                  sim_videos={k: len(json.loads((ROOT / 'references/source_sim_videos' / k / 'sim-provenance.json').read_text())['episodes']) for k in KEYS},
                  real_videos_shown={k: len(json.loads((ROOT / 'references/l2_real_batch' / k / 'media-provenance.json').read_text())['episodes'][:3]) for k in KEYS},
                  definition_status={k: ('approved' if re.search('已定稿|已审完', next((l for l in (ROOT / 'apis' / k / 'README.md').read_text().splitlines() if l.startswith('状态：')), '')) else 'draft') for k in KEYS},
                  index_bytes=(ROOT / 'preview/index.html').stat().st_size)
    (ROOT / 'preview/build-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('Built Task 1 review:', len(KEYS), 'pages,', report['index_bytes'], 'bytes')


if __name__ == '__main__':
    main()
