"""Coverage figures for the review pages: what the producer's 50 episodes per subtask actually sampled, drawn against the
sampling range written in the producer's generator code. CPU only; writes inline-SVG fragments, no plotting library.

Input : references/data_side/coverage_rows.json      per-episode values (written by this tool from the batch manifests)
        references/data_side/generator_ranges.json   ranges quoted from the generator code (file:line in the same JSON)
Output: preview/<key>/coverage.html                  <figure> with small-multiple panels + a table view of the same values

Form: one panel per sampled quantity. XY offsets are scatter panels with equal axes; angles are strip panels.
Colour = identity of at most two groups (arm, or small/large perturbation batch), palette slots 1–2 of the documented
default, validated with `validate_palette.js --pairs all --mode light` (the review site is light-only).
Every dot has a native hover title and the same values are in the table below the figure."""
import html, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_sim_episodes as sel

NOMINAL = dict(machine=(2.569096825, 0.166750471), cup=(2.360096825, -0.100249529))
SERIES = ['#2a78d6', '#eb6834']
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
PANEL, GAP, TOP, LEFT, BOTTOM = 190, 34, 62, 46, 40

# page key -> (series field, series label map, panels). XY panel: ('xy', title, fx, fy, range key); strip: ('strip', title, f, range key)
SPEC = {
    '1-1': ('arm', {'left': '左臂', 'right': '右臂'}, [('xy', '杯子初始位置偏移 (m)', 'cup_dx', 'cup_dy', 'cup_xy'), ('xy', '咖啡机位置偏移 (m)', 'machine_dx', 'machine_dy', 'machine_xy'),
                                                 ('strip', '杯柄朝向 (°，世界系)', 'handle_yaw_deg', 'handle_yaw')]),
    '1-2': ('group', {'小扰动': '小扰动批', '大扰动': '大扰动批'}, [('xy', '咖啡机位置偏移 (m)', 'machine_dx', 'machine_dy', 'machine_xy'), ('xy', '豆袋位置偏移 (m)', 'bag_dx', 'bag_dy', 'bag_xy'),
                                                        ('strip', '咖啡机转角 (°)', 'machine_yaw_deg', 'machine_yaw'), ('strip', '豆袋转角 (°)', 'bag_yaw_deg', 'bag_yaw')]),
    '1-3': (None, {}, [('xy', '咖啡机位置增量 (m，叠加在基准摆位上)', 'machine_dx', 'machine_dy', 'machine_xy'), ('strip', '咖啡机转角增量 (°)', 'machine_dyaw_deg', 'machine_yaw')]),
    '1-4': ('arm', {'left': '左臂', 'right': '右臂'}, [('xy', '咖啡机位置偏移 (m)', 'machine_dx', 'machine_dy', 'machine_xy'), ('strip', '咖啡机转角 (°)', 'machine_yaw_deg', 'machine_yaw'),
                                                 ('strip', '杯柄朝向 (°，世界系)', 'handle_yaw_deg', 'handle_yaw')]),
}


def rows():
    out = {}
    for sub, eps in sel.features().items():
        key = sel.API[sub]
        out[key] = []
        for name, f in sorted(eps.items()):
            v = dict(zip(f['feat_names'], f['feats']))
            if sub == '1.1':   # manifest stores absolute positions; offsets are relative to the producer's nominal layout
                v = dict(handle_yaw_deg=v['handle_yaw_deg'], cup_dx=v['cup_x'] - NOMINAL['cup'][0], cup_dy=v['cup_y'] - NOMINAL['cup'][1],
                         machine_dx=v['machine_x'] - NOMINAL['machine'][0], machine_dy=v['machine_y'] - NOMINAL['machine'][1])
            out[key].append(dict(episode=name, **{k: round(x, 5) for k, x in v.items()}, **{k: c for k, c in f['cats'].items() if k in ('arm', 'group', 'cup_style', 'start')}))
    return out


def ticks(lo, hi, n=4):
    span = hi - lo
    step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90) if span / s <= n)
    t = int(lo / step - 1) * step
    out = []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9: out.append(round(t, 6))
        t += step
    return out


def fmt(v):
    return f'{v:g}'


def dot(x, y, colour, title):
    return (f"<g class='pt' tabindex='0'><title>{html.escape(title)}</title><circle cx='{x:.1f}' cy='{y:.1f}' r='12' fill='transparent'/>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{colour}' stroke='{SURFACE}' stroke-width='2'/></g>")


def panel(kind, title, data, series_of, order, ranges, x0):
    g = [f"<text x='{x0}' y='{TOP - 44}' font-size='12.5' font-weight='600' fill='{INK}'>{html.escape(title)}</text>"]
    y0 = TOP
    boxes = ranges.get('boxes', [])
    by_series = ranges.get('boxes_by_series', {})
    if kind == 'xy':
        vals = [abs(v) for d in data for v in (d['x'], d['y'])] + [abs(v) for b in boxes for v in b['range']]
        lim = max(0.06, max(vals) * 1.12)
        sx = lambda v: x0 + (v + lim) / (2 * lim) * PANEL
        sy = lambda v: y0 + PANEL - (v + lim) / (2 * lim) * PANEL
        for t in ticks(-lim, lim):
            g.append(f"<line x1='{sx(t):.1f}' y1='{y0}' x2='{sx(t):.1f}' y2='{y0 + PANEL}' stroke='{GRID}'/><line x1='{x0}' y1='{sy(t):.1f}' x2='{x0 + PANEL}' y2='{sy(t):.1f}' stroke='{GRID}'/>")
            g.append(f"<text x='{sx(t):.1f}' y='{y0 + PANEL + 14}' font-size='10.5' fill='{MUTED}' text-anchor='middle'>{fmt(t)}</text><text x='{x0 - 6}' y='{sy(t) + 3.5:.1f}' font-size='10.5' fill='{MUTED}' text-anchor='end'>{fmt(t)}</text>")
        g.append(f"<line x1='{sx(0):.1f}' y1='{y0}' x2='{sx(0):.1f}' y2='{y0 + PANEL}' stroke='{AXIS}'/><line x1='{x0}' y1='{sy(0):.1f}' x2='{x0 + PANEL}' y2='{sy(0):.1f}' stroke='{AXIS}'/>")
        for b in boxes:   # nested boxes: each labelled at its own top-left corner, text in ink (never a series colour)
            r = b['range']
            g.append(f"<rect x='{sx(r[0]):.1f}' y='{sy(r[3]):.1f}' width='{sx(r[1]) - sx(r[0]):.1f}' height='{sy(r[2]) - sy(r[3]):.1f}' fill='{INK2}' fill-opacity='0.06' stroke='{INK2}' stroke-width='1'/>")
        g.append(f"<text x='{x0 + PANEL / 2}' y='{y0 + PANEL + 30}' font-size='10.5' fill='{INK2}' text-anchor='middle'>Δx（世界系，m）</text>")
        g.append(f"<text transform='translate({x0 - 34},{y0 + PANEL / 2}) rotate(-90)' font-size='10.5' fill='{INK2}' text-anchor='middle'>Δy（m）</text>")
        for d in data:
            g.append(dot(sx(d['x']), sy(d['y']), SERIES[order.index(d['s'])] if order else SERIES[0], f"{d['episode']}：Δx {d['x']:+.4f} m，Δy {d['y']:+.4f} m" + (f"（{series_of[d['s']]}）" if order else '')))
    else:
        vals = [d['x'] for d in data] + [v for b in boxes for v in b['range']] + [v for b in by_series.values() for v in b['range']]
        lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.08 or 1
        lo, hi = lo - pad, hi + pad
        sy = lambda v: y0 + PANEL - (v - lo) / (hi - lo) * PANEL
        for t in ticks(lo, hi, 5):
            g.append(f"<line x1='{x0}' y1='{sy(t):.1f}' x2='{x0 + PANEL}' y2='{sy(t):.1f}' stroke='{GRID}'/><text x='{x0 - 6}' y='{sy(t) + 3.5:.1f}' font-size='10.5' fill='{MUTED}' text-anchor='end'>{fmt(t)}</text>")
        cols = order or [None]
        for bi, b in enumerate(boxes):   # full-width bands, widest last so labels do not overlap
            r = b['range']; inset = 4 + 10 * (len(boxes) - 1 - bi)
            g.append(f"<rect x='{x0 + inset}' y='{sy(r[1]):.1f}' width='{PANEL - 2 * inset}' height='{sy(r[0]) - sy(r[1]):.1f}' fill='{INK2}' fill-opacity='0.06' stroke='{INK2}' stroke-width='1'/>")
        for ci, s in enumerate(cols):
            if s in by_series:
                r = by_series[s]['range']; w = PANEL / len(cols)
                g.append(f"<rect x='{x0 + ci * w + 5:.1f}' y='{sy(r[1]):.1f}' width='{w - 10:.1f}' height='{sy(r[0]) - sy(r[1]):.1f}' fill='{INK2}' fill-opacity='0.06' stroke='{INK2}' stroke-width='1'/>")
        for ci, s in enumerate(cols):
            pts = sorted((d for d in data if d['s'] == s), key=lambda d: d['x'])
            cx = x0 + PANEL * (ci + 0.5) / len(cols)
            for j, d in enumerate(pts):   # deterministic jitter so equal values stay visible
                off = ((j * 7) % 11 - 5) * (PANEL / len(cols) * 0.07)
                g.append(dot(cx + off, sy(d['x']), SERIES[ci], f"{d['episode']}：{d['x']:+.2f}°" + (f"（{series_of[s]}）" if s else '')))
            if s: g.append(f"<text x='{cx:.1f}' y='{y0 + PANEL + 14}' font-size='10.5' fill='{MUTED}' text-anchor='middle'>{html.escape(series_of[s])}</text>")
    texts = [b['text'] for b in boxes] + [f"{series_of.get(s, s)} {b['text']}" for s, b in by_series.items()]
    g.append(f"<text x='{x0}' y='{TOP - 28}' font-size='10.5' fill='{INK2}'>{html.escape(ranges.get('label', '生成器范围：未定位'))}</text>")
    g.append(f"<text x='{x0}' y='{TOP - 13}' font-size='10.5' fill='{INK2}'>{html.escape(('灰框：' + ' / '.join(texts)) if texts else '无范围框')}</text>")
    return ''.join(g)


def figure(key, data, gen):
    field, names, panels = SPEC[key]
    order = [s for s in names if any(d.get(field) == s for d in data)] if field else []
    w = LEFT + len(panels) * PANEL + (len(panels) - 1) * (GAP + 24) + 20
    h = TOP + PANEL + BOTTOM + 26
    body = []
    for i, p in enumerate(panels):
        x0 = LEFT + i * (PANEL + GAP + 24)
        rk = p[-1]; ranges = gen.get(key, {}).get(rk, {})
        if p[0] == 'xy':
            pts = [dict(episode=d['episode'], x=d[p[2]], y=d[p[3]], s=d.get(field)) for d in data]
        else:
            pts = [dict(episode=d['episode'], x=d[p[2]], s=d.get(field)) for d in data]
        body.append(panel(p[0], p[1], pts, names, order, ranges, x0))
    legend = ''
    if order:
        x = LEFT
        for i, s in enumerate(order):
            n = sum(1 for d in data if d.get(field) == s)
            legend += f"<circle cx='{x + 5}' cy='{h - 14}' r='4' fill='{SERIES[i]}'/><text x='{x + 14}' y='{h - 10}' font-size='11.5' fill='{INK2}'>{html.escape(names[s])}（{n} 条）</text>"
            x += 130
        legend += f"<rect x='{x}' y='{h - 20}' width='14' height='11' fill='{INK2}' fill-opacity='0.07' stroke='{INK2}'/><text x='{x + 20}' y='{h - 10}' font-size='11.5' fill='{INK2}'>生成器代码里的采样范围</text>"
    else:
        legend = f"<rect x='{LEFT}' y='{h - 20}' width='14' height='11' fill='{INK2}' fill-opacity='0.07' stroke='{INK2}'/><text x='{LEFT + 20}' y='{h - 10}' font-size='11.5' fill='{INK2}'>生成器代码里的采样范围；点 = 50 条 episode 的实际采样值</text>"
    svg = (f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='{key} 数据方实际采样值与生成器范围' style='max-width:{w}px;background:{SURFACE};font-family:system-ui,-apple-system,\"Segoe UI\",\"PingFang SC\",\"Noto Sans CJK SC\",sans-serif'>"
           f"<style>.pt:hover circle:last-child,.pt:focus circle:last-child{{stroke:{INK};stroke-width:1.5}}.pt{{outline:none}}</style>{''.join(body)}{legend}</svg>")
    cols = [c for c in data[0] if c != 'episode']
    table = ("<details><summary>表格视图：50 条 episode 的逐条采样值</summary><table class='definition rand'><thead><tr><th>episode</th>" + ''.join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr></thead><tbody>"
             + ''.join('<tr><td><code>' + html.escape(d['episode']) + '</code></td>' + ''.join(f"<td>{html.escape(str(d[c]))}</td>" for c in cols) + '</tr>' for d in data) + "</tbody></table></details>")
    return (f"<figure class='cover'>{svg}<figcaption>数据方这批 50 条的实际采样值（点；鼠标悬停看 episode 与数值）与生成器代码里的采样范围（灰框）。位置偏移相对数据方名义摆位"
            f"（咖啡机 (2.5691, 0.1668)、杯子 (2.3601, −0.1002)，世界系 m）。API 侧随机化尚未实现；实现并对齐后在此并列 API 采样分布。"
            f"数据：<code>references/data_side/coverage_rows.json</code>，范围出处：<code>references/data_side/generator_ranges.json</code>。</figcaption>{table}</figure>")


def main():
    data = rows()
    ref = ROOT / 'references/data_side'; ref.mkdir(parents=True, exist_ok=True)
    (ref / 'coverage_rows.json').write_text(json.dumps(dict(source='batch manifests written by the data producer next to the episodes (see tools/select_sim_episodes.py: features)', nominal=NOMINAL, rows=data), indent=1, ensure_ascii=False) + '\n')
    gen_path = ref / 'generator_ranges.json'
    gen = json.loads(gen_path.read_text()) if gen_path.is_file() else {}
    for key in SPEC:
        assert len(data[key]) == 50, (key, len(data[key]))
        out = ROOT / 'preview' / key / 'coverage.html'
        out.write_text(figure(key, data[key], gen))
        print(key, out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
