"""Task identity catalogue shared by the review-site builders.

Two CSVs at the repo root are the only sources:
  data_requirement.csv  — "Task N" families (一类任务) and their 子任务ID / 原子动作描述 / 对象;
  VLA_task.csv          — 任务编号 (== 子任务ID) → API 任务 ID + 任务描述.
Every review page shows, for each subtask, 【Task N 家族名 · 子任务 x.y · API id · 任务编号 x.y · 任务描述】.
"""
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# family -> existing directory under evaluation/tasks (None: no directory yet; add when the family is taken on)
FAMILY_DIR = {1: None, 2: 'air_fryer', 3: 'shelf_pick + coffee_table_place', 4: 'tabletop_pick + small_table_pick (+ tidy_sofa 6-x, 10-3)',
              5: 'personal_item_pick + basket_place', 6: 'tidy_sofa (+ hang_hat, apis/15-2)', 7: 'trash_disposal + tabletop_pick + small_table_pick', 8: None,
              9: 'refrigerator_take + microwave_put', 10: 'microwave_take + dining_place + coffee_table_place (30-x)'}
# API id (regex) -> preview dir relative to the site root (each family's preview/ is symlinked next to the hub index.html)
API_PAGE_DIRS = {r'^4-\d+$': 'shelf_pick', r'^(5-\d+|30-\d+)$': 'coffee_table_place', r'^7-\d+$': 'tabletop_pick', r'^11-\d+$': 'small_table_pick',
                 r'^13-[1235]$': 'personal_item_pick', r'^14-[1235]$': 'basket_place', r'^18-\d+$': 'trash_disposal', r'^26-\d+$': 'refrigerator_take',
                 r'^(9\.2|27-\d+)$': 'microwave_put', r'^10\.1$': 'microwave_take', r'^(10\.2|29-\d+)$': 'dining_place', r'^17-1$': 'hang_hat',
                 r'^20-2$': 'book_cloth',
                 r'^air_fryer': 'air_fryer'}
# data_requirement 子任务 -> runtime task id when VLA_task.csv has no API id for it, or the runtime uses a different id
SUBTASK_RUNTIME_ID = {'2.1': 'air_fryer_load_no_sugar', '2.2': 'air_fryer_load_sugar', '2.3': 'air_fryer_pull_sugar', '2.4': 'air_fryer_pull_no_sugar',
                      '7.2': '18-1', '7.4': '18-2', '7.8': '18-3', '7.10': '18-4', '9.2': '9.2', '10.1': '10.1', '10.4': '10.1', '10.2': '10.2'}
# runner-config id -> server registration id (air fryer variants share one registration each)
# 手工单任务页（没有 data-api 分页）：目录 -> 该页对应的任务 ID
SINGLE_PAGE_API = {'microwave_put': '9.2'}
RUNNER_TO_SERVER = {'air_fryer_load_no_sugar': 'air_fryer_load', 'air_fryer_load_sugar': 'air_fryer_load', 'air_fryer_pull_sugar': 'air_fryer_pull', 'air_fryer_pull_no_sugar': 'air_fryer_pull'}


def _rows(path):
    with open(path, newline='', encoding='utf-8') as f: return list(csv.reader(f))


def vla_rows():
    """任务编号 -> dict(api, description, note) in CSV order."""
    rows = _rows(REPO / 'VLA_task.csv'); head = rows[0]
    i_no, i_desc, i_api, i_note = head.index('任务编号'), head.index('任务描述'), head.index('API 任务 ID'), head.index('备注')
    out = {}
    for r in rows[1:]:
        if not r or not r[0].strip(): continue
        out[r[i_no].strip()] = dict(api=r[i_api].strip(), description=r[i_desc].strip(), note=r[i_note].strip() if len(r) > i_note else '')
    return out


def families():
    """Task N -> dict(id, name, slug, owner, dir, subtasks=[dict(sub, action, target, need, have, api, description, note)])."""
    rows = _rows(REPO / 'data_requirement.csv'); vla = vla_rows()
    fams, current = {}, None
    for r in rows[3:]:
        if not any(x.strip() for x in r): continue
        head = r[0].strip()
        m = re.match(r'^Task (\d+)\s*(?:\(([^)]*)\))?[：:]\s*(.*?)\s*(?:\(([^)]*)\))?\s*[|｜]\s*family \d+ (\S+)', head)
        if m:
            n = int(m.group(1)); owner = m.group(2) or m.group(4) or ''
            current = fams[n] = dict(id=n, name=m.group(3).strip(), slug=m.group(5), owner=owner, dir=FAMILY_DIR[n], subtasks=[])
            continue
        if '小计' in head or '总计' in head: current = None; continue
        if current is None or not re.match(r'^\d+\.\d+$', head): continue
        v = vla.get(head, {})
        current['subtasks'].append(dict(sub=head, action=r[3].strip(), target=r[4].strip(), need=r[8].strip(), have=r[9].strip(),
                                        api=v.get('api', ''), description=v.get('description', r[3].strip()), note=v.get('note', '')))
    return fams


def api_index():
    """API id -> list of (family, subtask) tuples in CSV order (shared APIs appear once per 任务编号)."""
    out = {}
    for fam in families().values():
        for s in fam['subtasks']:
            if s['api']: out.setdefault(s['api'], []).append((fam, s))
    return out


def identity_items(api):
    """['Task 6 整理沙发 · 子任务 6.5 · API 16-1 · 任务编号 6.5 · 将账单放入茶几抽屉', ...] for one API."""
    return [f"Task {fam['id']} {fam['name']} · 子任务 {s['sub']} · API {api} · 任务编号 {s['sub']} · {s['description']}" for fam, s in api_index().get(api, [])]


def identity_html(api):
    import html as _h
    items = identity_items(api)
    if not items: return f"<p class='identity'>【API {_h.escape(api)}：VLA_task.csv 中没有对应任务编号】</p>"
    return "<p class='identity'>" + ' '.join(f"【{_h.escape(x)}】" for x in items) + "</p>"


def runtime_id(sub, api):
    """Server/runtime task id for a data_requirement subtask (CSV API id, or the alias table when the CSV has none)."""
    return SUBTASK_RUNTIME_ID.get(sub) or api or ''


def api_page_href(api):
    """Site-root-relative link to the API's detail page, or None when no page exists yet."""
    if not api: return None
    tasks = REPO / 'evaluation/tasks'
    for pat, d in API_PAGE_DIRS.items():
        if not re.match(pat, api): continue
        pdir = tasks / {'hang_hat': 'tidy_sofa/hang_hat', 'book_cloth': 'tabletop_pick/book_cloth'}.get(d, d) / 'preview'
        page = pdir / 'index.html'
        if (pdir / api).is_dir(): return f'{d}/index.html#{api}'
        # 只有页面里真的有这个任务的分页时才给链接（例如 shelf_pick 的 4-15 尚未建页）
        if SINGLE_PAGE_API.get(d) == api and page.exists(): return f'{d}/index.html'
        if page.exists() and f"data-api='{RUNNER_TO_SERVER.get(api, api)}'" in page.read_text(): return f'{d}/index.html#{RUNNER_TO_SERVER.get(api, api)}'
    for fam, s in api_index().get(api, []):
        if fam['id'] == 6 and (tasks / 'tidy_sofa/preview' / api).is_dir(): return f"tidy_sofa/index.html#{api}"
    return None


if __name__ == '__main__':
    for fam in families().values():
        print(f"Task {fam['id']} {fam['name']} ({fam['slug']}, {fam['owner']}) -> {fam['dir']}: {len(fam['subtasks'])} subtasks")
        for s in fam['subtasks']: print('   ', s['sub'], s['api'] or '—', s['description'])
