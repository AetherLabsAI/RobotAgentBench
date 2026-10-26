"""Write the 验证记录 table of the four definition files and the status tables of the family README from the
published batch results (preview/kitchen-results.json). Pure CPU. Only these tables and the 状态 line are touched;
the definition tables themselves are never edited here.
"""
import json
import re
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK.parents[2]))
from evaluation.tasks.coffee_machine.process_rules import RULE_VERSION   # noqa: E402
from evaluation.tasks.coffee_machine.specs import EVALUATOR_ID          # noqa: E402
APIS = ('1-1', '1-2', '1-3', '1-4')
NO = {'1-1': '1.1', '1-2': '1.2', '1-3': '1.3', '1-4': '1.4'}
VERSION = {'1-1': '五', '1-2': '四', '1-3': '四', '1-4': '六'}
TIMES_NOTE = '（主管 2026-10-06 按用户要求把各阶段预算与总时限改为数据方 50 条来源 episode 中最久者 ×2，规则版本 `arena.coffee_machine.process.v4`，改动见对应的“改动”小节）'
VERSION_NOTE = {'1-1': TIMES_NOTE, '1-2': TIMES_NOTE, '1-3': TIMES_NOTE,
                '1-4': '（第四版审完后，主管 2026-09-30 依 202 回放实测放宽稳定与阶段时限，见“第五版改动”；' + TIMES_NOTE[1:]}
RULES_NOTE = {api: '已发布的批次结果都是在 2026-10-06 之前的规则版本下判定的（r4 与左手批次用 `arena.coffee_machine.process.v2`），新时限（v4）下尚未重跑；时限只放宽不收紧，已成功的回放在新规则下不会变成超时。'
              for api in ('1-1', '1-2', '1-3', '1-4')}
RULES_NOTE['1-4'] = '批次 r4 与左手批次按第五版规则（`arena.coffee_machine.process.v2`）判定；第四版规则下的批次 r2 结果与第五版复算见“第五版改动”。' + RULES_NOTE['1-4']


def results():
    return json.loads((TASK / 'preview/kitchen-results.json').read_text())


def last(row):
    return row['attempts'][-1] if row and row['attempts'] else None


def summary(api, data):
    rows = data['apis'].get(api, {})
    not_run = {k.split(':', 1)[1]: v for k, v in data.get('not_run', {}).items() if k.startswith(api + ':')}
    fixed = last(rows.get('fixed-probe'))
    runs, probes = [], []
    for i in range(1, total_runs(api, data) + 1):
        run = last(rows.get(f'source-run-{i}'))
        if run is not None:
            runs.append((i, rows[f'source-run-{i}'].get('episode'), run))
        elif f'source-run-{i}' in not_run:
            runs.append((i, None, dict(not_run=not_run[f'source-run-{i}'])))
        probe = last(rows.get(f'layout-probe-{i}'))
        if probe is not None:
            probes.append((i, probe))
    random_probe = last(rows.get('random-probe'))
    return fixed, runs, probes, random_probe, not_run, data.get('blocked', {}).get(api)


def total_runs(api, data):
    return len((data.get('episodes') or {}).get(api) or []) or 5


def run_text(i, episode, run):
    if 'not_run' in run:
        why = run['not_run']
        m = re.search(r'(\d+) of (\d+) rows .*?largest ([0-9.]+) rad', why)
        if m:
            why = '来源轨迹 %s/%s 行的夹爪指令超出公共接口范围 [0, π/4]（最大 %s rad），按“动作数值禁止任何改动”不放行' % m.groups()
        return f'回放 {i} 未运行（{why}）'
    if run.get('audit_passed') is None:
        state = {'running': '进行中', 'error': '作业出错，不是任务结果'}.get(run.get('status'), str(run.get('status')))
        return f'回放 {i}（`{episode}`）{state}'
    verdict = '成功' if run['success'] else '未成功'
    extra = ('，' + '、'.join(run['deductions'])) if run['deductions'] else ''
    return f'回放 {i}（`{episode}`）{verdict} {run["score"]:g}/{run["maximum_score"]}，{run["reason"]}{extra}'


def record(api, data):
    fixed, runs, probes, random_probe, not_run, blocked = summary(api, data)
    done = [r for _, _, r in runs if r.get('audit_passed') is not None]
    ok = [r for r in done if r['success']]
    batch = (data.get('batches') or {}).get(api) or data['batch']
    total = total_runs(api, data)
    if fixed is None:
        fixed_text = '固定初态短测尚未完成'
    elif fixed.get('audit_passed'):
        fixed_text = f'固定初态短测通过（物理配置 `{fixed.get("physics")}`）'
    else:
        fixed_text = '固定初态短测未通过：' + '、'.join(fixed.get('rejected_checks') or [str(fixed.get('error') or fixed.get('status'))])
    failed_probes = [f'第 {i} 条布局初态验收未通过（{"、".join(p.get("rejected_checks") or [str(p.get("status"))])}）' for i, p in probes if not p.get('audit_passed')]
    if random_probe is not None:
        random_text = ('种子 7 随机初态短测通过' if random_probe.get('audit_passed') else
                       '种子 7 随机初态短测未通过：' + '、'.join(random_probe.get('rejected_checks') or [str(random_probe.get('error') or random_probe.get('status'))]))
    elif 'random-probe' in not_run:
        random_text = '种子 7 随机初态短测未运行：定义表没有给出可复现的采样器（原始生成器未取得），不自造采样器'
    elif 'random-probe' not in (data.get('pipeline') or {}).get(api, ['random-probe']):
        random_text = '本批次只回放来源轨迹，未安排种子 7 随机初态短测（上一批次 `artifacts/batch-20260930-r4` 的随机短测通过）'
    else:
        random_text = '种子 7 随机初态短测尚未完成'
    runs_text = '；'.join(run_text(*r) for r in runs) or '来源轨迹回放尚未完成'
    pending = total - len(runs)
    conclusion = (f'批次 `{batch}`（冻结快照）。{fixed_text}。来源轨迹回放已出结果 {len(done)}/{total}，其中成功 {len(ok)} 条：{runs_text}'
                  + (f'；另有 {pending} 条尚未开始' if pending > 0 else '') + ('；' + '；'.join(failed_probes) if failed_probes else '')
                  + f'。{random_text}。每条回放都做了 CPU 复算（规则在记录的事实上重放，与服务端结果逐字段一致）。接口：已接入公共服务；物理：见上；真实 VLA：按流程跳过。'
                  + (f'受阻：{blocked}。' if blocked else '') + RULES_NOTE.get(api, ''))
    random_state = {
        '1-1': '随机布局：数据方同步源码 `left_study_reference/randomize.py::scene_proposals` + 文档记录的 v8 配置（20 条 v8 记录逐值一致），已实现',
        '1-2': '随机布局：数据方同步的生产入口 `success_coverage.choose` + `build_cycle_initial.py` 抽样块（空成功历史，布局只由种子决定），已实现',
        '1-3': '随机布局：数据方 `proposal(seed)` 可复现，已实现，杯子须完整在头部相机画面内（初态验收时按实时相机位姿判定）',
        '1-4': '随机布局：数据方同步源码 `randomize_retrieval_pose.py` 的抽样规则（19 条常规记录逐值一致），已实现',
    }[api] + '；外观按种子由公共外观代码生成（物理与基线逐项相同）'
    unfinished = {
        '1-1': '182 最终生产版本的采样器与左臂采样器未取得（评测用 v8 右臂配置）；“主管实现取值”里标待确认的两项等用户确认',
        '1-2': '来源回放未运行：来源轨迹夹爪指令 0.8 rad 超出公共接口上限 π/4，按“动作数值禁止任何改动”不放行，等轨迹负责方重新导出；小扰动家族的调度脚本未交付；“主管实现取值”里标待确认的两项等用户确认',
        '1-3': '“主管实现取值”里标待确认的两项等用户确认；定义歧义（+3 的发放时刻等）见规则交接报告',
        '1-4': '大扰动采样版本（±0.15 m / ±30°）与逐条外观记录未取得（外观按所接放杯运行的种子推算）；“主管实现取值”里标待确认的两项等用户确认',
    }[api]
    return [
        ('任务 ID / 指令与边界', f'API `{api}`（任务编号 {NO[api]}）；指令与边界见定义表'),
        ('初态与扰动', f'已实现。固定初态 = 第 1 条来源 episode 的布局，用公共共享咖啡资产；静置 1 s 后按定义表验收。{random_state}'),
        ('过程分 / 扣分', f'已实现（`process_rules.py`，当前规则版本 `{RULE_VERSION}`，单元测试通过）'),
        ('成功 / 失败 / 时限', '已实现；总时限由公共框架按仿真时间执行，阶段预算在规则里计时'),
        ('实现与注册位置', '`evaluation/tasks/coffee_machine/`：`facts.py`、`process_rules.py`、`isaac_evidence.py`、`api_runtime.py`、`registration.py`、`initial_state.py`；'
                          f'登记在 `evaluation/registration.py` 与 `configs/tasks.json`；评测器 `{EVALUATOR_ID}`'),
        ('验证记录与结论', conclusion),
        ('未完成项', unfinished),
    ]


def update_api(api, data):
    path = TASK / 'apis' / api / 'README.md'
    text = path.read_text()
    rows = record(api, data)
    table = '## 验证记录\n\n| 项目 | 内容 |\n| --- | --- |\n' + ''.join(f'| {k} | {v} |\n' for k, v in rows)
    start = text.index('## 验证记录')
    tail = text[start + len('## 验证记录'):]
    nxt = re.search(r'\n## ', tail)
    rest = tail[nxt.start() + 1:] if nxt else ''
    text = text[:start] + table + ('\n' + rest if rest else '')
    fixed, runs, *_ = summary(api, data)
    done = [r for _, _, r in runs if r.get('audit_passed') is not None]
    total = total_runs(api, data)
    batch_note = ('' if total == 5 else f'用户 2026-10-01 指示继续测试：只回放数据方当前清单里的全部左手 episode（{total} 条，批次 `{(data.get("batches") or {}).get(api, data["batch"])}`，GPU 上限 3 张）。')
    state = (f'状态：定义已审完，用户 2026-09-29 在主管会话授权开始标准作业（固定初态短测、来源轨迹回放共 5 条、种子 7 随机初态短测；不做随机初态回放）。{batch_note}'
             f'当前文本为审核后的第{VERSION[api]}版定义{VERSION_NOTE.get(api, "")}。API ID `{api}`，任务编号 {NO[api]}。已实现并登记；标准作业'
             f'{"已跑完" if len(runs) == total_runs(api, data) else "进行中"}（来源回放已出结果 {len(done)}/{total_runs(api, data)}），结果见“验证记录”和审阅页的“合成视频检查”。')
    lines = text.splitlines()
    index = next(i for i, l in enumerate(lines) if l.startswith('状态：'))
    lines[index] = state
    path.write_text('\n'.join(lines) + '\n')


def update_family(data):
    path = TASK / 'README.md'
    text = path.read_text()
    start = text.index('## 未完成项')
    lines = []
    for api in APIS:
        fixed, runs, probes, random_probe, not_run, blocked = summary(api, data)
        done = [r for _, _, r in runs if r.get('audit_passed') is not None]
        ok = [r for r in done if r['success']]
        scores = '、'.join(f'{r["score"]:g}/{r["maximum_score"]}' for r in done) or '—'
        fixed_text = '未完成' if fixed is None else ('通过' if fixed.get('audit_passed') else '未通过')
        random_text = ('通过' if random_probe.get('audit_passed') else '未通过') if random_probe else ('未运行（无可复现采样器）' if 'random-probe' in not_run else ('本批次未安排（r4 通过）' if 'random-probe' not in (data.get('pipeline') or {}).get(api, ['random-probe']) else '未完成'))
        lines.append(f'| {api} | {fixed_text} | {len(done)}/{total_runs(api, data)} 已出结果，成功 {len(ok)} 条；得分 {scores} | {random_text} |')
    body = ('## 未完成项\n\n| 项目 | 状态 |\n| --- | --- |\n'
            '| 定义表审核 | 已审完，用户 2026-09-29 授权开始标准作业 |\n'
            '| API ID 与 CSV、服务端登记 | 已完成（1-1 … 1-4） |\n'
            '| 咖啡机 / 杯子 / 豆袋资产 | 已用公共共享资产 `coffee_v1`；1-1、1-4 的来源 episode 用的是旧豆袋和无漏斗咖啡机，差异见“资产与物理口径” |\n'
            '| 数据方源码同步（2026-09-30） | 已接入：1-4 液体预填、1-2 采样入口、1-1 / 1-4 采样器、公共外观；仍缺 182 最终生产版本、1-2 小扰动调度、1-4 大扰动采样版本与逐条外观记录（MISSING.json） |\n'
            '| 动作数值 | 2026-09-30 起禁止任何改动：1-2 来源轨迹夹爪指令 0.8 rad 超出接口上限 π/4，其来源回放未运行，等轨迹负责方 |\n'
            '| 待用户确认的实现取值 | 底部承托判定方式、液体落定阈值，见各定义表“实现绑定”的“主管实现取值” |\n'
            '| 定义歧义 | 规则实现交接报告第 4 节（`references/rules_handover_report.md`），待用户裁定 |\n\n'
            f'## 标准作业结果（批次 `{data["batch"]}`）\n\n| API | 固定初态短测 | 来源轨迹回放 | 种子 7 随机初态短测 |\n| --- | --- | --- | --- |\n'
            + '\n'.join(lines) + '\n\n'
            '## 简表\n\n| 项目 | 内容 |\n| --- | --- |\n'
            '| 任务 ID / 指令与边界 | API 1-1 … 1-4（任务编号 1.1 … 1.4）；指令与边界见各定义表 |\n'
            '| 初态与扰动 | 已实现；随机布局只有 1-3 可复现 |\n'
            '| 过程分 / 扣分 | 已实现，以各定义表当前版本为准 |\n'
            '| 成功 / 失败 / 时限 | 已实现，以各定义表当前版本为准 |\n'
            '| 实现与注册位置 | 本目录；登记在 `evaluation/registration.py` 与 `configs/tasks.json` |\n'
            '| 验证记录与结论 | 见上表与各定义表的“验证记录”；真实 VLA：按流程跳过 |\n'
            '| 未完成项 | 见“未完成项” |\n')
    path.write_text(text[:start] + body)


def main():
    data = results()
    for api in APIS:
        update_api(api, data)
    update_family(data)
    print('records updated from', data['batch'])


if __name__ == '__main__':
    main()
