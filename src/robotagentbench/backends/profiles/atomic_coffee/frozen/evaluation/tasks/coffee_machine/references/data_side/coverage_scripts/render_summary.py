#!/usr/bin/env python3
"""Render coverage_summary.md from coverage_work/stats.json (+ per-episode tables).
All numbers are pulled from stats.json; the prose describes what was inspected in the files."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, SCRATCH  # noqa

ST = json.load(open(os.path.join(OUT_DIR, 'stats.json')))
TABLES = open(os.path.join(OUT_DIR, 'summary_tables.md')).read()


def g(task, *path):
    o = ST[task]
    for p in path:
        o = o[p]
    return o


def rng(s, unit='', nd=None):
    if s is None:
        return 'n/a'
    return f"{s['min']} … {s['max']}{unit} (mean {s['mean']}, sd {s['sd']}, n={s['n']})"


def cn(c):
    return ', '.join(f'{k} {v}' for k, v in c.items())


def row(name, src, s, unit='', note=''):
    if s is None:
        return f'| {name} | {src} | n/a | | | | {note} |'
    return f"| {name} | {src} | {s['n']} | {s['min']} | {s['max']} | {s['mean']} | {note} |"


HDR = '| quantity | source field | n | min | max | mean | note |\n|---|---|---|---|---|---|---|'


def appearance_block(task):
    a = g(task, 'appearance')
    out = []
    out.append(f"- Files: {a['n']}/50 appearance records, policy `{cn(a['policy'])}`; appearance seed == episode seed in {cn(a['seed_equals_episode_seed'])}; {a['distinct_full_appearance_records']} distinct records. `source_snapshot_sha256` constant ({len(a['source_snapshot_sha256'])} value).")
    out.append(f"- Varying keys: {g(task, 'appearance_varying_key_count')} flattened keys vary, {g(task, 'appearance_constant_key_count')} are constant (schema/policy/version strings, prim-path targets, invariants flags, `harmony.status`={cn(a['harmony_status'])}, `physics_effect`, `scope`).")
    out.append(f"- palette / furniture style: {cn(a['palette'])}")
    out.append(f"- environment style + floor texture are tied 1:1 to palette: {cn(a['palette_env_floor_combos'])}")
    out.append(f"- lighting_scheme: {cn(a['lighting_scheme'])}")
    out.append(f"- ceiling lights ({cn(a['ceiling_lights_per_episode'])} per episode): intensity_scale {rng(a['ceiling_intensity_scale'])}; color temperature {rng(a['ceiling_color_temperature_k'], ' K')} (one temperature per episode for the whole group)")
    out.append(f"- daylight lights ({cn(a['daylight_lights_per_episode'])} per episode): intensity_scale {rng(a['daylight_intensity_scale'])}; color temperature {rng(a['daylight_color_temperature_k'], ' K')}")
    fl = a['furniture_looks']
    out.append('- furniture looks (5 living-room items, each with look/rgb_scale/roughness/metallic): ' + '; '.join(f"{k}: {cn(v)}" for k, v in fl.items()))
    out.append(f"- kitchen central table option: {cn(a['central_table_option'])}")
    out.append(f"- kitchen stove rule: {cn(a['stove_rule'])}; lower cabinets: {cn(a['stove_lower'])}; upper cabinets: {cn(a['stove_upper'])}")
    out.append(f"- decor selection recorded in the plan (artwork {cn(a['planned_artwork'])}; lamp mode {cn(a['planned_lamp_mode'])}; {len(a['planned_lamp'])} lamp models; {a['planned_plants_distinct_species']} plant species over 5 slots) — but the file's own `scope` string says: \"{list(a['scope'])[0]}\" → decor values are planned, not applied (not verified in renders).")
    return '\n'.join(out)


def video_block(task):
    v = g(task, 'video')
    out = []
    for k in ('head', 'left', 'right'):
        x = v[k]
        out.append(f"| {k} | {cn(x['resolution'])} | {cn(x['fps'])} | {cn(x['codec'])} | {x['nb_frames']['min']:.0f} … {x['nb_frames']['max']:.0f} (mean {x['nb_frames']['mean']}) | {x['duration_s']['min']} … {x['duration_s']['max']} (mean {x['duration_s']['mean']}) | {x['bytes']['min'] / 1e6:.2f} … {x['bytes']['max'] / 1e6:.2f} | {len(x['missing'])} |")
    return '| view | resolution | fps | codec | frames | duration s | size MB | missing |\n|---|---|---|---|---|---|---|---|\n' + '\n'.join(out) + f"\n\nThree views have identical frame counts per episode: {cn(v['three_views_same_frame_count'])}."


def main():
    o = []
    A = o.append
    A('# Coffee-machine task family 1.1–1.4 — measured data coverage\n')
    A('Read-only survey on server 202, 2026-09-29. Episode list = `data_list.json` records with task 1.1/1.2/1.3/1.4 (50 each, 200 total). '
      'All numbers below are computed from the files on disk by the scripts in `coverage_scripts/` (per-episode rows: `coverage_rows.json`). '
      'Data status labels (`source_status`) are reported as found and not re-judged. Nothing was simulated, rendered or hashed; sha256 values are only those already recorded in manifests.\n')
    A('Conventions: nominal machine = (2.569096825, 0.166750471, 0.92) facing −X; machine yaw = rotation of the machine front (local +Z) about world Z relative to −X; '
      'nominal bag = (2.5659632, −0.1072122, 0.92) (bag position in every trace whose plan has `bag_offset_m`=0), bag local-Z heading = bag_yaw_deg − 8.0° in every episode where both are recorded; '
      'handle yaw = heading of the cup handle axis in world (0–360°). Lengths in metres, angles in degrees.\n')

    A('## Cross-task overview\n')
    A('| | 1.1 放杯子 | 1.2 倾倒咖啡豆 | 1.3 按按钮 | 1.4 取杯子 |\n|---|---|---|---|---|')
    A(f"| dataset dir | coffee_cup_left15_right35_20260925 | coffee_pour_final50_20260924 | coffee_button_v2_20260925 | coffee_retrieve_mixed_original11_20260926 |")
    A(f"| data_list `collection` label | coffee_cup_selected50_20260925 | coffee_pour_20260924 | coffee_button_v2_20260925 | coffee_retrieve50_20260925 |")
    A(f"| episodes in data_list / dirs on disk | 50 / {g('1.1', 'extra', 'n_dirs')} | 50 / {g('1.2', 'extra', 'n_dirs')} | 50 / {g('1.3', 'extra', 'n_dirs')} | 50 / {g('1.4', 'extra', 'n_dirs')} |")
    A(f"| `source_status` | 来源任务自动通过 · 人工待审阅 ×50 | 来源任务自动通过 ×50 | 来源任务自动通过 · 人工待审阅 ×50 | …人工待审阅 ×42, …人工待审阅 · 旧起点保留 ×8 |")
    A(f"| `source_exists_on_202` | false ×50 (source on 182) | true ×50 | false ×50 (path no longer present) | false ×50 (source on 182) |")
    A(f"| arm | {cn(g('1.1', 'arm'))} | {cn(g('1.2', 'arm'))} | {cn(g('1.3', 'arm'))} | {cn(g('1.4', 'arm'))} |")
    A(f"| group labels | {cn(g('1.1', 'group'))} | {cn(g('1.2', 'group'))} | none recorded | {cn(g('1.4', 'group'))} |")
    A(f"| cup style | {cn(g('1.1', 'style'))} | {cn(g('1.2', 'style'))} | not recorded in episode files | {cn(g('1.4', 'style'))} |")
    A(f"| machine dx | {g('1.1', 'all', 'machine_dx')['min']} … {g('1.1', 'all', 'machine_dx')['max']} | {g('1.2', 'all', 'machine_dx')['min']} … {g('1.2', 'all', 'machine_dx')['max']} | {g('1.3', 'machine_world', 'dx_from_nominal')['min']} … {g('1.3', 'machine_world', 'dx_from_nominal')['max']} | {g('1.4', 'all', 'machine_dx')['min']} … {g('1.4', 'all', 'machine_dx')['max']} |")
    A(f"| machine dy | {g('1.1', 'all', 'machine_dy')['min']} … {g('1.1', 'all', 'machine_dy')['max']} | {g('1.2', 'all', 'machine_dy')['min']} … {g('1.2', 'all', 'machine_dy')['max']} | {g('1.3', 'machine_world', 'dy_from_nominal')['min']} … {g('1.3', 'machine_world', 'dy_from_nominal')['max']} | {g('1.4', 'all', 'machine_dy')['min']} … {g('1.4', 'all', 'machine_dy')['max']} |")
    A(f"| machine yaw | {g('1.1', 'all', 'machine_yaw_deg')['min']} … {g('1.1', 'all', 'machine_yaw_deg')['max']} | {g('1.2', 'all', 'machine_yaw_deg')['min']} … {g('1.2', 'all', 'machine_yaw_deg')['max']} | {g('1.3', 'machine_world', 'yaw_from_nominal_deg')['min']} … {g('1.3', 'machine_world', 'yaw_from_nominal_deg')['max']} | {g('1.4', 'all', 'machine_yaw_deg')['min']} … {g('1.4', 'all', 'machine_yaw_deg')['max']} |")
    A(f"| bag yaw | {g('1.1', 'all', 'bag_yaw_deg')['min']} … {g('1.1', 'all', 'bag_yaw_deg')['max']} (16 at nominal) | {g('1.2', 'all', 'bag_yaw_deg')['min']} … {g('1.2', 'all', 'bag_yaw_deg')['max']} | constant {list(g('1.3', 'bag_world', 'yaw_deg(localZ heading + 8)'))[0]} | {g('1.4', 'all', 'bag_yaw_deg')['min']} … {g('1.4', 'all', 'bag_yaw_deg')['max']} ({g('1.4', 'all', 'bag_at_nominal')} at nominal) |")
    A(f"| robot base | root_xyz {list(g('1.1', 'robot', 'root_xyz'))[0]}, yaw 0 — identical in all 200 episodes | same | same | same |")
    A(f"| appearance record | appearance.json ×50 | inside initialization.json ×50 | appearance.json ×50 | absent (seed only) |")
    A(f"| checks | {cn(g('1.1', 'checks', 'n_checks'))} checks, all true ×50 | 19 checks, all true ×50 | 14 checks, all true ×50 | 19 checks, all true ×50 |")
    A(f"| plan length (control steps @20 Hz) | {g('1.1', 'all', 'plan_steps')['min']:.0f} … {g('1.1', 'all', 'plan_steps')['max']:.0f} | {g('1.2', 'all', 'plan_steps')['min']:.0f} … {g('1.2', 'all', 'plan_steps')['max']:.0f} | 641 (all) | {g('1.4', 'all', 'plan_steps')['min']:.0f} … {g('1.4', 'all', 'plan_steps')['max']:.0f} |")
    A(f"| video | 20 fps, {g('1.1', 'video', 'head', 'duration_s')['min']} … {g('1.1', 'video', 'head', 'duration_s')['max']} s | 30 fps, {g('1.2', 'video', 'head', 'duration_s')['min']} … {g('1.2', 'video', 'head', 'duration_s')['max']} s | 30 fps, 32.067 s (all) | 20 fps, {g('1.4', 'video', 'head', 'duration_s')['min']} … {g('1.4', 'video', 'head', 'duration_s')['max']} s |")
    A(f"| video sha256 recorded | dataset_manifest.json (150/150) | none | coffee_updates_0925_verification.json (150/150) | dataset_manifest.json (150/150) |")
    A('\nAll 600 videos exist, h264/yuv420p, head 1280×800, left/right wrist 848×480. Recorded byte sizes in the manifests equal the file sizes on disk for 1.1, 1.3, 1.4 (sha256 not recomputed).\n')

    # ------------------------------------------------------------ 1.1
    s = ST['1.1']
    A('## 1.1 放杯子 — coffee_cup_left15_right35_20260925\n')
    A('### Inventory\n')
    A(f"- 50 episodes in data_list = 50 dirs = 50 `dataset_manifest.json` samples; no extra/missing dirs. Manifest summary: left {s['extra']['manifest_summary']['left']}, right {s['extra']['manifest_summary']['right']}; method: \"{s['extra']['manifest_summary']['method']}\"")
    A(f"- arm × group: {cn(s['arm_x_group'])}; arm × style: {cn(s['arm_x_style'])}.")
    A(f"- Two structurally different populations:\n"
      f"  - **A (16 episodes: 001–015 left + 016 right)** — have `proposal.json`; randomization keys {cn(s['A_randomization_keys'])}; machine and bag at nominal (offset 0, yaw 0); all cream_classic; seeds {s['seeds_A']}.\n"
      f"  - **B (34 episodes: 017–050, all right arm)** — **no `proposal.json`**; randomization lives in `plan.json.randomization` with keys {cn(s['B_randomization_keys'])}; machine, bag and cup style all vary; seeds {s['seeds_B']}.")
    A(f"- Source run families (manifest `name`): {cn(s['source_run_prefix'])}. The 3 `targeted` episodes carry a `study` tag: {s['A_studies']}.")
    A(f"- Seeds: proposal / plan / result / appearance seeds agree in {cn(s['seed_sources_agree'])}; duplicated seeds: {s['duplicate_seeds'] or 'none'}.\n")
    A('### Measured ranges\n')
    A(HDR)
    A(row('handle world yaw, all', 'result.trace[0].handle_axis (= manifest angle)', s['all']['handle_world_yaw_deg']))
    A(row('handle world yaw, left arm', '', s['left']['handle_world_yaw_deg']))
    A(row('handle world yaw, right arm', '', s['right']['handle_world_yaw_deg'], note='016 = 180.0 targeted; others 197.9–249.2'))
    A(row('cup start x', 'result.initial_cup', s['all']['cup_start_x']))
    A(row('cup start y', 'result.initial_cup', s['all']['cup_start_y']))
    A(row('cup start z', 'result.initial_cup', s['all']['cup_start_z'], note='constant (table height)'))
    A(row('cup xy offset x (A)', 'proposal.cup_xy_offset_m', s['A_cup_xy_offset_x'], note='relative to (2.3701, −0.1002)'))
    A(row('cup xy offset y (A)', 'proposal.cup_xy_offset_m', s['A_cup_xy_offset_y']))
    A(row('cup offset x (B)', 'plan.randomization.offset_m', s['B_offset_m_x'], note='relative to (2.3601, −0.1002)'))
    A(row('cup offset y (B)', 'plan.randomization.offset_m', s['B_offset_m_y']))
    A(row('cup yaw_delta (B)', 'plan.randomization.yaw_delta_deg', s['B_yaw_delta_deg'], note='= handle yaw − 225°'))
    A(row('machine dx (B)', 'plan.machine_offset_m = trace', s['B']['machine_dx'], note='A: all 0'))
    A(row('machine dy (B)', '', s['B']['machine_dy'], note='A: all 0'))
    A(row('machine yaw (B)', 'plan.machine_yaw_deg = trace', s['B']['machine_yaw_deg'], note='A: all 0'))
    A(row('bag dx (B)', 'plan.bag_offset_m = trace', s['B']['bag_dx'], note='A: all 0'))
    A(row('bag dy (B)', '', s['B']['bag_dy'], note='A: all 0'))
    A(row('bag yaw (B)', 'plan.bag_yaw_deg', s['B']['bag_yaw_deg'], note='A: all 0'))
    A(row('final cup → goal, xy', 'result.final_cup vs result.goal', s['all']['final_to_goal_xy_m'], note='cup distance to target centre'))
    A(row('final cup → goal, xyz', '', s['all']['final_to_goal_xyz_m']))
    A(row('final z − goal z', '', s['all']['final_minus_goal_z_m'], note='constant'))
    A(row('plan steps, A', 'plan.steps', s['A']['plan_steps']))
    A(row('plan steps, B', 'plan.steps', s['B']['plan_steps']))
    A(row('initial yaw error', 'result.initial_yaw_error_deg', s['checks']['initial_yaw_error_deg']))
    A(f"\n- Angle bins recorded (A only): {cn(s['A_angle_bin'])}. No randomization *range* is recorded in any 1.1 episode file (only the sampled values; A has `angle_bin`).")
    A(f"- Style by population: A {cn(s['A']['style'])}; B {cn(s['B']['style'])}. Left arm is only ever paired with cream_classic, nominal machine and nominal bag.")
    A(f"- Robot: root_xyz/yaw identical; `initial_targets` (20 joints) identical hash in 50/50; result.trace[0].all_joint_positions (34 joints) spread across the 50 episodes = {s['robot']['trace0_joint_spread_max_rad']} rad → initial posture identical. `initial_state_reference` = {cn(s['robot']['initial_state_reference'])}.")
    A(f"- max_bilateral_contact_frames: {cn(s['all']['max_bilateral_contact_frames'])}.\n")
    A('### Appearance (appearance.json)\n')
    A(appearance_block('1.1'))
    A('\n### Checks and outcomes\n')
    A(f"- result.passed true ×50, failure null ×50, unexpected contacts {cn(s['checks']['unexpected_contacts'])}; preflight passed {cn(s['preflight']['passed'])}, violations {cn(s['preflight']['violations'])}, clearance {cn(s['preflight']['clearance_m'])} m, checked_samples {rng(s['preflight']['checked_samples'])}, collision model dir {cn(s['preflight']['model_directory'])}.")
    A(f"- Check count differs by arm: {cn(s['checks']['n_checks_by_arm'])}. The 16 common checks: {', '.join(s['checks']['names_16'])}. Left-arm episodes add: {', '.join(s['checks']['extra_in_17'])}. (016 right has proposal.json but 16 checks.)")
    A(f"- Manifest `checks` field is only the count (16/17), equal to result.json in {cn(s['checks']['manifest_checks_count_equal'])}.\n")
    A('### Episode length\n')
    A(video_block('1.1'))
    A(f"\n- Video frames = plan steps + 3 in {cn(s['plan']['frames_minus_steps'])} (20 fps = one frame per 20 Hz control step). trace length == plan steps {cn(s['plan']['trace_len_equals_plan_steps'])}; trace dt {cn(s['plan']['trace_dt'])} s.")
    A(f"- `capture_events.jsonl` lists only {rng(s['capture_events']['n_events'])} events (every 2nd control step, 10 Hz, first at sim t={cn(s['capture_events']['first_sim_time_s'])} s) — fewer than the video frame count; capture_events mtime precedes the mp4 mtime. Relationship between the event list and the final 20 fps video is not documented in the files (unverified).\n")
    A('### Anomalies\n')
    A('- 34/50 episodes (all of population B) lack `proposal.json`; otherwise all expected files present.')
    A('- Population A has no machine/bag/style variation at all; population B has no left-arm episodes → arm is confounded with every other randomized quantity.')
    A('- data_list `collection` says coffee_cup_selected50_20260925 but the videos/artifacts are under coffee_cup_left15_right35_20260925 (a different directory named coffee_cup_selected50_20260925 also exists with different sample names).')
    A('- Phase list: 46 episodes have 13 phases, 4 episodes skip `lower_pregrasp` (12 phases).\n')
    A('### JSON structure / replayability\n')
    A('- `proposal.json` (A only): seed, handle_world_yaw_deg, cup_xy_offset_m, angle_bin, active_arm (+ `study`, `cup_start_xyz` in targeted ones).')
    A('- `plan.json`: initial_state_reference, initial_manifest_sha256, initial_targets (20 named joints: 2 body, 2 head, 7+7 arm, 2 gripper-outer), root_xyz, yaw, initial (7 q of active arm), **steps[]** = {phase, q[7] active-arm joint positions, tcp[3], opening (gripper rad), bodylift, waist; +rotation 3×3 in B}, cup_style, cup_start_xyz, cup_start_yaw_deg, machine_offset_m, machine_yaw_deg, bag_offset_m, bag_yaw_deg, randomization, geometry (scene path, cup/handle/target/outlet poses, machine contract), head_visibility. → Yes, a joint-space trajectory: 450–715 rows, 7 arm joints + gripper opening + 2 torso values per row. Joint names are implicit (active arm idx21–27 left / idx61–67 right); no per-step timestamp, the 20 Hz rate is evidenced by result.trace `t` increments of 0.05 s and trace length == steps.')
    A('- `executed_plan.json`: byte-for-byte same content as plan.json in 50/50.')
    A('- `result.json` (1.4–2.6 MB): passed, checks, failure, initial_cup, initial_yaw_error_deg, final_cup, goal, max_bilateral_contact_frames, fk_error, **trace[]** (one per control step: t, phase, machine xyz+rotation, cup xyz + up + handle_axis, tcp, bag xyz+rotation, arm/tcp error, contacts, all_joint_positions{34}, opening, q, bodylift, waist), contact_pairs, unexpected_contacts, preflight_sha256, randomization, scope. → Records robot joint positions (34 joints) and machine/cup/bag poses at every step; first sample is t=0.05 s (after the first control step), there is no separate t=0 snapshot.')
    A('- `preflight_certificate.json`: passed, scope, plan_sha256, model_fingerprints, coverage, sources[149], checked_samples, violations, phase_summary, clearance_m, mode, files{68 sha256}, cup_asset{name, sha256}, model_directory.')
    A('- `appearance.json`: version, policy, seed, palette, lighting_scheme, lighting_groups, lights[10], materials[5], plan{environment, kitchen_materials, harmony, randomization_registry…}, kitchen_application, authored_property_paths, scope, source_snapshot_sha256.')
    A('- Replay caveat: scene USD (`/geniesim/main/demo_sim/coffee_task_workbench_20260922/scene_initial.usda`), cup assets and the source run dirs are referenced by container/182 paths and are not in the dataset dir (`source_exists_on_202` false).\n')

    # ------------------------------------------------------------ 1.2
    s = ST['1.2']
    A('## 1.2 倾倒咖啡豆 — coffee_pour_final50_20260924\n')
    A('### Inventory\n')
    A(f"- 50 episodes = 50 dirs = 50 `manifest.json` entries; no extra dirs. Groups: {cn(s['group'])}. Arm: right ×50 (plan q = right arm; left arm held constant). Cup style: {cn(s['style'])}; group × style: {cn(s['group_x_style'])}.")
    A(f"- Source batches (dir prefix): {cn(s['batch_prefix'])}.")
    A(f"- Seeds: initialization/plan/result/appearance/batch-episode seeds agree {cn(s['seed_sources_agree'])}; duplicated seeds: {s['duplicate_seeds'] or 'none'}. Two dirs carry no seed in their name: coffee_wide_machine_2100_v1_episode_0002 (seed 2102; batch seed 2100, third attempt of 3) and diversity_2101_v2_episode_0000 (seed 2101).")
    A(f"- Original run dirs (`local_artifacts[0]`, parent of `motion/`) exist on 202 for 50/50 under: {cn(s['original_run']['roots'])}. They additionally hold `initial/` (appearance.json, initialization.json, initial.usda, bean_load.usda/json, collision_model, shell_override.usda…), and in `motion/`: executed_plan.json, certificate.json, capture_events.jsonl, camera_manifest.json, runner.py, runtime.usda, scene_final.usda, physics_ticks.jsonl.gz, granular/spill diagnostics, PNG frames. `initial/appearance.json` == `initialization.json.appearance` in {cn(s['original_run']['appearance_json_equals_initialization_appearance'])}; original `executed_plan.json` == dataset `plan.json` in {cn(s['original_run']['executed_plan_equals_dataset_plan'])}.\n")
    A('### Measured ranges\n')
    A(HDR)
    for label, key in (('all 50', 'all'), ('小扰动 (29)', 'group:小扰动'), ('大扰动 (21)', 'group:大扰动')):
        b = s[key]
        A(row(f'machine dx — {label}', 'randomization.machine_offset_m', b['machine_dx']))
        A(row(f'machine dy — {label}', '', b['machine_dy']))
        A(row(f'machine yaw — {label}', 'randomization.machine_yaw_deg', b['machine_yaw_deg']))
        A(row(f'bag dx — {label}', 'randomization.bag_offset_m', b['bag_dx']))
        A(row(f'bag dy — {label}', '', b['bag_dy']))
        A(row(f'bag yaw — {label}', 'randomization.bag_yaw_deg', b['bag_yaw_deg']))
        A(row(f'beans in hopper — {label}', 'result.bean_outcome.in_hopper', b['beans_in_hopper'], note=cn(b['bean_outcome_counts']) + ' (hopper/bag/outside)'))
        A(row(f'plan steps — {label}', 'plan.steps', b['plan_steps']))
    b = s['all']
    A(row('machine world x', 'initialization.machine_world', b['machine_world_x'], note='z constant 0.92'))
    A(row('machine world y', '', b['machine_world_y']))
    A(row('bag world x', 'initialization.bag_world', b['bag_world_x'], note='z constant 0.92'))
    A(row('bag world y', '', b['bag_world_y']))
    A(row('cup world x', 'initialization.cup_world', b['cup_world_x'], note='cup sits on machine tray, follows machine'))
    A(row('cup world y', '', b['cup_world_y']))
    A(row('cup world z', '', b['cup_world_z'], note='constant'))
    A(row('bag return position error', 'result.bag_return.position_error_m', s['checks']['bag_return_position_error_m'], note='plan tolerance 0.01'))
    A(row('bag return rotation error (rad)', 'result.bag_return.rotation_error_rad', s['checks']['bag_return_rotation_error_rad'], note='plan tolerance 0.0873'))
    A(row('lift position drift', 'result.lift_position_drift_m', s['checks']['lift_position_drift_m']))
    A(row('recorded_overlap_events', 'result', s['checks']['recorded_overlap_events']))
    A(row('spill_contact_events', 'result', s['checks']['spill_contact_events']))
    A(row('max_bilateral_contact_frames', 'result', s['checks']['max_bilateral_contact_frames']))
    A(f"\n- Consistency: `machine_world`/`bag_world` matrices reproduce nominal + recorded offsets (machine xyz err ≤ {s['world_matches_offsets']['machine_xyz_max_abs_err_m']:.1e} m, yaw err ≤ {s['world_matches_offsets']['machine_yaw_max_abs_err_deg']:.1e}°, bag xy err ≤ {s['world_matches_offsets']['bag_xy_max_abs_err_m']:.1e} m); randomization block identical in initialization/result/manifest {cn(s['randomization_equal_init_result_manifest'])}.")
    A('- Randomization ranges recorded in the data:')
    A('  - `batch.json.random_ranges` of the original run dir, by group | batch | ranges:')
    for k, v in s['group_x_batch_prefix_x_batchjson_ranges'].items():
        A(f'    - {k} → {v} episodes')
    A(f"  - `initialization.json.bag_cycle_initialization.ranges`: {cn(s['initialization_bag_cycle_ranges'])} (same value in all 50, including 小扰动).")
    A('  - Measured per source batch:')
    for bname in s['batch_prefix']:
        bb = s['batch:' + bname]
        A(f"    - {bname} (n={bb['n']}): machine dx {bb['machine_dx']['min']} … {bb['machine_dx']['max']}, dy {bb['machine_dy']['min']} … {bb['machine_dy']['max']}, yaw {bb['machine_yaw_deg']['min']} … {bb['machine_yaw_deg']['max']}; bag dx {bb['bag_dx']['min']} … {bb['bag_dx']['max']}, dy {bb['bag_dy']['min']} … {bb['bag_dy']['max']}, yaw {bb['bag_yaw_deg']['min']} … {bb['bag_yaw_deg']['max']}")
    A(f"- Bag variant/size: single variant in 50/50 — {list(s['bag_geometry'])[0]}. Bag position z constant 0.92.")
    A(f"- Beans: bean_paths {cn(s['beans']['bean_paths'])}, bean_count {cn(s['beans']['bean_count'])}, bean_scale {cn(s['beans']['bean_scale'])} (plan.bean_geometry_scale {cn(s['beans']['plan_bean_geometry_scale'])}), 3 bean assets (a/b/c); initially in bag {cn(s['beans']['initial_in_bag'])}. Hopper revision {cn(s['machine_hopper_revision'])}.")
    A(f"- Side-grasp parameters [policy, height_m, angle_deg, side, roll_deg, depth_m] vary: {cn(s['plan']['side_grasp_height_angle_depth'])}.")
    A(f"- Robot: root_xyz/yaw identical (49 plans + 1 executed plan); plan `initial_targets` hash identical; `initialization.json.robot_targets` (34 joints) identical hash in 50/50.\n")
    A('### Appearance (initialization.json → appearance; identical to original-run initial/appearance.json)\n')
    A(appearance_block('1.2'))
    A('\n### Checks and outcomes\n')
    A(f"- result.passed true ×50, failure null ×50, manifest checks == result checks {cn(s['checks']['manifest_checks_equal'])}, unexpected contacts {cn(s['checks']['unexpected_contacts'])}. initialization.checks = {cn(s['checks']['initialization_checks'])}.")
    A(f"- 19 checks: {', '.join(s['checks']['names'])}.")
    A(f"- Bean outcome (in_hopper/in_bag/outside): {cn(s['all']['bean_outcome_counts'])}. `strict_pour_quality.all_beans_in_hopper`: {cn(s['beans']['strict_pour_quality_all_beans_in_hopper'])}. result.acceptance_scope = \"{list(s['checks']['acceptance_scope'])[0]}\". So `passed`/`beans_entered_hopper` are true in all 50 while 30 episodes spill 1–6 beans.")
    A(f"- Original-run `certificate.json` passed: {cn(s['original_run']['certificate_passed'])} (null = file absent for diversity_2101_v2). batch.json attempted/physical_successes: {cn(s['original_run']['batch_json_attempted_vs_success'])}.\n")
    A('### Episode length\n')
    A(video_block('1.2'))
    A(f"\n- video_manifest: actual_render_fps {cn(s['video_manifest']['actual_render_fps'])}, synchronized {cn(s['video_manifest']['synchronized'])}, frame list length == ffprobe frames {cn(s['video_manifest']['n_frames_equals_ffprobe'])}, simulation_seconds {rng(s['video_manifest']['simulation_seconds'])} == plan steps / 20 Hz in {cn(s['video_manifest']['sim_seconds_equals_steps_over_20'])}. trace length == plan steps {cn(s['plan']['trace_len_equals_plan_steps'])}. No sha256 of the videos is recorded in manifest.json or video_manifest.json.\n")
    A('### Anomalies\n')
    A('- `diversity_2101_v2_episode_0000`: dataset dir has **no plan.json** (original run has executed_plan.json but no plan.json/certificate.json, and no initial/initial.usda); numbers for this episode use the original executed_plan.json (759 steps).')
    A('- Range metadata is inconsistent with the group label: the 11 `bag_large_extra_gpu*` episodes are labelled 大扰动 and reach |machine xy| 0.0996 m / |yaw| 18.8°, but their batch.json says machine_xy_m 0.05 / yaw 5°; conversely every initialization.json says 0.1 / 20° even for 小扰动. Measured 小扰动 values stay within ±0.05 m / ±5°; measured 大扰动 within ±0.1 m / ±20°.')
    A('- Bag xy range is the same (±0.05 m) in both groups; only the machine perturbation differs between 小扰动 and 大扰动. Bag yaw is always positive (3.5–88.9°), never 0.')
    A('- data_list `collection` = coffee_pour_20260924 while files are under coffee_pour_final50_20260924.')
    A('- 30/50 episodes have 1–6 beans outside the hopper (see above) — reported, not re-judged.\n')
    A('### JSON structure / replayability\n')
    A('- `initialization.json`: stage, scope, seed, randomization{machine_offset_m, machine_yaw_deg, bag_offset_m, bag_yaw_deg, cup_style}, cup_placement_policy, bean_paths[100], bean_assets[3], bean_local[24], machine_contract, **machine_world / bag_world / cup_world** (4×4), **robot_targets** (34 joints), checks, passed, bag_geometry_revision, appearance (full appearance record), machine_hopper_revision, bag_cycle_initialization{ranges}.')
    A('- `plan.json` (0.9 MB): stage, scope, initial_directory, root_xyz, yaw, initial_targets(20), initial(7, right arm), grasp 4×4, loaded/hold opening, **steps[]** = {q[7] right arm, left_q[7] (constant), bodylift, waist, opening, phase, tcp, rotation}, side_grasp, bag_cycle{return_world, tolerances}, bean_count, bean_geometry_scale, video_capture_control_stride/fps, randomization, seed, bag_collision_variant{sha256 of 11 initial files}, motion_timing{control_hz 20, per-phase seconds}, intermediate_hold_removal. → Joint-space trajectory: 703–927 rows at 20 Hz (`motion_timing.control_hz`), 12 phases pregrasp_bag … settle.')
    A('- `result.json` (≈13 MB): passed, checks, failure, **initial** / **final** {bag, cup, gripper_base 4×4, pad_poses, beans[100] xyz, in_bag/in_hopper/outside}, trace[] per control step {bag, cup, gripper_base, pad_poses, beans[100], counts, phase, contacts, pad impulses, gripper_joints(8), actual_arm[7], opening, arm_error, bodylift, waist}, bag_return, bean_outcome, strict_pour_quality, overlap/spill counters, seed, randomization. → Object poses incl. all 100 bean positions are recorded at every step; robot state is recorded only for the right arm + right gripper + torso (no 34-joint dict; initial full-body posture is in initialization.json.robot_targets). Machine pose is not in result.json (it is static; see initialization.machine_world).')
    A('- `video_manifest.json`: source, control_stride, simulation_seconds, views{head,left,right → video, frames[{file, simulation_time}]}, synchronized, actual_render_fps.')
    A('- Replay inputs available on 202 for 49/50: original `initial/initial.usda` + `motion/runtime.usda` + `motion/runner.py` + executed_plan.json (paths inside refer to /geniesim/main/... container mounts; not executed here, unverified).\n')

    # ------------------------------------------------------------ 1.3
    s = ST['1.3']
    A('## 1.3 按按钮 — coffee_button_v2_20260925\n')
    A('### Inventory\n')
    A(f"- 50 episodes = 50 dirs = 50 `manifest.json` episodes (batch `{s['extra']['manifest_top']['batch']}`, target 50, passed 50, finished 50, status {s['extra']['manifest_top']['status']}, GPUs {s['extra']['manifest_top']['gpus']}). No missing files, no extra dirs.")
    A(f"- Arm: right ×50. No group label. Seeds (dir name = proposal = manifest = initialization = appearance, {cn(s['seed_sources_agree'])}): {s['seeds']}; duplicates: {s['duplicate_seeds'] or 'none'}. Seeds are a non-contiguous subset of 5001–5121 (`proposal_policy`: \"one uniform proposal per seed; retain all rejections; no resampling\" → the gaps are seeds that are not in this batch).")
    A(f"- manifest `source_run` differs from `run` in {cn(s['manifest']['source_run_differs_from_run'])} (e.g. button_batch_seed5001_v1 → button_two_press_seed5001_v2).\n")
    A('### Measured ranges\n')
    A(HDR)
    A(row('proposal machine_dx_m', 'proposal.json', s['proposal']['machine_dx_m'], note='claimed ±0.15 (proposal.config.machine_xy_m)'))
    A(row('proposal machine_dy_m', 'proposal.json', s['proposal']['machine_dy_m'], note='claimed ±0.15'))
    A(row('proposal machine_dyaw_deg', 'proposal.json', s['proposal']['machine_dyaw_deg'], note='claimed ±30 (proposal.config.machine_yaw_deg)'))
    A(row('machine dx vs nominal', 'initialization.machine_world', s['machine_world']['dx_from_nominal'], note='= proposal + reference offset −0.005154'))
    A(row('machine dy vs nominal', '', s['machine_world']['dy_from_nominal'], note='= proposal + reference offset +0.044716'))
    A(row('machine yaw vs nominal', '', s['machine_world']['yaw_from_nominal_deg'], note='= proposal + reference yaw −0.2553'))
    A(row('machine world x', '', s['machine_world']['x']))
    A(row('machine world y', '', s['machine_world']['y']))
    A(row('button world x', 'initialization.button_world', s['button_world']['x']))
    A(row('button world y', '', s['button_world']['y']))
    A(row('button world z', '', s['button_world']['z'], note='constant'))
    A(row('cup world x', 'initialization.cup_world', s['cup_world']['x'], note='cup follows machine'))
    A(row('cup world y', '', s['cup_world']['y']))
    A(row('press displacement, deepest', 'min over result.trace[].status.button_displacement_m', s['checks']['press_displacement_min_m(deepest)'], note='plan contract press_limit −0.005'))
    A(row('press displacement at rest', 'max over trace', s['checks']['press_displacement_max_m(rest)']))
    A(row('displacement at "started" event', 'result.events[0]', s['checks']['event_started_displacement_m']))
    A(row('displacement at "stopped_by_button"', 'result.events[1]', s['checks']['event_stopped_displacement_m']))
    A(row('time of start event (s)', 'result.events[0].time_s', s['checks']['event_started_time_s']))
    A(row('time of stop event (s)', 'result.events[1].time_s', s['checks']['event_stopped_time_s']))
    A(row('fluid particles emitted', 'result.particles', s['checks']['particles_final'], note=cn(s['checks']['particles_final_counts'])))
    A(row('control steps with button_pressed', 'trace', s['checks']['pressed_control_steps']))
    A(row('control steps dispensing', 'trace', s['checks']['dispensing_control_steps']))
    A(row('button contact records', 'result.button_contacts', s['checks']['button_contacts']))
    A(row('certificate samples', 'button_certificate.samples', s['certificate']['samples']))
    A(f"\n- The proposal is applied on top of a reference initialization, not on the nominal machine pose: implied reference = ({s['machine_world']['implied_reference_x']['mean']}, {s['machine_world']['implied_reference_y']['mean']}), yaw {s['machine_world']['implied_reference_yaw_deg']['mean']}° — identical in 50/50 and equal to the randomization of `coffee_fast_side_canary_2000_v1/episode_0000/initial/initialization.json` (seed 2000: machine_offset (−0.005154, 0.044716), yaw −0.2553°, bag_offset (0.026823, 0.029090), bag_yaw 84.069°, cup_style blue_compact), which is `resize_source_initial` of all 50 episodes.")
    A(f"- Bag: **constant** in 50/50 — world xyz {list(s['bag_world']['xyz'])[0]}, yaw {list(s['bag_world']['yaw_deg(localZ heading + 8)'])[0]}°, offset from nominal bag {s['bag_world']['offset_from_nominal_bag']}; geometry {list(s['bag_world']['geometry'])[0]}.")
    A(f"- Cup: position follows the machine (\"{list(s['cup_world']['placement_policy'])[0]}\"), z constant {s['cup_world']['z']['mean']}. **Cup style is not recorded in any JSON of the 50 episode dirs** (no occurrence of cream_classic/blue_compact/green_tall); the shared reference initialization says blue_compact → inferred constant, unverified per episode.")
    A(f"- Beans: bean_paths {cn(s['beans']['bean_paths'])}, bean_count {cn(s['beans']['bean_count'])}, bean_scale {cn(s['beans']['bean_scale'])}; beans_initially_in_hopper {cn(s['beans']['beans_initially_in_hopper'])}, in bag {cn(s['beans']['beans_initially_in_bag'])}; bean_count during the whole trace {cn(s['beans']['trace_bean_count_values'])}.")
    A(f"- Approach route yaw (`horizontal_route.yaw`): {cn(s['route']['yaw'])}; nodes {cn(s['route']['n_nodes'])}; vertical_travel constant {s['route']['vertical_travel']['mean']} m.")
    A(f"- Robot: root_xyz/yaw, plan `initial_targets` and `initialization.robot_targets` (34 joints) identical in 50/50 (same hashes as 1.2).")
    A(f"- Machine yaw coverage is asymmetric: proposal dyaw spans −19.96 … +29.84 (nothing below −20°) although ±30° is claimed.\n")
    A('### Appearance (appearance.json; identical to initialization.json.appearance in 50/50)\n')
    A(appearance_block('1.3'))
    A(f"- result.json additionally records a constant fluid look `appearance_only_changes` = {{diffuse [0.3, 0.105, 0.027], roughness 0.12, isosurface_distance 0.0018}} in 50/50.")
    A('\n### Checks and outcomes\n')
    A(f"- result.passed true ×50, failure null ×50; manifest status {cn(s['checks']['manifest_status'])}; manifest checks/events == result.json {cn(s['checks']['manifest_checks_equal'])}; unexpected contacts {cn(s['checks']['unexpected'])}; button_certificate passed {cn(s['certificate']['passed'])}, commands {cn(s['certificate']['commands'])}.")
    A(f"- 14 checks: {', '.join(s['checks']['names'])}.")
    A(f"- Events in every episode: {cn(s['checks']['events'])} (first press starts dispensing, 10 s wait, second press stops). bean_count at both events {cn(s['checks']['event_bean_count'])}.")
    A('- Liquid volume *in the cup* is not recorded — only the number of emitted fluid particles.\n')
    A('### Episode length\n')
    A(video_block('1.3'))
    A(f"\n- Every episode has the same length: plan steps {cn(s['plan']['n_steps'])}, control {cn(s['plan']['control_hz'])} Hz, physics {cn(s['plan']['physics_hz'])} Hz, simulation_seconds {cn(s['checks']['simulation_seconds'])}; video 962 frames / 32.067 s @30 fps; capture_events.jsonl 963 lines (frame 962 = final snapshot excluded from the video per videos_ready.json). Phase lengths are identical in 50/50: {list(s['plan']['phases'])[0]}.\n")
    A('### Anomalies\n')
    A('- None of the 50 has missing files or a false check; no duplicated seeds.')
    A('- Only the machine pose (3 numbers) and appearance vary; bag pose, bean state, robot posture, episode duration and phase timing are constant; cup style unrecorded.')
    A('- `source_exists_on_202` is false although `source_path` names a 202 path (/home/lingjun/coffee-workspace/render-workers/gpuN/…/runs/button_two_press_seedNNNN_v2) — the dataset copy has no initial.usda / runner / collision model.')
    A('- `initialization.json.stage` is "pour_beans" and `bag_cycle_initialization.ranges` says machine_xy 0.05 / yaw 5° (inherited from the pour fixture), which differs from the button proposal config (0.15 / 30°).\n')
    A('### JSON structure / replayability\n')
    A('- `proposal.json`: seed, machine_dx_m, machine_dy_m, machine_dyaw_deg, config{version, machine_xy_m 0.15, machine_yaw_deg 30.0, first_seed 5000, attempts 10, reference, physics_hz 240, control_hz 20, capture_fps 30, appearance, proposal_policy}.')
    A('- `initialization.json`: same layout as 1.2 plus beans_initially_in_hopper/in_bag, button_world, button_up, reference_initialization, resize_source_initial, randomization_config, head_visibility; randomization = {seed, machine_dx_m, machine_dy_m, machine_dyaw_deg}; machine_world/bag_world/cup_world 4×4; robot_targets (34 joints). `bean_local` has 24 entries for 100 bean_paths, i.e. per-bean initial positions in the hopper are not listed.')
    A('- `button_plan.json` (0.3 MB): root_xyz, yaw, initial_targets(20), initial(7, right arm), task, **steps[641]** = {q[7], opening, phase, button_travel, tcp, bodylift, waist}, physics_hz 240, control_hz 20, scope, two_press_wait_seconds 10, horizontal_height. → Joint-space trajectory, 641 rows at 20 Hz (explicit `control_hz`).')
    A('- `button_certificate.json`: passed, samples, commands, fingerprints{26 sha256 incl. button_plan.json, initial.usda, initialization.json, collision model}. `horizontal_route.json`: yaw, nodes[98]{q, phase, travel, tcp 4×4}, opening, initial, horizontal_height, vertical_travel.')
    A('- `motion/result.json` (0.3 MB): passed, failure, checks, trace[641]{step, time_s, phase, status{bean counts, button_pressed, button_displacement_m, dispensing, event}, arm_error, fixed_error, particles}, events[2], button_contacts, unexpected, particles, simulation_seconds, video_capture_fps, appearance_only_changes. → **No joint positions and no object poses in result.json**; initial state must come from initialization.json, commanded trajectory from button_plan.json.\n')

    # ------------------------------------------------------------ 1.4
    s = ST['1.4']
    A('## 1.4 取杯子 — coffee_retrieve_mixed_original11_20260926\n')
    A('### Inventory\n')
    A(f"- data_list 50 = `dataset_manifest.json` samples 50 (status `{s['manifest_top']['status']}`, counts {s['manifest_top']['counts']}, full_initial_count {s['manifest_top']['full_initial_count']}, legacy_initial_count {s['manifest_top']['legacy_initial_count']}); directory holds {s['extra']['n_dirs']} run dirs.")
    A(f"- **Dirs present but not in data_list / manifest (4):** {', '.join(s['extra']['dirs_not_in_data_list'])}.")
    for e in s['extra_dirs']:
        A(f"  - {e['dir']}: {e['arm']} arm, {e['style']}, {e['plan_steps']} steps, removal_ready {e['removal_ready_steps']} step(s), gripper opening at start {e['trace0_opening_rad']} rad, plan.initial standard: {e['plan_initial_q_is_standard']}, result passed {e['result_passed']} / all checks true {e['all_checks_true']}, video {e['video_s']} s; same machine pose/handle yaw as batch member **{e['batch_member_with_same_original_selected_id']}** whose manifest `original_selected_id` is this dir → these are the original (old-start) runs of the four repaired samples that were kept in repaired form.")
    A(f"- manifest `user_decision`: \"{s['manifest_top']['user_decision']}\"; `replacement_receipt` lists 11 from→to pairs; `storage`: \"{s['manifest_top']['storage']}\" (on 202 the dirs are real directories whose files have hard-link count 2).")
    A(f"- arm × group: {cn(s['arm_x_group'])}; arm × style: {cn(s['arm_x_style'])}. `active_arm` is written in plan.json for only {sum(v for k, v in s['arm_recorded_in_plan'].items() if k != 'null')} episodes ({cn(s['arm_recorded_in_plan'])}); for the rest the arm comes from manifest/queue.json.")
    A(f"- Group definition in the files: {cn(s['group_x_plan_bounds'])}. Machine-pose source: {cn(s['group_x_has_retrieval_pose_randomization'])} (\"no\" = machine/bag pose inherited unchanged from the cup-placing run the episode continues from).")
    A(f"- Seeds: `plan.randomization.seed` is the seed of the preceding cup-placing run; duplicated placing seeds (same placing run reused with a different retrieval pose): {s['seeds']['placing_seed(plan.randomization.seed)_duplicates']}. `retrieval_pose_randomization.seed` present in {cn(s['seeds']['retrieval_pose_seed_present'])}, duplicates: {s['seeds']['retrieval_pose_seed_duplicates'] or 'none'}. `pose_proposal.json` exists only in: {', '.join(s['seeds']['pose_proposal_json_present'])}.\n")
    A('### Measured ranges\n')
    A(HDR)
    for label, key in (('all 50', 'all'), ('regular (33)', 'regular'), ('large (17)', 'large'), ('left (25)', 'left'), ('right (25)', 'right')):
        b = s[key]
        A(row(f'machine dx — {label}', 'plan.machine_offset_m = manifest = trace', b['machine_dx'], note='large: claimed ±0.15' if key == 'large' else ''))
        A(row(f'machine dy — {label}', '', b['machine_dy']))
        A(row(f'machine yaw — {label}', 'plan.machine_yaw_deg', b['machine_yaw_deg'], note='large: claimed ±30' if key == 'large' else ''))
        A(row(f'initial handle yaw — {label}', 'trace[0].handle_axis = manifest', b['initial_handle_world_yaw_deg']))
    b = s['all']
    A(row('initial cup x', 'result.initial_cup', b['initial_cup_x'], note='cup on machine tray'))
    A(row('initial cup y', '', b['initial_cup_y']))
    A(row('initial cup z', '', b['initial_cup_z'], note='constant'))
    A(row('initial cup → tray target, xy', 'result.initial_cup vs plan.geometry.cup_target', b['initial_cup_to_tray_target_xy_m']))
    A(row('bag dx', 'plan.bag_offset_m', b['bag_dx'], note=f"{b['bag_at_nominal']} episodes at nominal (all 25 left + 1 right)"))
    A(row('bag dy', '', b['bag_dy']))
    A(row('bag yaw', 'plan.bag_yaw_deg', b['bag_yaw_deg']))
    A(row('final cup → goal, xyz', 'result.final_cup vs result.goal', b['final_to_goal_xyz_m']))
    A(row('final TCP → chest goal', 'trace[-1].tcp vs plan.chest_tcp_goal', b['final_tcp_to_chest_goal_m']))
    A(row('cup lift (final z − initial z)', '', b['cup_lift_dz_m']))
    A(row('cup displacement', '', b['cup_displacement_m']))
    A(row('final cup x', 'result.final_cup', b['final_cup_x']))
    A(row('final cup y', '', b['final_cup_y']))
    A(row('final cup z', '', b['final_cup_z']))
    A(row('plan steps — all', 'plan.steps', b['plan_steps']))
    A(row('plan steps — full-initial 42', '', s['full42']['plan_steps']))
    A(row('plan steps — 旧起点 8', '', s['legacy8']['plan_steps']))
    A(row('removal_ready steps — full 42', 'first phase length', s['full42']['first_phase(removal_ready)_steps']))
    A(row('removal_ready steps — 旧起点 8', '', s['legacy8']['first_phase(removal_ready)_steps']))
    A(f"\n- Liquid: `liquid_report.json` identical in 50/50 — initial count 10304 / inside 10304 / fraction 1.0, final 10304 / 10304 / 1.0; method \"{list(s['liquid']['method'])[0]}\".")
    A(f"- Consistency: plan vs manifest vs trace machine pose agree (≤ {s['consistency']['plan_vs_trace_machine_offset_max_abs']:.1e} m, ≤ {s['consistency']['plan_vs_trace_machine_yaw_max_abs_deg']:.1e}°); manifest handle yaw vs trace ≤ {s['consistency']['manifest_vs_trace_handle_yaw_max_abs_deg']:.1e}°; manifest `actual_initial_cup` == result.initial_cup (≤ {s['consistency']['manifest_vs_result_initial_cup_max_abs_m']:.1e} m).")
    A(f"- Robot base root_xyz/yaw identical; torso bodylift {s['robot']['trace0_bodylift']['mean']} / waist {s['robot']['trace0_waist']['mean']} identical; non-active arm at the standard pose in 50/50 (max joint diff ≤ {s['all']['trace0_other_arm_max_joint_diff_from_standard_rad']['max']} rad).\n")
    A('### The 8 旧起点 (legacy initial) episodes vs the other 42\n')
    A('| quantity at first trace sample | 42 full-initial | 8 旧起点 |\n|---|---|---|')
    f, l = s['full42'], s['legacy8']
    A(f"| manifest initial_state_status | approved_full_initial | legacy_post_placement_initial_unrepaired |")
    A(f"| active-arm joints vs standard initial pose (max abs diff, rad) | {f['trace0_active_arm_max_joint_diff_from_standard_rad']['min']} … {f['trace0_active_arm_max_joint_diff_from_standard_rad']['max']} | {l['trace0_active_arm_max_joint_diff_from_standard_rad']['min']} … {l['trace0_active_arm_max_joint_diff_from_standard_rad']['max']} |")
    A(f"| plan.initial == standard arm pose | {cn(f['plan_initial_q_is_standard'])} | {cn(l['plan_initial_q_is_standard'])} |")
    A(f"| gripper opening (rad) | {f['trace0_gripper_opening_rad']['min']} … {f['trace0_gripper_opening_rad']['max']} | {l['trace0_gripper_opening_rad']['min']} … {l['trace0_gripper_opening_rad']['max']} |")
    A(f"| TCP distance to cup (m) | {f['trace0_tcp_to_cup_m']['min']} … {f['trace0_tcp_to_cup_m']['max']} | {l['trace0_tcp_to_cup_m']['min']} … {l['trace0_tcp_to_cup_m']['max']} |")
    A(f"| `removal_ready` phase length (steps) | {f['first_phase(removal_ready)_steps']['min']:.0f} … {f['first_phase(removal_ready)_steps']['max']:.0f} | {l['first_phase(removal_ready)_steps']['min']:.0f} … {l['first_phase(removal_ready)_steps']['max']:.0f} |")
    A(f"| plan steps | {f['plan_steps']['min']:.0f} … {f['plan_steps']['max']:.0f} (mean {f['plan_steps']['mean']}) | {l['plan_steps']['min']:.0f} … {l['plan_steps']['max']:.0f} (mean {l['plan_steps']['mean']}) |")
    A(f"| video duration (s) | {f['video_duration_s']['min']} … {f['video_duration_s']['max']} (mean {f['video_duration_s']['mean']}) | {l['video_duration_s']['min']} … {l['video_duration_s']['max']} (mean {l['video_duration_s']['mean']}) |")
    A(f"| plan.initial_prefix_repair present | {cn(f['has_initial_prefix_repair'])} | {cn(l['has_initial_prefix_repair'])} |")
    A(f"| initial_arm_fixture.json present | {cn(f['has_initial_arm_fixture_json'])} | {cn(l['has_initial_arm_fixture_json'])} |")
    A(f"| plan.initial_targets hash | {len(f['initial_targets_hash'])} distinct (same as 1.1–1.3) | {len(l['initial_targets_hash'])} distinct (one per episode) |")
    A(f"| arm / group / style | {cn(f['arm'])}; {cn(f['group'])}; {cn(f['style'])} | {cn(l['arm'])}; {cn(l['group'])}; {cn(l['style'])} |")
    A(f"| machine dx / dy / yaw | {f['machine_dx']['min']} … {f['machine_dx']['max']} / {f['machine_dy']['min']} … {f['machine_dy']['max']} / {f['machine_yaw_deg']['min']} … {f['machine_yaw_deg']['max']} | {l['machine_dx']['min']} … {l['machine_dx']['max']} / {l['machine_dy']['min']} … {l['machine_dy']['max']} / {l['machine_yaw_deg']['min']} … {l['machine_yaw_deg']['max']} |")
    A('\nIn words: the 42 start with the active arm in the same home posture as 1.1–1.3 with the gripper open (0.781 rad) and execute a recorded approach prefix (`removal_ready`, 70–279 steps); the 8 旧起点 start with the active arm already at the post-placement pose next to the cup (0.14–0.20 m from it), gripper almost closed (0.175–0.18 rad), and `removal_ready` lasts only 1–58 steps. Object state (machine, cup on tray, liquid, bag) is of the same kind in both sets.\n')
    A('| 旧起点 episode | arm | group | style | supersedes | steps | ready steps | opening | TCP→cup m | arm diff rad | fixture json | evidence_checks all true | video s |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for e in s['legacy_detail']:
        A(f"| {e['episode']} | {e['arm']} | {e['group']} | {e['style']} | {e['supersedes']} | {e['plan_steps']} | {e['removal_ready_steps']} | {e['trace0_opening_rad']} | {e['trace0_tcp_to_cup_m']} | {e['active_arm_max_joint_diff_from_standard_rad']} | {e['has_initial_arm_fixture_json']} | {e['evidence_checks_all_true']} | {e['video_s']} |")
    A('\n7 of the 8 carry `supersedes` + `evidence_checks` + user_decision "Use original pre-prefix trajectory; no trimming or new physical claims."; cup_retrieve50_right_0029 has none of these fields (it was never replaced). None of the 8 has `initial_repair_audit`.\n')
    A('### Appearance\n')
    A(f"- **No appearance.json in any of the 54 dirs.** Every `runner.py` calls `apply_appearance(stage, seed=plan.randomization.seed)` and writes appearance.json to the run output dir, but that file was not copied to 202. Only the seed is known: {s['appearance_seed_used_by_runner']['n_distinct']} distinct appearance seeds over 50 episodes (7 seeds used twice). 24 episodes (21 seeds) share their seed and source placing run with a 1.1 episode whose appearance.json exists; whether 1.4 rendered the same look is not verifiable from files on 202.\n")
    A('### Checks and outcomes\n')
    A(f"- result.passed true ×50, failure null ×50, status.json passed ×50 with identical checks, manifest physical_checks == result checks {cn(s['checks']['manifest_physical_checks_equal'])}, unexpected contacts {cn(s['checks']['unexpected_contacts'])}, preflight passed {cn(s['preflight']['passed'])} with {cn(s['preflight']['violations'])} violations.")
    A(f"- 19 checks: {', '.join(s['checks']['names'])}.")
    A(f"- Manifest audit fields: {cn(s['checks']['manifest_initial_repair_audit'])}.\n")
    A('### Episode length\n')
    A(video_block('1.4'))
    A(f"\n- Video frames = plan steps + 3 in {cn(s['plan']['frames_minus_steps'])}; trace length == plan steps {cn(s['plan']['trace_len_equals_plan_steps'])}; trace dt {cn(s['plan']['trace_dt'])} s (20 Hz). capture_events.jsonl: {rng(s['capture_events']['n_events'])} events, first event phase/step/sim-time {cn(s['capture_events']['first'])}. Manifest durations equal ffprobe durations.\n")
    A('### Anomalies\n')
    A('- 4 extra run dirs on disk (listed above); batch membership must be taken from data_list.json / dataset_manifest.json.')
    A('- 8 旧起点 episodes start from a different robot state (table above).')
    A(f"- Manifest `actual_final_cup` differs from result.json `final_cup` for the 4 repaired samples: {s['consistency']['manifest_vs_result_final_cup_mismatch_gt_0.01mm']} (max abs component diff, m). For the other 46 they are equal.")
    A(f"- `executed_plan.json` != `plan.json` in 50/50: steps are identical in all 50, but `continued_from` differs (host vs container path) and in 5 旧起点 episodes `initial_targets` differs in one gripper joint (0.18 vs 0.174).")
    A('- Missing optional files: appearance.json 50/50; pose_proposal.json 47/50; queue.json 4/50; initial_arm_fixture.json 4/50 (all 旧起点).')
    A('- plan.json schema is heterogeneous (12 different runner.py hashes across the 54 dirs; keys such as active_arm, retrieval_pose_randomization, perturbation_group, initial_prefix_repair, approach_shortcut_repair, local_route_repair, grip_retry, extract_yaw_adjustment_deg exist only in subsets).')
    A('- Left arm is only paired with cream_classic and a nominal bag; blue_compact/green_tall and bag perturbation occur only with the right arm. Group `large` has 15 cream_classic, 1 blue_compact, 1 green_tall.')
    A('- data_list `collection` = coffee_retrieve50_20260925 while files are under coffee_retrieve_mixed_original11_20260926.\n')
    A('### JSON structure / replayability\n')
    A('- `pose_proposal.json` (3 episodes only): seed, (proposal_index), machine_offset_m, machine_yaw_deg, source, scope. `queue.json`: source, arm, out, root, gpu, container, run, code, source_alias, previous_run.')
    A('- `plan.json` (0.5 MB): same core as 1.1 (initial_state_reference, initial_manifest_sha256, initial_targets(20), root_xyz, yaw, initial(7), **steps[]** = {phase, q[7], tcp, opening, bodylift, waist, rotation; +seconds in 24 episodes}, cup_style, cup_start_xyz/yaw of the placing run, machine_offset_m, machine_yaw_deg, bag_offset_m, bag_yaw_deg, randomization (of the placing run), geometry, head_visibility) plus task "cup_removal", continued_from, liquid_initialization, removal_goal, chest_tcp_goal and the optional repair/randomization blocks. → Joint-space trajectory: 498–827 rows at 20 Hz (rate evidenced by trace dt 0.05 s), 8 phases removal_ready → reapproach → regrasp → pickup_lift → pickup_hold → extract → carry_out → hold_outside.')
    A('- `executed_plan.json`: same steps as plan.json (see anomaly on continued_from / initial_targets).')
    A('- `result.json` (≈2.3 MB): same layout as 1.1 — passed, checks, initial_cup, final_cup, goal, trace[] with machine/cup/bag poses, tcp and all_joint_positions{34} per control step, contact_pairs, unexpected_contacts, randomization, continued_from. → Initial robot joints and object poses are recorded (first sample t=0.05 s). Liquid particle positions are not recorded (counts only, in liquid_report.json).')
    A('- `preflight_certificate.json`: as 1.1. `status.json`: status, checks, failure. `liquid_report.json`: initial/final {count, inside, fraction}, method. `runner.py`, `runtime.usda` (sublayers the placing run\'s scene_final.usda by container path) and `scene_final.usda` are included in each dir; the placing-run scene they reference is on 182, not on 202.\n')

    A('## Appendix — per-episode tables\n')
    A('Full machine-readable rows (all fields incl. checks, joint dicts, ffprobe results, sha256 from manifests): `coverage_rows.json`.\n')
    A(TABLES)
    with open(os.path.join(SCRATCH, 'coverage_summary.md'), 'w') as fh:
        fh.write('\n'.join(o) + '\n')
    print('written', os.path.join(SCRATCH, 'coverage_summary.md'), sum(len(x) for x in o))


if __name__ == '__main__':
    main()
