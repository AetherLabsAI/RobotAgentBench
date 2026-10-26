"""Local orchestrator: deterministic placement then actual CPU USD build on 202."""
import json
import subprocess
import sys
import uuid
import fcntl
from pathlib import Path
from room_request import resolve

HERE=Path(__file__).resolve().parent
REMOTE='/home/lingjun/robot_agent_arena/openend/tasks/scene_requirements'
KEY='/Users/dylan/.ssh/id_ed25519_hlk'
HOST='lingjun@64.62.194.202'
NODE='/Users/dylan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
SSH=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10',HOST]
SCP=['scp','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10']
SERVER_MODE=str(HERE)==REMOTE

def node_command(input_file=None):
    if SERVER_MODE:
        extra=[]
        if input_file is not None and HERE not in input_file.resolve().parents:
            extra=['-v',str(input_file.resolve())+':'+str(input_file.resolve())+':ro']
        return ['docker','run','--rm','--network=none','-v',str(HERE)+':'+str(HERE)+':ro',*extra,
                'node:22-bookworm-slim','node']
    return [NODE]

def cpu_command(remote):
    launcher='/home/lingjun/robot_agent_arena/api-protocol/evaluation/tools/run_cpu_usd.sh'
    if SERVER_MODE:return ['bash',launcher,str(HERE/'build_selected_room.py'),remote+'/input.json']
    return SSH+[f'bash {launcher} {REMOTE}/build_selected_room.py {remote}/input.json']

def fetch(remote,name,folder):
    if not SERVER_MODE:run(SCP+[HOST+':'+remote+'/'+name,str(folder/name)])

def run(cmd,timeout=120):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError((p.stderr or p.stdout)[-5000:])
    return p.stdout

def choose_plan(resolved,folder):
    layouts=json.loads((HERE/'placement_sources/random-layouts-v1.json').read_text())['layouts']
    attempts=[]
    for lid in resolved['layout_candidates']:
        resolved['layout']=next(l for l in layouts if l['id']==lid)
        resolved['layout']['station_obstacles']=[x for x in resolved.get('station_obstacle_feedback',[]) if x['layout_id']==lid]
        resolved['request']['layout_id']=lid
        resolved['requirements']['station_request']['layout_id']=lid
        available={s['group'] for s in resolved['layout']['surfaces']}
        missing=[t['placement']['surface_group'] for t in resolved['request']['tasks']
                 if t.get('placement',{}).get('surface_group') is not None
                 and t['placement']['surface_group'] not in available]
        if missing or resolved['layout']['errors']:
            attempts.append(dict(layout_id=lid,status='required_surface_missing_or_geometry_unavailable',missing_surface_groups=missing))
            (folder/'layout_attempts.json').write_text(json.dumps(attempts,indent=2))
            continue
        inp=folder/'resolved.json';inp.write_text(json.dumps(resolved))
        p=subprocess.run(node_command(inp)+[str(HERE/'room_plan.cjs'),str(inp)],capture_output=True,text=True,timeout=120)
        if p.returncode not in (0,2):
            # Invalid user constraints or a code error are not placement failures.
            raise ValueError(p.stderr[-3000:])
        value=json.loads(p.stdout)
        attempts.append(dict(layout_id=lid,status=value['plan']['status'],
                             candidate_counts=value['plan']['candidate_counts']))
        (folder/'layout_attempts.json').write_text(json.dumps(attempts,indent=2))
        if p.returncode==0:
            value['layout_selection'].update(selected=lid,rerolls=len(attempts)-1,attempts=attempts)
            return value
    raise ValueError('候选布局均未找到满足约束的完整摆法；失败记录已保留')

def _generate(request,progress,excluded=(),prior_attempts=(),feedback=()):
    resolved=resolve(request)
    resolved['station_obstacle_feedback']=list(feedback)
    resolved['layout_candidates']=[x for x in resolved['layout_candidates'] if x not in excluded]
    if not resolved['layout_candidates']:raise ValueError('所有候选布局均未满足场景约束；失败记录已保留')
    ident='room_'+uuid.uuid4().hex[:12]
    folder=HERE/'generated_rooms'/ident;folder.mkdir(parents=True,exist_ok=False)
    (folder/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
    progress(dict(stage='plan',message='选择布局并检查摆放约束'))
    planned=choose_plan(resolved,folder)
    attempts=list(prior_attempts)+planned['layout_selection']['attempts']
    planned['layout_selection'].update(attempts=attempts,rerolls=len(attempts)-1)
    (folder/'input.json').write_text(json.dumps(planned,ensure_ascii=False))
    progress(dict(stage='compose',message='202 CPU：合成实际房间 USD，提取场景概览'))
    remote=f'{REMOTE}/generated_rooms/{ident}'
    if not SERVER_MODE:
        run(SSH+['mkdir -p '+remote])
        run(SCP+[str(folder/'input.json'),HOST+':'+remote+'/input.json'])
    # Deploy only task-local code. Source assets and shared entry points stay untouched.
    code=['build_selected_room.py','station_geometry_v2.json','station_scene_checks.py','descendant_transforms.py','build_living.py','build_fridge.py','build_shelf.py','shelf_sampler.py','shelf_inventory_policy.py','shelf_binding.py','cloth_placement.py','task_choices.py','workspace_views.py','render_room.py','render_quality.py','render_reuse.py','launch_render.py','gpu_selection.py','gpu1_reservation.py','gpu_policy.json']
    if not SERVER_MODE:
        run(SCP+[str(HERE/f) for f in code]+[HOST+':'+REMOTE+'/'])
        run(SCP+[str(HERE/'gpu1_signal_helper.py'),HOST+':'+REMOTE+'/gpu1_signal_helper.py'])
    p=subprocess.run(cpu_command(remote),capture_output=True,text=True,timeout=300)
    (folder/'build.log').write_text(p.stdout+p.stderr)
    if p.returncode:
        detail=(p.stderr or p.stdout)[-1800:]
        if 'Station scene overlap:' in detail:
            fetch(remote,'station_clearance.json',folder)
            failed=json.loads((folder/'station_clearance.json').read_text())
            lid=planned['layout']['id']
            additions=[dict(layout_id=lid,station=f['station'],path=f['obstacle'],min=f['obstacle_bounds'][0],max=f['obstacle_bounds'][1],evidence_episode=ident) for f in failed['failures']]
            existing={(x['layout_id'],x['station'],x['path']) for x in feedback}
            additions=[x for x in additions if (x['layout_id'],x['station'],x['path']) not in existing]
            attempts[-1].update(status='rejected_actual_scene_overlap',failed_episode=ident)
            rounds=sum(a.get('status')=='rejected_actual_scene_overlap' and a['layout_id']==lid for a in attempts)
            if additions and rounds<3:
                progress('实际几何复核发现占位冲突；保留失败，用该障碍约束重新选位（种子不变）')
                return _generate(request,progress,excluded,attempts,tuple(feedback)+tuple(additions))
            if request.get('layout_id') is None:
                progress('此布局未找到无重叠摆法，按原规则换下一个布局')
                return _generate(request,progress,tuple(set(excluded)|{lid}),attempts,feedback)
        layout_mismatch=any(s in detail for s in ('Furniture geometry differs from approved sampler:',
                           'Furniture transform must remain rigid and upright:', 'Drawer initial transform incompatible'))
        if request.get('layout_id') is None and layout_mismatch:
            attempts[-1].update(status='scene_geometry_incompatible',failed_episode=ident,error=detail.splitlines()[-1])
            progress('此布局的家具不兼容已标定摆放区域，自动尝试下一个布局')
            skipped=tuple(set(excluded)|{x['layout_id'] for x in attempts})
            return _generate(request,progress,skipped,attempts,feedback)
        raise RuntimeError('USD 合成失败，记录已保留：'+detail)
    for name in ('composition.json','room.usda','runtime_request.json'):
        fetch(remote,name,folder)
    result=json.loads((folder/'composition.json').read_text())
    progress(dict(stage='gpu',message='场景已保留；准备专用 GPU1'))
    from render_stream import stream
    command=['python3','-B',str(HERE/'launch_render.py'),remote] if SERVER_MODE else SSH+[f'python3 -B {REMOTE}/launch_render.py {remote}']
    stream(command,folder/'render.log',progress)
    progress(dict(stage='publish',message='下载渲染结果并校验文件'))
    if not SERVER_MODE:run(SCP+['-r',HOST+':'+remote+'/renders_v2',str(folder)+'/'])
    fetch(remote,'render_gpu.json',folder)
    result['render']=json.loads((folder/'renders_v2/manifest.json').read_text())
    result['render']['directory']='renders_v2'
    result['render']['gpu']=json.loads((folder/'render_gpu.json').read_text())
    import hashlib
    receipt=result['render']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if receipt['scene_sha256']!=result['scene_sha256'] or digest(folder/'room.usda')!=receipt['scene_sha256']:
        raise ValueError('Rendered scene hash mismatch')
    if not receipt['images']:raise ValueError('No usable rendered images')
    for item in receipt['images']:
        if Path(item['file']).name!=item['file'] or digest(folder/'renders_v2'/item['file'])!=item['sha256']:
            raise ValueError('Rendered image hash mismatch')
    serialized=json.dumps(result,ensure_ascii=False,indent=2)
    (folder/'composition.json').write_text(serialized)
    if not SERVER_MODE:run(SCP+[str(folder/'composition.json'),HOST+':'+remote+'/composition.json'])
    tmp=HERE/'current_room.next.json';tmp.write_text(serialized);tmp.replace(HERE/'current_room.json')
    warnings=receipt.get('warnings',[])
    progress(dict(stage='done',message=('渲染结束，部分视角可能存在问题；尚未启动物理' if warnings else '真实房间渲染完成；尚未启动物理'),warnings=warnings,valid_images=len(receipt['images']),completed=receipt.get('attempted_views',len(receipt['images'])),total=receipt.get('total_views',len(receipt['images']))))
    return result

def generate(request,progress=lambda value:None):
    with (HERE/'generation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('已有一个房间正在生成；请等待其完成')
        return _generate(request,progress)

if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    result=generate(request,lambda x:print(x,flush=True))
    print(json.dumps({k:result[k] for k in ('episode_id','layout_id','seed','scene')},ensure_ascii=False))
