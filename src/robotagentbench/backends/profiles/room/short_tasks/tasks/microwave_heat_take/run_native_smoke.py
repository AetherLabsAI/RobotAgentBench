"""One unchanged native-owner hold smoke; no model calls or task policy."""
from pathlib import Path
import json
import subprocess
import sys

from run_cpu import HERE, SOURCE, PROJECT, VP, G, TASK, IMAGE


def main():
    attempt = sys.argv[1] if len(sys.argv) == 2 else '001'
    if attempt not in ('001', '002'):
        raise ValueError('Unknown native smoke attempt')
    out = HERE/'native_validation'/('native_smoke_'+attempt)
    if out.exists():
        raise ValueError('Existing native check must be retained')
    row = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    uuid = row[0].strip()
    assert int(row[1]) == 0 and int(row[2]) == 0, 'GPU not completely idle'
    pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-i', '0', '-c', '1'], text=True)
    rows = [r.split() for r in pmon.splitlines() if r.strip() and not r.lstrip().startswith('#')]
    assert rows and all(r[1] == '-' for r in rows), 'GPU has a compute or graphics process'
    lock = SOURCE/'short_tasks/agent_probe_v1'/('gpu-'+uuid+'.lock')
    assert lock.is_file() and not lock.is_symlink(), 'Existing native GPU lease required'
    out.mkdir(parents=True, exist_ok=False)
    cmd = ['docker', 'run', '--rm', '--read-only', '--network=none', '--user', '0:0',
           '--name', 'short_microwave_heat_take_native_smoke_'+attempt, '--gpus', 'device='+uuid,
           '-e', 'ACCEPT_EULA=Y', '-e', 'PRIVACY_CONSENT=Y',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'NVIDIA_DRIVER_CAPABILITIES=all',
           '-e', 'PYTHONPATH='+str(G)+':'+str(VP/'api-protocol'), '-w', str(G)]
    def mount(source, target, readonly=True):
        cmd.extend(['--mount', 'type=bind,src='+str(source)+',dst='+str(target)+(',readonly' if readonly else '')])
    mount(PROJECT, VP)
    mount(PROJECT, PROJECT)
    deps = Path('/workspace/lukun/robot_agent_arena_dependencies/rootfs')
    mount(deps/'mnt', '/mnt')
    mount(deps, deps)
    mount(HERE, TASK, False)
    mount(HERE, HERE, False)
    mount(lock, G/'short_tasks/agent_probe_v1'/lock.name, False)
    for name, target in [('home', '/root'), ('tmp', '/tmp'),
                         ('kit_cache', '/isaac-sim/kit/cache'), ('kit_data', '/isaac-sim/kit/data')]:
        folder = out/'cache'/name
        folder.mkdir(parents=True, exist_ok=True)
        mount(folder, target, False)
    cmd.extend(['--entrypoint', '/isaac-sim/python.sh', IMAGE, '-B', '-u', '-m',
                'short_tasks.agent_probe_v1.native_owner', '--launch',
                str(TASK/'execution/cpu_001/native/launch.json'), '--output',
                str(TASK/'native_validation'/('native_smoke_'+attempt)/'native'), '--gpu', '0', '--smoke'])
    (out/'admission.json').write_text(json.dumps(dict(gpu_uuid=uuid, memory_mib=0,
        utilization_percent=0, pmon=pmon, original_gpu_lease=str(lock), model_calls=0), indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    with (out/'native.log').open('x') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    (out/'process.json').write_text(json.dumps(dict(returncode=result.returncode,
        model_calls=0, expected_execution='unchanged internal hold smoke'))+'\n')
    print(json.dumps(dict(returncode=result.returncode, output=str(out))))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
