"""Server-only CPU checks using the already deployed full-source environment."""
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]
PROJECT = SOURCE.parents[3]
VP = Path('/home/lingjun/robot_agent_arena')
G = VP/SOURCE.relative_to(PROJECT)
TASK = G/'short_tasks/tasks/microwave_heat_take'
PRIOR = Path('/workspace/lukun/arena_checkpoints_short14_20261018')
IMAGE = 'sha256:b272c84fbf8a4b7ec921a6fd680bc1bc1f3c1d5e153b653d394a9d6cc07912b6'


def command(mode):
    usd = '/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311'
    phys = '/isaac-sim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353'
    w = G/'short_tasks/agent_probe_v1/cap_v1/native_tools/openend_control_adapter_v1/task_discovery_v1/four_scenarios_min10_v1'
    cmd = ['docker', 'run', '--rm', '--read-only', '--network=none', '--user', '0:0',
           '--name', 'short_microwave_heat_take_cpu_'+mode, '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'CUDA_VISIBLE_DEVICES=', '-w', str(G),
           '-e', 'PYTHONPATH='+':'.join(map(str, [G, VP/'api-protocol', usd, phys,
                G/'openend/tasks/scene_requirements/cpu_dependencies',
                w/'generation_dependencies/ziche/tools/family3',
                w/'generation_dependencies/ziche/work/shelf_training_visible360_20260916'])),
           '-e', 'LD_LIBRARY_PATH='+usd+'/bin:'+phys+'/bin',
           '-e', 'PXR_AR_DEFAULT_SEARCH_PATH=/isaac-sim/kit/mdl/core/Base',
           '-e', 'PATH=/arena_node:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin']
    def mount(source, target, readonly=True):
        cmd.extend(['--mount', 'type=bind,src='+str(source)+',dst='+str(target)+(',readonly' if readonly else '')])
    mount(PROJECT, VP)
    mount(PROJECT, PROJECT)
    mount(PRIOR, PRIOR)
    deps = Path('/workspace/lukun/robot_agent_arena_dependencies/rootfs')
    mount(deps/'mnt', '/mnt')
    mount(deps, deps)
    mount(PRIOR/'followup_random_batches/batch2/tools/node-v22.23.3-linux-x64/bin', '/arena_node')
    mount(HERE, TASK, False)
    mount(HERE, HERE, False)
    for name, target in [('home', '/root'), ('tmp', '/tmp')]:
        folder = HERE/'cpu_cache'/name
        folder.mkdir(parents=True, exist_ok=True)
        mount(folder, target, False)
    return cmd+['--entrypoint', '/isaac-sim/python.sh', IMAGE, '-B', str(TASK/'check.py'), mode]


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode not in ('baseline', 'contracts', 'contracts_002', 'contracts_003', 'prepare_001', 'prepare_002', 'prepare_003', 'prepare_004'):
        raise SystemExit('Unknown check mode')
    out = HERE/'integration_v1/validation'
    out.mkdir(parents=True, exist_ok=True)
    cmd = command(mode)
    log = out/(mode+'.log')
    if log.exists():
        raise SystemExit('Existing check is preserved; choose a new check implementation/output explicitly')
    (out/(mode+'.command.json')).write_text(json.dumps(cmd, indent=2)+'\n')
    with log.open('x') as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps(dict(mode=mode, returncode=result.returncode, log=str(log))))
    raise SystemExit(result.returncode)
