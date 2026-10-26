"""Task-local device admission for this user-authorized GPU0/1 test.

No process termination, utilization exception, physics or rendering changes.
The existing explicit shared-GPU2 mode is preserved, never selected silently.
"""
import subprocess
import xml.etree.ElementTree as ET


def admit(index, *, authorized_shared_gpu2=False, authorized_gpu1_residual_1mib=False):
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError('Only user-approved idle GPU0/1 or explicit legacy shared GPU2')
    if index == 2 and not authorized_shared_gpu2:
        raise ValueError('Shared GPU2 needs separate explicit authorization')
    raw = subprocess.check_output(['nvidia-smi', '--id', str(index),
        '--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    values = [value.strip() for value in raw.split(',')]
    if len(values) != 4 or not values[0].startswith('GPU-'):
        raise RuntimeError('Invalid GPU inventory')
    uuid = values[0]
    used, total, utilization = map(int, values[1:])
    xml = subprocess.check_output(['nvidia-smi', '--id', str(index), '-q', '-x'], text=True)
    processes = [node.findtext('pid') for node in ET.fromstring(xml).findall('.//process_info')]
    shared = index == 2
    permitted_residual = index == 1 and authorized_gpu1_residual_1mib is True and used == 1
    if not shared and ((used != 0 and not permitted_residual) or utilization != 0 or processes):
        raise RuntimeError(f'GPU{index} no longer idle; no process stopped')
    return dict(index=index, uuid=uuid, memory_used_mib=used, memory_total_mib=total,
                utilization_percent=utilization, process_ids=processes,
                shared=shared, idle_gpu1_1mib_exception=permitted_residual, no_process_stopped=True)
