"""The source-pinned request used by the original native shelf-relocation task."""
import hashlib
import json

SOURCE_SHA256 = 'f41894566f3c5a00f8af76844d1a53b0963000ff3bb4cfe5968b451ae7f39db3'
SOURCE_LAUNCH_SHA256 = '423cfe2ca2653bb247eaa0bcd5f0a1cd12043b8d568d7b8330f63b50b968fe6e'
_SOURCE = '''{
  "schema": "openend.short_task.request.v1",
  "task_name": "stock_wooden_tray_interference",
  "rule_version": "openend.short_task.stock_wooden_tray_interference.v1",
  "seed": 8,
  "layout_id": 3,
  "targets": [
    {
      "category": "gatorade_t",
      "quantity": 1
    }
  ],
  "options": {
    "relocation_distance": "medium"
  }
}
'''


def request_for(seed):
    """Return the formal request, replacing only its original seed.

    Task, target, layout and distance selection remain those of the original
    task. This does not alter the source sampler, score, physics or thresholds.
    """
    if type(seed) is not int or not 1 <= seed < 2**32:
        raise ValueError('An original positive uint32 seed is required')
    if hashlib.sha256(_SOURCE.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('The original source-pinned request changed')
    value = json.loads(_SOURCE)
    value['seed'] = seed
    return value
