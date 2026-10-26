"""Success uses ordered physical milestones and the CURRENT post-close hold.

Process points remain historical; collisions are soft penalties, not a veto.
"""
REQUIRED = ('open', 'grasp', 'take_out', 'payload_retained', 'close')


def check_success(score):
    if score['status'] not in ('scoring', 'finalized') or score['error'] is not None:
        raise ValueError('No success verdict without valid physical evidence')
    stages = score['stages']
    if not all(stages.get(name) is not None for name in REQUIRED):
        return False
    order = [stages[name]['step_index'] for name in REQUIRED]
    # Payload may be confirmed on the extraction/closing endpoint. It must
    # not be first recovered AFTER the rewarded close.
    if not (order[0] < order[1] < order[2] <= order[3] <= order[4]):
        return False
    return (score['hold_requirement_status'] == 'satisfied'
            and score['hold_elapsed_s'] + 1e-9 >= 5.0)
