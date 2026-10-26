"""Task-local continuous action source, independent of human intervention.

Only drive targets cross this boundary. No recorded object poses, contacts,
grasp assistance, controller settings or source collision filters are applied.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


class RecordedApproach:
    def __init__(self, directory, adapter):
        directory = Path(directory)
        episode = json.loads((directory / 'episode.json').read_text())
        plan = json.loads((directory / 'PLAN.json').read_text())
        if episode['product']['name'] != 'gatorade_t' or plan['hand'] not in ('l','r'):
            raise ValueError('This audited approach is Gatorade with an explicit hand only')
        with np.load(directory / 'command_reference.npz', allow_pickle=False) as data:
            self.names = list(map(str, data['joint_names']))
            self.q = data['q'].copy()
            self.times = data['time_s'].copy()
        if self.q.shape != (len(self.times), len(self.names)) or not np.isfinite(self.q).all():
            raise ValueError('Invalid recorded commands')
        if not np.allclose(np.diff(self.times), 0.05, atol=1e-8):
            raise ValueError('Recorded commands must already be 20 Hz')
        phases = plan['phases']
        if [p['name'] for p in phases[:3]] != ['reach', 'insert', 'close']:
            raise ValueError('Unexpected source phase order')
        self.trigger_last_frame = int(phases[1]['last_frame'])
        self.last_frame = len(self.q) - 1
        self.adapter = adapter
        self.order = tuple(n for n in adapter.profile.order if not n.startswith('base_'))
        if any(n not in self.names for n in self.order):
            raise ValueError('Missing commanded joint')
        # Validate every target using the existing public joint adapter before
        # the first physical action; never clip or silently substitute limits.
        for i in range(self.last_frame + 1):
            self.native(i)
        self.evidence = dict(
            source_episode=episode['episode_id'], source_directory=str(directory),
            files_sha256={n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
                          for n in ('episode.json', 'PLAN.json', 'command_reference.npz')},
            source_hand=plan['hand'], source_base_xyz=episode['robot']['base_xyz_m'],
            source_base_wxyz=episode['robot']['base_wxyz'], source_target=episode['target']['xyz_m'],
            policy='continuous_recorded_commands_no_pause_or_retreat_on_proximity',
            source_fps=20, last_approach_frame=self.trigger_last_frame,
            last_recorded_frame=self.last_frame,
            replayed_object_poses=False, copied_grasp_assistance=False,
            copied_physics_parameters=False, full_pickup_claimed=False)

    def native(self, frame):
        if not 0 <= frame <= self.last_frame:
            raise ValueError('Recorded trajectory exhausted; do not silently hold the robot')
        values = dict(zip(self.names, map(float, self.q[frame])))
        return self.adapter.expand({n: values[n] for n in self.order})

    def command_at_step(self, physical_step):
        """One uninterrupted 20 Hz source clock on the existing 60 Hz demo.

        A trigger between control ticks must not reset the clock, duplicate a
        frame, or advance the next command early. The caller keeps this global
        step across both the pre-trigger and intervention loops.
        """
        if physical_step < 0:
            raise ValueError('Negative physical step')
        if physical_step % 3:
            return None
        frame = physical_step // 3
        return frame, self.native(frame)
