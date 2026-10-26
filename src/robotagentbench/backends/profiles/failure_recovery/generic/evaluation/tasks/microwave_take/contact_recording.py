"""Optional raw contact logging, independent of live physical measurements.

Default: retain per-step aggregate evidence, never serialize raw contact rows.
Diagnostic --record-raw-contacts: preserve the historical lossless JSONL format.
Neither mode changes contact reads, current-step scoring, or collision physics.
"""
import json
import math
from pathlib import Path

SCHEMA = 'arena.microwave_take.contact_recording.v1'
SUMMARY_SCHEMA = 'arena.microwave_take.contact_summary.v1'


def summarize_contacts(rows):
    """Summarize already-validated reader values without recomputing forces.

Sleeping contact history is NOT counted as positive current-frame impulse.
None/missing flags are not allowed to masquerade as an empty contact stream.
"""
    if not isinstance(rows, (list, tuple)):
        raise ValueError('Missing measured contact rows')
    positive = sleeping = robot_plate = 0
    for row in rows:
        cached, loaded = row['cached_sleeping'], row['had_positive_impulse']
        if type(cached) is not bool or type(loaded) is not bool:
            raise ValueError('Unknown current/sleeping contact evidence')
        paths = row['colliders']
        positive += loaded and not cached
        sleeping += cached
        robot_plate += any(x.startswith('/World/Robot/') for x in paths) and any(
            x.startswith('/World/Assets/MicrowaveTakePlate/') for x in paths)
    return dict(schema=SUMMARY_SCHEMA, measurements_complete=True,
                point_count=len(rows), positive_impulse_point_count=positive,
                sleep_cached_point_count=sleeping, robot_plate_point_count=robot_plate)


class ContactRecorder:
    def __init__(self, output, *, record_raw_contacts=False):
        if type(record_raw_contacts) is not bool:
            raise ValueError('Raw contact recording must be explicit boolean')
        self.output = Path(output)
        raw = self.output/'process_contacts.jsonl'
        if raw.exists():
            raise FileExistsError(raw)
        self.spec = dict(schema=SCHEMA, mode='full' if record_raw_contacts else 'summary',
            raw_points_recorded=record_raw_contacts,
            raw_file=raw.name if record_raw_contacts else None,
            live_contact_reads='every_physics_step', live_scoring='every_physics_step')
        with (self.output/'process_contact_recording.json').open('x') as stream:
            json.dump(self.spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        self.raw_file = raw.open('x') if record_raw_contacts else None
        self.frames = self.points = 0

    def record(self, step, time_s, rows):
        if type(step) is not int or step != self.frames or not math.isfinite(time_s):
            raise ValueError('Missing/reordered contact recording frame')
        summary = summarize_contacts(rows)
        if self.raw_file is not None:
            self.raw_file.write(json.dumps(dict(reader_step=step,time_s=time_s,points=rows),
                                           allow_nan=False)+'\n')
            self.raw_file.flush()
        self.frames += 1
        self.points += len(rows)
        return summary

    def report(self):
        return dict(self.spec, measured_frames=self.frames, measured_points=self.points,
                    raw_points_saved=self.points if self.raw_file is not None else 0)

    def close(self):
        if self.raw_file is not None:
            self.raw_file.close()
