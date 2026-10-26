"""Append-only raw contact and merge evidence; container report only, never in obs.

Attach as an ADDITIONAL subscriber:
    sim.subscribe_contact_report_events(log.on_report)
probe_native.contact() (the 1 mm rule) keeps its own subscription unchanged;
this logger only reads the same headers/points and never filters or edits them.

File: <out>/contact_events.jsonl, one JSON object per line:
  {"k":"run","schema":...,"run_id":r,"limits":{...},"provenance":p}  first line
  {"k":"path","id":n,"path":"/World/..."}                    interned once
  {"k":"step","s":step,"h":[[type,a0,a1,c0,c1,npts,min_sep,impulse_sum,pts|null],...],
   "floor":m}                                                one row per substep with robot contacts
  {"k":"merge","seq":n,"run_id":r,"tick":t,"control_frame":k,"source":...,"rule":...,
   "read_call":c,"headers":[raw header + points],"detail":{...}}
                                                             contact merges, appended (OPENEND-09)
  {"k":"reader_error",...}                                   raw headers of a failed contact read
  {"k":"cap","bytes":b}                                      step-detail byte cap reached
  {"k":"incomplete","what":...,"reason":...}                 record capacity reached (explicit)
  {"k":"settlement",...}                                     pointer to the once-only settlement
  {"k":"summary",...}                                        on close
Growth control:
- only headers with a robot actor are kept (the 1 mm rule's population);
- floor/ground headers are counted per substep, not listed (probe exempts them);
- per-point detail only when min separation < POINT_DETAIL_BELOW_M (half the
  1 mm rule) so near-threshold evidence is complete but resting contact stays compact;
- hard byte cap: past it only per-pair counters are kept and written at close,
  with an explicit count of rows that lost detail. Nothing is dropped silently.
- merge / reader-error records have their own capacity (count and bytes). When
  it is reached ONE "incomplete" row is written, later records are counted only,
  and the summary says evidence_complete=false. Old rows are never rewritten.
Estimate: ~10 robot headers/substep x ~70 B -> ~170 KB per simulated second at
240 Hz (~45 KB at 60 Hz); 20 s closed loop ~3.5 MB; default cap 256 MB.

Merge sources (OPENEND-09) call record_merge() on the ACTIVE log (install()).
The registry lives in sys.modules so the same log is found whether this file
was imported as `contact_log` (closed-loop hook) or by its package path.
"""
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

SCHEMA = 'openend09.contact_log.v2'
POINT_DETAIL_BELOW_M = -0.0005
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_RECORDS = 4096                 # merge + reader_error rows
DEFAULT_MAX_RECORD_BYTES = 32 * 1024 * 1024
ROBOT_PREFIX = '/World/Robot/'
_REGISTRY = sys.modules.setdefault('openend_contact_audit_registry', types.ModuleType('openend_contact_audit_registry'))
if not hasattr(_REGISTRY, 'log'):
    _REGISTRY.log = None


def install(log):
    if active() not in (None, log):
        raise RuntimeError('another contact log is active')
    _REGISTRY.log = log


def uninstall(log):
    if _REGISTRY.log is log:
        _REGISTRY.log = None


def active():
    log = _REGISTRY.log
    return None if log is None or log.closed else log


def _floor(path):
    p = path.lower()
    return 'floor' in p or 'ground' in p


def _finite(obj):
    """JSON-safe copy: non-finite floats become strings so they stay visible."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def raw_header(h, points, *, paths=None, index=None):
    """Every header field the readers use, plus all its points, unmodified."""
    row = {k: int(getattr(h, k)) for k in ('stage_id', 'actor0', 'actor1', 'collider0', 'collider1',
                                           'contact_data_offset', 'num_contact_data') if hasattr(h, k)}
    row['type'] = str(h.type)
    if index is not None:
        row['header_index'] = int(index)
    if paths is not None:
        row['paths'] = list(paths)
    start, count = int(h.contact_data_offset), int(h.num_contact_data)
    if 0 <= start and 0 <= count and start + count <= len(points):
        row['points'] = []
        for p in points[start:start + count]:
            d = {k: [float(x) for x in getattr(p, k)] for k in ('position', 'normal', 'impulse')}
            sep = getattr(p, 'separation', None)
            d['separation'] = None if sep is None else float(sep)
            row['points'].append(d)
    else:
        row['points'] = None
        row['points_out_of_range'] = True
    return row


class ContactEventLog:
    def __init__(self, out_dir, *, path_of, step, run_id=None, control_frame=None, provenance='closed_loop_hook',
                 max_bytes=DEFAULT_MAX_BYTES, max_records=DEFAULT_MAX_RECORDS,
                 max_record_bytes=DEFAULT_MAX_RECORD_BYTES, point_detail_below_m=POINT_DETAIL_BELOW_M):
        self.file = Path(out_dir) / 'contact_events.jsonl'
        self.fh = self.file.open('x')
        self.path_of, self.step, self.control_frame = path_of, step, control_frame
        self.run_id = run_id
        self.max_bytes, self.detail_below = max_bytes, point_detail_below_m
        self.max_records, self.max_record_bytes = max_records, max_record_bytes
        self.ids = {}
        self.bytes = 0
        self.capped = False
        self.detail_lost_rows = 0
        self.pair_counts = Counter()
        self.min_sep = Counter()
        self.rows = 0
        self.records = Counter()          # written merge / reader_error rows
        self.record_bytes = 0
        self.records_dropped = Counter()  # counted, explicitly, after capacity
        self.records_full = False
        self.write_errors = 0
        self.closed = False
        self._write(dict(k='run', schema=SCHEMA, run_id=run_id, provenance=provenance,
                         limits=dict(max_bytes=max_bytes, max_records=max_records,
                                     max_record_bytes=max_record_bytes, point_detail_below_m=point_detail_below_m)),
                    essential=True, flush=True)

    # --- low level -------------------------------------------------------
    def _line(self, obj):
        try:
            return json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n'
        except ValueError:
            return json.dumps(_finite(obj), separators=(',', ':'), allow_nan=False) + '\n'

    def _emit(self, line, flush):
        try:
            self.fh.write(line)
            if flush:
                self.fh.flush()
        except OSError:
            self.write_errors += 1
            return False
        self.bytes += len(line)
        return True

    def _write(self, obj, *, essential=False, flush=False):
        line = self._line(obj)
        if not essential and self.bytes + len(line) > self.max_bytes:
            if not self.capped:
                self.capped = True
                self._emit(self._line(dict(k='cap', bytes=self.bytes, what='step_detail', tick=self._tick(),
                                           control_frame=self._frame())), True)
            return False
        return self._emit(line, flush)

    def _tick(self):
        try:
            return None if self.step is None else int(self.step())
        except Exception as exc:              # provenance must not hide the record itself
            return 'unavailable:' + type(exc).__name__

    def _frame(self):
        try:
            return None if self.control_frame is None else int(self.control_frame())
        except Exception as exc:
            return 'unavailable:' + type(exc).__name__

    def _id(self, path):
        if path not in self.ids:
            self.ids[path] = len(self.ids)
            self._write(dict(k='path', id=self.ids[path], path=path), essential=True)
        return self.ids[path]

    # --- per-substep compact rows -----------------------------------------
    def on_report(self, headers, points):
        rows, floor = [], 0
        for h in headers:
            actors = [self.path_of(h.actor0), self.path_of(h.actor1)]
            if not any(a.startswith(ROBOT_PREFIX) for a in actors):
                continue
            colliders = [self.path_of(h.collider0), self.path_of(h.collider1)]
            external = [c for c in colliders if not c.startswith(ROBOT_PREFIX)]
            if external and all(_floor(c) for c in external):
                floor += 1
                continue
            n, start = int(h.num_contact_data), int(h.contact_data_offset)
            pts = points[start:start + n] if n else []
            sep = min((float(p.separation) for p in pts), default=None)
            imp = sum(math.sqrt(sum(float(x) ** 2 for x in p.impulse)) for p in pts) if pts else 0.
            detail = None
            if sep is not None and sep < self.detail_below:
                detail = [[list(map(float, p.position)), list(map(float, p.normal)), float(p.separation),
                           list(map(float, p.impulse))] for p in pts]
            ids = [self._id(x) for x in actors + colliders]
            key = tuple(ids[2:])
            self.pair_counts[key] += 1
            if sep is not None:
                self.min_sep[key] = min(self.min_sep.get(key, sep), sep)
            rows.append([str(h.type), *ids, n, sep, imp, detail])
        if rows or floor:
            self.rows += 1
            if self.capped or not self._write(dict(k='step', s=int(self.step()), h=rows, floor=floor)):
                self.detail_lost_rows += 1

    # --- merge / error records (OPENEND-09) --------------------------------
    def _record(self, kind, body):
        if self.closed:
            raise RuntimeError('contact log already closed')
        seq = sum(self.records.values()) + sum(self.records_dropped.values()) + 1
        row = dict(k=kind, seq=seq, run_id=self.run_id, tick=self._tick(), control_frame=self._frame(), **body)
        line = self._line(row)
        if (sum(self.records.values()) >= self.max_records
                or self.record_bytes + len(line) > self.max_record_bytes):
            if not self.records_full:
                self.records_full = True
                self._emit(self._line(dict(k='incomplete', what='merge_and_reader_error_records',
                                           reason='record_capacity', first_dropped_seq=seq, run_id=self.run_id,
                                           tick=row['tick'], control_frame=row['control_frame'],
                                           written=dict(self.records), max_records=self.max_records,
                                           max_record_bytes=self.max_record_bytes)), True)
            self.records_dropped[kind] += 1
            return None
        if not self._emit(line, True):
            self.records_dropped[kind] += 1
            return None
        self.records[kind] += 1
        self.record_bytes += len(line)
        return seq

    def record_merge(self, *, source, rule, headers, read_call=None, detail=None):
        """Append one contact-merge event with its raw headers and points. Never rewrites."""
        return self._record('merge', dict(source=source, rule=rule, read_call=read_call,
                                          headers=headers, detail=detail))

    def record_reader_error(self, *, source, error, headers, read_call=None):
        return self._record('reader_error', dict(source=source, error=str(error)[:500], read_call=read_call,
                                                 headers=headers))

    def record_settlement(self, record_sha256, status):
        self._write(dict(k='settlement', run_id=self.run_id, status=status, sha256=record_sha256),
                    essential=True, flush=True)

    # --- status ------------------------------------------------------------
    def status(self):
        reasons = []
        if self.capped:
            reasons.append('step_detail_byte_cap')
        if self.records_dropped:
            reasons.append('record_capacity')
        if self.write_errors:
            reasons.append('write_errors')
        return dict(file=self.file.name, evidence_complete=not reasons, incomplete_reasons=reasons,
                    step_rows=self.rows, detail_lost_rows=self.detail_lost_rows, records=dict(self.records),
                    records_dropped=dict(self.records_dropped), write_errors=self.write_errors)

    def close(self):
        if self.closed:
            return
        pairs = [dict(colliders=list(k), headers=v, min_separation_m=self.min_sep.get(k))
                 for k, v in self.pair_counts.most_common()]
        self._write(dict(k='summary', bytes=self.bytes, capped=self.capped, pairs=pairs, **self.status()),
                    essential=True)
        self.closed = True
        uninstall(self)
        self.fh.close()
