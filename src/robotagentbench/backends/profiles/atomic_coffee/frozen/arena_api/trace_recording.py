"""Append-only client evidence; the compatible trace.json is published once.

Journal records are flushed before image/evaluator work, not fsynced per step.
This preserves acknowledged actions across process interruption without claiming
power-loss durability. Recovery is for auditing only, never simulator resumption.
"""
import json
from pathlib import Path


SCHEMA = "arena.trace.journal.v1"


class TraceRecorder:
    def __init__(self, output):
        self.output = Path(output)
        self.stream = (self.output / "trace.jsonl").open("x", encoding="utf-8")
        self.sequence = 0

    def _append(self, kind, data):
        record = dict(schema_version=SCHEMA, sequence=self.sequence, kind=kind, data=data)
        # Serialize only this record. A nonserializable value cannot leave half a
        # JSON record behind; an interrupted write can only affect the last line.
        line = json.dumps(record, allow_nan=False) + "\n"
        self.stream.write(line)
        self.stream.flush()
        self.sequence += 1

    def metadata(self, transcript, *, final=False):
        self._append("final" if final else "metadata",
                     {k: v for k, v in transcript.items() if k not in ("calls", "steps")})

    def batch(self, index, call):
        self._append("batch", dict(index=index, call=call))

    def step(self, record, call):
        self._append("step", dict(record=record, call_progress={
            k: call[k] for k in ("executed_steps", "discarded_steps")}))

    def finish(self, transcript):
        try:
            self.metadata(transcript, final=True)
        finally:
            self.stream.close()
        # One full serialization, after success/error cleanup. Atomic publication
        # keeps consumers from seeing a partially written compatible summary.
        destination = self.output / "trace.json"
        temporary = self.output / "trace.json.tmp"
        temporary.write_text(json.dumps(transcript, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(destination)

    def close(self):
        self.stream.close()


def load_journal(path):
    """Recover recorded evidence, ignoring only an unterminated final line.

    Completed journals reconstruct the original trace dictionary. An incomplete
    journal is explicitly invalid/undetermined, even if intermediate metadata
    contained a success flag. The run-level report remains authoritative.
    """
    transcript = {"calls": [], "steps": []}
    finalized = False
    with Path(path).open("rb") as stream:
        for sequence, line in enumerate(stream):
            if finalized:
                raise ValueError("Evidence after finalized journal")
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if event.get("schema_version") != SCHEMA or event.get("sequence") != sequence:
                raise ValueError("Invalid journal schema or sequence")
            kind, data = event["kind"], event["data"]
            if kind in ("metadata", "final"):
                if "calls" in data or "steps" in data:
                    raise ValueError("Journal metadata must not replace history")
                transcript.update(data)
                finalized = kind == "final"
            elif kind == "batch":
                if data["index"] != len(transcript["calls"]):
                    raise ValueError("Out-of-order journal batch")
                transcript["calls"].append(data["call"])
            elif kind == "step":
                record = data["record"]
                cycle = record["cycle"]
                if (type(cycle) is not int or not 0 <= cycle < len(transcript["calls"])
                        or record["observation"]["step_index"] != len(transcript["steps"]) + 1):
                    raise ValueError("Out-of-order journal step")
                transcript["steps"].append(record)
                transcript["calls"][cycle].update(data["call_progress"])
            else:
                raise ValueError("Unknown journal event")
    if not finalized:
        transcript.update(evaluation_valid=False, task_success=None,
                          closed_loop_transport_pass=False,
                          termination_reason="incomplete_journal",
                          error={"type": "IncompleteTrace", "message": "No final cleanup record"},
                          total_executed_steps=len(transcript["steps"]))
    return transcript
