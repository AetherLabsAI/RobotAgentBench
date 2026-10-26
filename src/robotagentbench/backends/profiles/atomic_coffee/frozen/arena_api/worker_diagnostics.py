"""Opt-in local diagnostic recorder. No network module loading or task scoring."""
import json


def validate_diagnostic(identifier):
    if identifier is not None:
        from evaluation.diagnostics import DIAGNOSTIC_IDS
        if not isinstance(identifier, str) or identifier not in DIAGNOSTIC_IDS:
            raise ValueError("Unknown trusted diagnostic ID")
    return identifier


class DiagnosticRecorder:
    def __init__(self, identifier, stage, world, output):
        from evaluation.diagnostics import create_reader
        validate_diagnostic(identifier)
        self.identifier = identifier
        self.output = output
        self.closed = False
        self.count = 0
        self.last_time = None
        self.reader = create_reader(identifier, stage, output)
        self.file = None
        try:
            self.reader.initialize(world)
            self.file = (output / "diagnostic_samples.jsonl").open("x")
            self.sample("initialized")
        except BaseException:
            self.reader.close()
            if self.file is not None:
                self.file.close()
            raise

    def sample(self, phase):
        if self.closed:
            raise RuntimeError("Diagnostic is closed")
        row = self.reader.sample()
        now = row["time_s"]
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("Diagnostic time did not advance")
        record = dict(diagnostic_id=self.identifier, phase=phase, evidence=row)
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()
        self.count += 1
        self.last_time = now

    def close(self, completed):
        if self.closed:
            return
        self.closed = True
        try:
            self.reader.close()
        except BaseException:
            completed = False
            raise
        finally:
            self.file.close()
            report = dict(diagnostic_id=self.identifier, samples=self.count,
                          last_time_s=self.last_time, completed=bool(completed),
                          scoring_implemented=False, task_success=None)
            with (self.output / "diagnostic_recording.json").open("x") as stream:
                json.dump(report, stream, allow_nan=False, indent=2)
