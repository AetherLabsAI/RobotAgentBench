"""Do not acknowledge episode closure before the native owner has shut down."""
import json


class CompletionStream:
    def __init__(self, output):
        self.output = output
        self.pending = None

    def write(self, line):
        # serve() writes one complete JSON line per call.
        if json.loads(line).get('op') == 'closed':
            if self.pending is not None:
                raise RuntimeError('Duplicate close acknowledgement')
            self.pending = line
        else:
            self.output.write(line)

    def flush(self):
        self.output.flush()

    def complete(self):
        if self.pending is not None:
            self.output.write(self.pending)
            self.pending = None
        self.output.flush()
