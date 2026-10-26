"""Structured task admission errors; no scoring or fallback results."""


class TaskError(ValueError):
    def __init__(self, message, *, kind="input", reason="invalid_request"):
        super().__init__(message)
        self.kind, self.reason = kind, reason
