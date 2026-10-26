"""Best-effort native teardown with explicit evidence, never score settlement."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path


@dataclass
class NativeResources:
    app: object = None
    simulation: object = None
    session: object = None
    host: object = None
    collision_guard: object = None
    annotators: dict = field(default_factory=dict)
    products: list = field(default_factory=list)
    source_hashes: dict | None = None
    lease: object = None
    closed: bool = False

    def shutdown(self, report, checkpoint):
        if self.closed:
            return
        self.closed = True
        errors = report.setdefault('cleanup_errors', [])

        def attempt(label, callback):
            try:
                callback()
                return True
            except BaseException as error:
                errors.append({'operation': label, 'error': str(error) or type(error).__name__})
                return False

        for label, resource, method in (
                ('host', self.host, 'close'), ('session', self.session, 'close'),
                ('physics', self.simulation, 'detach_stage'),
                ('collision_guard', self.collision_guard, 'close')):
            if resource is not None:
                attempt(label, lambda resource=resource, method=method: getattr(resource, method)())
        for pair in self.annotators.values():
            for annotator in pair:
                attempt('camera', lambda annotator=annotator: annotator.detach())
        for product in self.products:
            attempt('render_product', lambda product=product: product.destroy())
        if self.source_hashes is not None:
            report['source_layers_unchanged'] = True
            for path, expected in self.source_hashes.items():
                def verify(path=path, expected=expected):
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                        raise RuntimeError('Source layer changed')
                if not attempt('source_layers', verify):
                    report['source_layers_unchanged'] = False
        attempt('report_before_shutdown', checkpoint)
        report['native_shutdown_complete'] = self.app is not None and attempt('app', lambda: self.app.close())
        if report.get('status') == 'physics_completed':
            if (not errors and report['native_shutdown_complete']
                    and report.get('source_layers_unchanged') is True):
                report['status'] = 'completed'
            else:
                report['status'] = 'error'
        # The owner lock is released even if app.close() or report writing fails.
        if self.lease is not None:
            attempt('gpu_lock', lambda: self.lease.close())
        if errors and report.get('status') == 'completed':
            report['status'] = 'error'
        attempt('report_after_shutdown', checkpoint)
        if errors and report.get('status') == 'completed':
            report['status'] = 'error'


def shutdown_succeeded(report):
    return (report.get('status') == 'completed' and not report.get('cleanup_errors')
            and report.get('source_layers_unchanged') is True
            and report.get('native_shutdown_complete') is True)
