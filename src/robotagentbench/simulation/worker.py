"""Dedicated-process entry for explicitly bound native task owners."""
from dataclasses import dataclass
import os
import sys

from robotagentbench.runtime.completion import CompletionStream
from .admission import admit_prepared_launch, verify_manifest
from .gpu import acquire_gpu
from .native_execution import execute_native
from .shutdown import shutdown_succeeded


@dataclass(frozen=True)
class NativeRunOptions:
    gpu: int
    smoke: bool = False
    coding_policy: bool = False
    collision_guard_audit: bool = False
    allow_idle_utilization: bool = False
    allow_shared_gpu: bool = False

    def __post_init__(self):
        if type(self.gpu) is not int or self.gpu < 0:
            raise ValueError('An explicit nonnegative GPU index is required')
        for name in ('smoke', 'coding_policy', 'collision_guard_audit',
                     'allow_idle_utilization', 'allow_shared_gpu'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('Native run options must be explicit booleans')
        if self.allow_shared_gpu and self.allow_idle_utilization:
            raise ValueError('Shared GPU and idle-utilization modes are mutually exclusive')
        if self.smoke and self.coding_policy:
            raise ValueError('Use the coding smoke client for coding-policy verification')


def run_native_worker(launch_path, output, *, options, backend, workspace, public_input_binding=None):
    """Admit, execute, shut down, and exit this dedicated process.

    This is not a library call for an interactive Python session: native stdout
    is redirected to stderr, CUDA/cache environment is process-local, and the
    process exits after flushing the completion pipe. No module or Python path
    can be selected through the launch JSON or the action RPC.
    """
    if not isinstance(options, NativeRunOptions):
        raise TypeError('Explicit native run options are required')
    if options.coding_policy and backend.wrap_coding_host is None:
        raise ValueError('Original code-tool sandbox and broker are not bound')
    extra = {} if public_input_binding is None else {'public_input_binding': public_input_binding}
    admitted = admit_prepared_launch(launch_path, output, backend=backend, workspace=workspace, **extra)
    if admitted.owner_kind in ('failure_recovery', 'atomic_coffee'):
        # The original fresh-interpreter handoff must not fall through to the
        # generic CPU owner or substitute its physics/scorer imports.
        backend.exec_special_owner(admitted, options=options, workspace=workspace)
        raise RuntimeError('Task-specific interpreter handoff returned unexpectedly')
    admitted.verify_inputs()
    backend.verify_scene_sources()
    verify_manifest(admitted.manifest)
    if admitted.launch.get('source_hashes'):
        verify_manifest({'source_hashes': admitted.launch['source_hashes']})
    if admitted.owner_kind == 'standalone':
        backend.registry.verify_sources(admitted.plan)
    lease = acquire_gpu(options.gpu, workspace=workspace,
                        allow_idle_utilization=options.allow_idle_utilization,
                        allow_shared_gpu=options.allow_shared_gpu)
    rpc = None
    try:
        admitted.verify_inputs()
        verify_manifest(admitted.manifest)
        if admitted.launch.get('source_hashes'):
            verify_manifest({'source_hashes': admitted.launch['source_hashes']})
        workspace.create_output(admitted.output)
        lease.configure_environment(admitted.output)
        descriptor = os.dup(sys.stdout.fileno())
        try:
            rpc = os.fdopen(descriptor, 'w', buffering=1)
        except BaseException:
            os.close(descriptor)
            raise
        completion = CompletionStream(rpc)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        report = execute_native(admitted, options=options, backend=backend,
                                lease=lease, completion=completion)
        code = 0 if shutdown_succeeded(report) else 1
        if code == 0:
            completion.complete()
        rpc.flush()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        lease.close()
        if rpc is not None:
            rpc.close()
    # Preserve the native owner's dedicated-process exit after app.close().
    os._exit(code)
