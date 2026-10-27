"""Official stdio App Server client. No credential extraction or API-key fallback."""
from collections import deque
import json
import os
import selectors
import shutil
import subprocess
import tempfile
import time


DISABLED_FEATURES = (
    'apps', 'browser_use', 'browser_use_external', 'computer_use',
    'goals', 'hooks', 'image_generation', 'in_app_browser', 'in_app_chat',
    'in_app_local_automation', 'multi_agent', 'multi_agent_v2', 'plugins',
    'remote_plugin', 'shell_snapshot', 'shell_tool', 'skill_mcp_dependency_install',
    'skill_search', 'sleep_tool', 'tool_suggest', 'unified_exec', 'view_image',
    'workspace_dependencies', 'worktrees', 'memories', 'request_permissions_tool',
    'unbounded_connection_retries', 'realtime_conversation', 'code_mode',
)


def isolation_config(config=None):
    """Per-process/thread overrides only; never mutate the user's config file."""
    config = config or {}
    result = {'features': {name: False for name in DISABLED_FEATURES},
              'web_search': 'disabled', 'project_doc_max_bytes': 0,
              'developer_instructions': '', 'forced_login_method': 'chatgpt'}
    result['features']['skip_host_skill_discovery'] = True
    # The model's JS tool dispatcher needs this even with no host environment.
    # This is NOT the robot Python sandbox or a host shell permission.
    result['features']['code_mode_host'] = True
    result['features']['code_mode'] = {
        'enabled': True, 'direct_only_tool_namespaces': ['arena'],
        'excluded_tool_namespaces': ['collaboration'],
    }
    result['mcp_servers'] = {name: {'enabled': False}
                             for name in (config.get('mcp_servers') or {})}
    result['apps'] = {name: {'enabled': False} for name in (config.get('apps') or {})}
    result['apps']['_default'] = {'enabled': False}
    return result


class AppServer:
    def __init__(self, executable=None):
        executable = executable or shutil.which('codex')
        if not executable:
            raise RuntimeError('Install/login to the official Codex CLI first')
        self.scratch = tempfile.TemporaryDirectory(prefix='arena-codex-')
        self.log = tempfile.TemporaryFile()
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        argv = [executable, 'app-server', '--listen', 'stdio://']
        for key, value in isolation_config().items():
            if isinstance(value, dict):
                for child, val in value.items():
                    if not isinstance(val, dict):
                        argv += ['-c', f'{key}.{child}={json.dumps(val)}']
            else:
                argv += ['-c', f'{key}={json.dumps(value)}']
        self.process = subprocess.Popen(argv, cwd=self.scratch.name, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.events = deque()
        self.sequence = 0
        self.request('initialize', {'clientInfo': {'name': 'arena_codex_adapter',
            'version': '0.1.0'}, 'capabilities': {'experimentalApi': True}})
        self.send({'method': 'initialized', 'params': {}})

    def send(self, message):
        self.process.stdin.write((json.dumps(message, allow_nan=False)+'\n').encode())
        self.process.stdin.flush()

    def _read(self, timeout=300):
        deadline = time.monotonic()+timeout
        while b'\n' not in self.buffer:
            if len(self.buffer) > 64*1024*1024:
                raise RuntimeError('Oversized App Server message')
            if not self.selector.select(max(0, deadline-time.monotonic())):
                raise TimeoutError('App Server response timeout; do not replay robot RPC')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise EOFError('App Server exited')
            self.buffer.extend(chunk)
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return json.loads(line)

    def request(self, method, params, timeout=60):
        self.sequence += 1
        ident = self.sequence
        self.send({'id': ident, 'method': method, 'params': params})
        deadline = time.monotonic()+timeout
        while True:
            row = self._read(max(0, deadline-time.monotonic()))
            if row.get('id') == ident and 'method' not in row:
                if 'error' in row:
                    raise RuntimeError(f'{method}: {row["error"]}')
                return row['result']
            self.events.append(row)

    def next_event(self, timeout=300):
        return self.events.popleft() if self.events else self._read(timeout)

    def start_thread(self, model, instructions, tools):
        account = self.request('account/read', {'refreshToken': False})
        if (account.get('account') or {}).get('type') not in ('chatgpt', 'chatgptAuthTokens'):
            raise RuntimeError('ChatGPT login required; API-key accounts are not used')
        config = self.request('config/read', {'includeLayers': False})['config']
        started = self.request('thread/start', {
            'model': model, 'modelProvider': 'openai', 'allowProviderModelFallback': False,
            'developerInstructions': instructions, 'cwd': self.scratch.name,
            'ephemeral': True, 'environments': [], 'runtimeWorkspaceRoots': [],
            'experimentalRawEvents': True,
            'selectedCapabilityRoots': [], 'approvalPolicy': 'never',
            'sandbox': 'read-only', 'dynamicTools': [{'type':'namespace', 'name':'arena',
                'description':'Robot-only Arena tools. Call directly, sequentially; results include camera images.',
                'tools':tools}],
            'config': isolation_config(config),
        })
        if started['model'] != model:
            raise RuntimeError('Model substitution refused')
        return started

    def close(self):
        # Stops only this adapter-owned App Server, never the robot environment.
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.selector.close()
        self.process.stdout.close()
        self.log.close()
        self.scratch.cleanup()
