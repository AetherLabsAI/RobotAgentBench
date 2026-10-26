"""Task-owned copy of the existing no-network sandbox transport."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from uuid import uuid4

LIMIT = 1024*1024

class Sandbox:
    def __init__(self, initial, workspace):
        from . import sandbox_launcher
        self.launcher = sandbox_launcher
        self.launcher.validate_workspace(workspace)
        self.name = 'arena-gpt-online-' + uuid4().hex
        self.cid = None
        self.proc = None
        self.buffer = bytearray()
        self.selector = selectors.DefaultSelector()
        try:
            image = json.loads(self.launcher.docker('image','inspect',self.launcher.IMAGE))[0]
            if image['Config'].get('Volumes'):
                raise RuntimeError('Sandbox image declares additional volumes')
            argv = self.launcher.create_arguments(self.name,
                ['python','-u','-c',(Path(__file__).parent/'sandbox_boot.py').read_text()], workspace)
            argv.insert(1, '-i')  # Framed stdin/out only, still no network/extra mounts/GPU.
            self.cid = self.launcher.docker(*argv).strip()
            if len(self.cid) != 64 or any(x not in '0123456789abcdef' for x in self.cid):
                raise RuntimeError('Invalid container ID')
            self.proc = subprocess.Popen(['docker','start','-ai',self.cid], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
            self.send(initial)
            self.ready = self.receive(60)
            if self.ready.get('op') != 'ready':
                raise RuntimeError('Sandbox initialization failed')
        except BaseException:
            self.close()
            raise

    def send(self, value):
        data = (json.dumps(value,allow_nan=False)+'\n').encode()
        if len(data) > LIMIT:
            raise ValueError('Oversized input')
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def receive(self, seconds):
        deadline = time.monotonic() + seconds
        while b'\n' not in self.buffer:
            if len(self.buffer) > LIMIT:
                raise RuntimeError('Sandbox output limit')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Sandbox compute budget')
            if not self.selector.select(remaining):
                raise TimeoutError('Sandbox compute budget')
            data = os.read(self.proc.stdout.fileno(),65536)
            if not data:
                raise EOFError('Sandbox exited')
            self.buffer.extend(data)
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        if len(line) > LIMIT:
            raise RuntimeError('Sandbox output limit')
        row = json.loads(line)
        if not isinstance(row,dict):
            raise ValueError('Expected object')
        return row

    def close(self):
        if self.cid:
            meta = json.loads(self.launcher.docker('inspect',self.cid))[0]
            if meta['Name'] != '/' + self.name or meta['Config']['Labels'].get(self.launcher.LABEL) != 'v1':
                raise RuntimeError('Refuse cleanup of unowned container')
            self.launcher.docker('rm','--force',self.cid)
            self.cid = None
        if self.proc:
            self.proc.wait(timeout=10)
            self.proc.stdin.close()
            self.proc.stdout.close()
        self.selector.close()
