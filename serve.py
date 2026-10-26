"""Local preview server with HTTP Range support (needed for video seeking).
Usage: python3 serve.py [port]"""
import os, re, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class H(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get('Range')
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)', rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        f = open(path, 'rb'); f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        self._remain = end - start + 1
        return f
    def copyfile(self, src, dst):
        n = getattr(self, '_remain', None)
        if n is None: return super().copyfile(src, dst)
        while n > 0:
            b = src.read(min(65536, n))
            if not b: break
            try: dst.write(b)
            except (BrokenPipeError, ConnectionResetError): break
            n -= len(b)
    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes'); super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
