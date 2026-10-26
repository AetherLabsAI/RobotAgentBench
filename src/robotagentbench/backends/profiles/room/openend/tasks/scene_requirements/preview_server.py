"""Loopback-only room preview, single generator, allowlisted downloads."""
import json
import threading
import uuid
import time
import os
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse
from generate_room import HERE,generate
from task_choices import CHOICES

JOBS={};LOCK=threading.Lock()
ORIGINS={'http://127.0.0.1:18774','http://localhost:18774'}
PUBLIC_ORIGIN=os.environ.get('OPENEND_PUBLIC_ORIGIN')
if PUBLIC_ORIGIN:
    if not PUBLIC_ORIGIN.startswith('https://'):raise ValueError('Public origin requires HTTPS')
    ORIGINS.add(PUBLIC_ORIGIN)
class Handler(BaseHTTPRequestHandler):
    def send(self,value,status=200,kind='application/json; charset=utf-8'):
        body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/':return self.send((HERE/'web/room.html').read_bytes(),kind='text/html; charset=utf-8')
        if path in ('/app.js','/cloth-options.js','/style.css'):
            return self.send((HERE/'web'/path[1:]).read_bytes(),kind='text/javascript' if path.endswith('.js') else 'text/css')
        if path=='/api/catalog':return self.send(CHOICES)
        if path.startswith('/images/'):
            parts=path.split('/')
            if len(parts)==4:parts.insert(3,'renders_v2')
            if (len(parts)==5 and parts[2].startswith('room_') and all(c in 'room_0123456789abcdef' for c in parts[2])
                    and parts[3] in ('renders_v2','renders_v3','renders_v4')):
                folder=HERE/'generated_rooms'/parts[2]/parts[3]
                manifest=folder/'manifest.json'
                allowed={i['file'] for i in json.loads(manifest.read_text())['images']} if manifest.is_file() else set()
                if parts[4] in allowed and '/' not in parts[4] and parts[4].endswith('.png'):
                    p=folder/parts[4]
                    if p.is_file():return self.send(p.read_bytes(),kind='image/png')
            return self.send({'error':'Image not available'},404)
        if path=='/api/current':
            p=HERE/'current_room.json'
            return self.send(json.loads(p.read_text()) if p.exists() else None)
        if path=='/api/latest-job':
            return self.send(next(reversed(JOBS.values()),None) if JOBS else None)
        if path=='/api/layouts':
            data=json.loads((HERE/'placement_sources/random-layouts-v1.json').read_text())
            return self.send([x['id'] for x in data['layouts']])
        if path.startswith('/api/jobs/'):
            return self.send(JOBS.get(path.split('/')[-1],{'status':'unknown'}))
        if path.startswith('/download/'):
            parts=path.split('/')
            if len(parts)==4 and parts[2].startswith('room_') and all(c in 'room_0123456789abcdef' for c in parts[2]) and parts[3] in ('composition.json','room.usda','runtime_request.json'):
                p=HERE/'generated_rooms'/parts[2]/parts[3]
                if p.is_file():return self.send(p.read_bytes(),kind='application/octet-stream')
        self.send({'error':'Not found'},404)
    def do_POST(self):
        if self.path!='/api/generate':return self.send({'error':'Not found'},404)
        if self.headers.get('Origin') not in ({None}|ORIGINS):
            return self.send({'error':'Origin rejected'},403)
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send({'error':'JSON required'},415)
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=32000:raise ValueError('Invalid request size')
            request=json.loads(self.rfile.read(size))
            from room_request import resolve
            resolve(request)
        except Exception as e:return self.send({'error':str(e)},400)
        if not LOCK.acquire(blocking=False):return self.send({'error':'已有一个房间正在生成'},409)
        ident=uuid.uuid4().hex;JOBS[ident]={'job_id':ident,'status':'running','stage':'plan','message':'正在准备','started_at':time.time()}
        def report(value):
            if isinstance(value,dict):
                if value.get('stage') in ('plan','compose','gpu') and value.get('stage')!=JOBS[ident].get('stage'):
                    JOBS[ident].pop('completed',None);JOBS[ident].pop('total',None)
                JOBS[ident].update(value)
            else:JOBS[ident].update(message=value)
        def worker():
            try:
                result=generate(request,report)
                JOBS[ident].update(status='done',stage='done',result=result,ended_at=time.time())
            except Exception as e:JOBS[ident].update(status='failed',error=str(e),ended_at=time.time())
            finally:LOCK.release()
        threading.Thread(target=worker,daemon=True).start()
        self.send({'job_id':ident},202)

if __name__=='__main__':
    print('Room overview http://127.0.0.1:18774/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',18774),Handler).serve_forever()
