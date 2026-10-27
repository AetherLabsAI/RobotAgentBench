"""Small optional local HTTP adapter, not an upstream InternVLA HTTP API."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import math
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

MAX_BODY = 40*1024*1024


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http','https') or not parsed.hostname
            or parsed.path != '/infer' or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError('Explicit http(s) endpoint ending in /infer required; no URL credentials')
    return parsed


class Client:
    def __init__(self,url,timeout_s):
        endpoint(url)
        if type(timeout_s)not in(int,float)or not math.isfinite(timeout_s)or not 0<timeout_s<=600:
            raise ValueError('Finite bounded HTTP timeout required')
        self.url,self.timeout_s = url,timeout_s

    def infer(self,payload):
        raw = json.dumps(payload,allow_nan=False).encode()
        if len(raw)>MAX_BODY:raise ValueError('Oversized model request')
        request = Request(self.url,data=raw,headers={'Content-Type':'application/json'},method='POST')
        # Deliberately no automatic retry: inference/robot outcome may be uncertain.
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        with build_opener(NoRedirect()).open(request,timeout=self.timeout_s) as reply:
            if reply.status != 200:raise RuntimeError('Model endpoint refused inference')
            data = reply.read(MAX_BODY+1)
        if len(data)>MAX_BODY:raise ValueError('Oversized model response')
        return json.loads(data,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite response')))


def serve(policy,host='127.0.0.1',port=8765):
    if host not in ('127.0.0.1','::1','localhost'):
        raise ValueError('Example service binds only to loopback; use a protected operator tunnel')
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path!='/infer':self.send_error(404);return
            try:
                if self.headers.get('Transfer-Encoding'):raise ValueError('Chunked requests unsupported')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=MAX_BODY:raise ValueError('Invalid request size')
                request=json.loads(self.rfile.read(length),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite request')))
                result=policy.infer(request)
                data=json.dumps(result,allow_nan=False).encode()
            except Exception:
                # Do not echo checkpoint paths, traceback, state or task metadata.
                logging.exception('InternVLA example inference rejected (operator diagnostics only)')
                data=b'{"error":"Inference rejected; see operator diagnostics"}'
                self.send_response(422)
            else:self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    return HTTPServer((host,port),Handler)
