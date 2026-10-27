"""JSONL endpoint and original public TCP validation; never retries an RPC."""
from copy import deepcopy
import json,math,os,selectors,subprocess,time

SCHEMA='short_tasks.coding_robot22.v1'

class InputRejected(ValueError):
    pass

def tcp_public(value):
    if not isinstance(value,dict)or set(value)!={'left','right'}:raise ValueError('Missing robot FK')
    for pose in value.values():
        if pose is not None and(not isinstance(pose,list)or len(pose)!=4 or any(
            not isinstance(row,list)or len(row)!=4 or any(type(x)not in(float,int)or not math.isfinite(x)for x in row)for row in pose)):
            raise ValueError('Invalid robot FK')
    return deepcopy(value)

class Worker:
    """Trusted operator owns argv; model can only send validated robot tools."""
    def __init__(self,argv,stderr,timeout=360):
        # Match the original server transport: credentials belong to the local
        # App Server login, not to the remote robot/policy process environment.
        env={key:value for key,value in os.environ.items()
             if not any(marker in key.upper()for marker in ('KEY','TOKEN','SECRET','PASSWORD'))}
        self.process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env=env)
        self.timeout=timeout;self.buffer=bytearray();self.selection=selectors.DefaultSelector()
        self.selection.register(self.process.stdout,selectors.EVENT_READ)
    def read(self):
        deadline=time.monotonic()+self.timeout
        while b'\n'not in self.buffer:
            if len(self.buffer)>64*1024*1024:raise ValueError('Oversized public worker message')
            if not self.selection.select(max(0,deadline-time.monotonic())):raise TimeoutError('Worker unacknowledged; never replay')
            data=os.read(self.process.stdout.fileno(),65536)
            if not data:raise EOFError('Worker exited before acknowledgement')
            self.buffer.extend(data)
        line,_,rest=self.buffer.partition(b'\n');self.buffer=bytearray(rest)
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError('Worker message must be an object')
        return row
    def exchange(self,row):
        self.process.stdin.write((json.dumps(row,allow_nan=False)+'\n').encode());self.process.stdin.flush()
        return self.read()
    def close(self):
        self.process.stdin.close();code=self.process.wait(timeout=30)
        self.selection.close();self.process.stdout.close()
        return code
