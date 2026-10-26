import sys,tempfile,unittest
from pathlib import Path
from render_stream import stream

class StreamTests(unittest.TestCase):
    def test_real_events_and_log(self):
        events=[]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'render.log'
            stream([sys.executable,'-c','print(\'ROOM_PROGRESS {"stage":"render","completed":2,"total":5}\')'],p,events.append)
            self.assertEqual(events[0]['completed'],2);self.assertIn('ROOM_PROGRESS',p.read_text())
    def test_failure_does_not_become_completion(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):stream([sys.executable,'-c','raise RuntimeError("incomplete")'],Path(d)/'render.log',lambda x:None)
