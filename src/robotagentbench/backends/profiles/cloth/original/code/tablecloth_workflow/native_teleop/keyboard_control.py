"""Keyboard joint jogging. Only drive targets are changed, never physical state."""
import time
import numpy as np
class KeyboardControl:
 def __init__(self,names,targets,limits,headless=False):
  self.names=list(names);self.targets=np.array(targets,float);self.limits=limits;self.side='l';self.joint=1;self.keys=set();self.running=headless;self.quit=False;self.events=[];self.headless=headless;self.start=time.monotonic();self.label=None
  if not headless:
   import carb.input,omni.appwindow,omni.ui as ui
   self.input=carb.input.acquire_input_interface();self.keyboard=omni.appwindow.get_default_app_window().get_keyboard();self.sub=self.input.subscribe_to_keyboard_events(self.keyboard,self.on_key)
   self.window=ui.Window('Task8 native PhysX teleop — DIAGNOSTIC',width=570,height=180)
   with self.window.frame:
    with ui.VStack():
     ui.Label('SPACE: run/pause | L/R: arm | 1..7: joint | W/S: +/-')
     ui.Label('O/P: open/close selected gripper | ESC: save and exit')
     ui.Label('Records all simulated frames. Material NOT calibrated.')
     self.label=ui.Label('Paused. Select arm/joint, then press SPACE.')
 def on_key(self,event):
  import carb.input
  k=event.input.name
  if event.type==carb.input.KeyboardEventType.KEY_RELEASE:self.keys.discard(k)
  if event.type==carb.input.KeyboardEventType.KEY_PRESS:
   self.keys.add(k);self.events.append(dict(wall_s=time.monotonic()-self.start,key=k))
   if k=='SPACE':self.running=not self.running;self.keys.clear()
   elif k in ('L','R'):self.side=k.lower()
   elif k=='ESCAPE':self.quit=True
   else:
    for i in range(1,8):
     if k in (f'KEY_{i}',str(i),f'NUMPAD_{i}'):self.joint=i
  return True
 def step(self,dt,actual):
  # Restrict command lead; force-driven joint response remains authoritative.
  name=next((n for n in self.names if n.endswith(f'arm_{self.side}_joint{self.joint}')),None)
  if name:
   i=self.names.index(name);delta=(('W' in self.keys)-('S' in self.keys))*.25*dt
   if delta:self.targets[i]=np.clip(self.targets[i]+delta,actual[i]-.12,actual[i]+.12)
  grip=next((n for n in self.names if n.endswith(f'gripper_{self.side}_outer_joint1')),None)
  if grip:
   i=self.names.index(grip);self.targets[i]+=(('O' in self.keys)-('P' in self.keys))*.35*dt
  for i,n in enumerate(self.names):
   if n in self.limits:self.targets[i]=np.clip(self.targets[i],*self.limits[n])
  if self.label:self.label.text=f'{"RUNNING" if self.running else "PAUSED"} | arm {self.side.upper()} joint {self.joint} | target {self.targets[self.names.index(name)]:.3f} rad'
  return self.targets.copy()
