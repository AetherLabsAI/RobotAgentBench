import json,types
from pathlib import Path
import pytest
from real2sim.production import task8_environment as mod
class Layer:
 def __init__(self,path):self.realPath=str(path);self.text='robot+targets before physics';self.restores=0
 def ExportToString(self):return self.text
 def ImportFromString(self,text):self.text=text;self.restores+=1
 def Save(self):pass

def setup(tmp_path):
 layer=Layer(tmp_path/'scene.usda');stage=types.SimpleNamespace(GetRootLayer=lambda:layer)
 sidecar=tmp_path/'session.json';sidecar.write_text(json.dumps({'status':'open','task_ids':['8.2']}))
 session=types.SimpleNamespace(SIDECAR=sidecar,_atomic_sidecar=lambda d:sidecar.write_text(json.dumps(d)))
 return layer,stage,session

def test_invalid_geometry_rolls_back_before_next_owner_draw(tmp_path):
 layer,stage,session=setup(tmp_path);calls=[]
 def owner(**kw):
  calls.append(kw)
  assert layer.text=='robot+targets before physics'
  assert 'physical_distractor_seed' not in json.loads(session.SIDECAR.read_text())
  if len(calls)==1:
   layer.text='partial invalid clutter';session.SIDECAR.write_text(json.dumps({'physical_distractor_seed':4}))
   raise ValueError('sofa prop outside middle two seats: /World/Assets/Remote')
  return {'objects':['legal'],'seed':kw['seed']}
 result=mod._prepare_clutter_initialization(stage,{'key':'benchmark_8_2'},['target0','target1'],'cloth',{'seed':4},tmp_path,owner,session)
 assert calls[0]['seed']==4 and result['seed']==calls[1]['seed']!=4
 assert calls[1]['protected']==['target1','cloth'] and layer.restores==1
 evidence=json.loads((tmp_path/'clutter_initialization_sampling.json').read_text())
 assert [a['accepted'] for a in evidence['attempts']]==[False,True]
 assert evidence['physics_started'] is False

def test_non_geometry_failure_is_not_retried(tmp_path):
 layer,stage,session=setup(tmp_path);calls=[]
 def owner(**kw):calls.append(kw);raise ValueError('robot calibration mismatch')
 with pytest.raises(ValueError,match='calibration'):
  mod._prepare_clutter_initialization(stage,{'key':'benchmark_8_2'},['target'],'cloth',{'seed':4},tmp_path,owner,session)
 assert len(calls)==1 and layer.restores==0

def test_legal_first_draw_keeps_original_seed(tmp_path):
 layer,stage,session=setup(tmp_path);calls=[]
 def owner(**kw):calls.append(kw);return {'seed':kw['seed']}
 result=mod._prepare_clutter_initialization(stage,{'key':'benchmark_8_2'},['target'],'cloth',{'seed':17},tmp_path,owner,session)
 assert result=={'seed':17} and len(calls)==1 and layer.restores==0
