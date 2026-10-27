"""Portable extraction of the actually evaluated G1 serve_intern.py recipe.

Source a2e030601ee9531de6780a3a5220eb482ff523507acb440246590f56b9a5b3ea.
The original is retained under private/server_source, not installed as code.
"""
import base64
import io
import json
from pathlib import Path
import time

from .codec import contract
from .config import read, sha, validate_control_schema


def model_config(value):
    """Local path parameters and original codec checks; no manual SHA entry."""
    if set(value) != {'python','checkpoint','stats','stats_key','control_schema','vlm'}:
        raise ValueError('Explicit tested G1 model path configuration required')
    for name in ('python','checkpoint','stats','control_schema','vlm'):
        if not isinstance(value[name],str) or not Path(value[name]).is_absolute():
            raise ValueError('Absolute operator model path required: '+name)
    c=Path(value['checkpoint']);cfg=read(c/'config.json')
    validate_control_schema(read(value['control_schema']))
    if (cfg.get('type')!='internvla_a1_5' or cfg.get('chunk_size')!=50
            or cfg.get('max_state_dim')!=32 or cfg.get('max_action_dim')!=32):
        raise ValueError('The tested G1 checkpoint uses chunk50 and padded32')
    stats=read(value['stats'])[value['stats_key']]
    for key,size in (('observation.state',23),('action',22)):
        import math
        for field in ('mean','std'):
            vals=stats[key][field]
            if len(vals)!=size or any(type(x)not in(int,float)or not math.isfinite(x)for x in vals):
                raise ValueError('Expected original G1 state23/action22 finite statistics')
        if any(x<0 for x in stats[key]['std']):raise ValueError('Negative normalization scale')
    if not Path(value['vlm']).is_dir()or not Path(value['python']).is_file():
        raise ValueError('Existing separate model environment and local VLM required')
    pins={str(p):sha(p)for p in [c/'config.json',c/'model.safetensors',Path(value['stats']),Path(value['control_schema'])]}
    return value,pins


class TestedPolicy:
    """Same transforms, BF16 model, seed42, NumPy unnormalization and prompt lookup."""
    def __init__(self,config,prompts):
        config,self.pins=model_config(config)
        self.prompts=dict(prompts)
        if not self.prompts or any(not isinstance(k,str)or not isinstance(v,str)for k,v in self.prompts.items()):
            raise ValueError('Explicit public task-ID/instruction allowlist required')
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import InternVLAA15ChatProcessorTransformFn
        from lerobot.transforms.core import (NormalizeTransformFn,PadStateAndActionTransformFn,
            RemapImageKeyTransformFn,ResizeImagesWithPadFn,compose)
        self.np,self.torch=np,torch
        stats=read(config['stats'])[config['stats_key']]
        self.stat={k:{a:np.asarray(v,dtype=np.float32)for a,v in values.items()}for k,values in stats.items()}
        schema=read(config['control_schema']);self.order=[x['name']for x in schema['action_channels']]
        mapping={f'observation.images.{k}':f'observation.images.image{i}'for i,k in enumerate(('head','left','right'))}
        cfg=PreTrainedConfig.from_pretrained(config['checkpoint'])
        if not isinstance(cfg,InternVLAA15Config)or cfg.chunk_size!=50:raise ValueError('Original G1 A1.5 config required')
        cfg.vlm_model_name_or_path=config['vlm'];cfg.inference_backend='optimized';cfg.action_loss_only=True;cfg.device='cuda'
        self.transform=compose([ResizeImagesWithPadFn(height=224,width=224,mapping=mapping),
            RemapImageKeyTransformFn(mapping=mapping),
            NormalizeTransformFn(selected_keys=['observation.state'],norm_stats={'observation.state':self.stat['observation.state']}),
            InternVLAA15ChatProcessorTransformFn(pretrained_model_name_or_path=config['vlm'],mode='eval',tokenize_state=cfg.tokenize_state,max_state_dim=32),
            PadStateAndActionTransformFn(max_state_dim=32,max_action_dim=32)])
        started=time.monotonic();torch.manual_seed(42)
        self.policy=get_policy_class(cfg.type).from_pretrained(config['checkpoint'],config=cfg)
        self.policy=self.policy.to(device='cuda',dtype=torch.bfloat16).eval();self.policy.reset()
        self.loading_seconds=time.monotonic()-started

    def infer(self,req):
        from PIL import Image
        if req.get('op')!='infer'or req['action_order']!=self.order:raise ValueError('Original action order required')
        obs=req['observation'];prompt=self.prompts[req['training_task_index']]
        if prompt!=obs['instruction']:raise ValueError('Public prompt binding differs from inference observation')
        np,torch=self.np,self.torch
        state=np.asarray([obs['joints'][k]for k in self.order[:20]]+obs['base_state']['position_world_m'][:2]+[obs['base_state']['yaw_world_unwrapped_rad']],dtype=np.float32)
        if state.shape!=(23,)or not np.isfinite(state).all():raise ValueError('Original finite state23 required')
        sample={'observation.state':torch.from_numpy(state),'action':torch.zeros(50,22),'task':prompt}
        for k,shape in [('head',(800,1280,3)),('left',(480,848,3)),('right',(480,848,3))]:
            rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(obs['images'][k]))).convert('RGB')).copy()
            if rgb.shape!=shape:raise ValueError('Tested recipe requires full native head1280x800/wrist848x480 images')
            sample['observation.images.'+k]=torch.from_numpy(rgb).permute(2,0,1).float()/255
        sample=self.transform(sample)
        batch={k:(v.unsqueeze(0).to(device='cuda',dtype=torch.bfloat16 if v.dtype.is_floating_point else v.dtype)if isinstance(v,torch.Tensor)else[v])for k,v in sample.items()}
        self.policy.reset();start=time.monotonic()
        with torch.inference_mode():normalized=self.policy.predict_action_chunk(batch).float().cpu().numpy()
        if normalized.shape!=(1,50,32)or not np.isfinite(normalized).all():raise ValueError('Original padded50x32 prediction required')
        actions=normalized[0,:,:22]*(self.stat['action']['std']+1e-6)+self.stat['action']['mean']
        if actions.shape!=(50,22)or not np.isfinite(actions).all():raise ValueError('Nonfinite original decoded action')
        return dict(op='actions',episode_id=obs['episode_id'],step=obs['step'],actions=actions.tolist(),
            inference_s=time.monotonic()-start,prompt=prompt,training_task_index=req['training_task_index'])


def serve_stdio(model,prompts):
    """Original infer/actions/close protocol, without the server-specific Docker launcher."""
    import sys
    wire=sys.stdout;sys.stdout=sys.stderr
    policy=TestedPolicy(model,prompts)
    def emit(value):wire.write(json.dumps(value,allow_nan=False)+'\n');wire.flush()
    emit(dict(op='ready',model='internvla_a15',action_shape=[50,22],loading_seconds=policy.loading_seconds,
        backend='native_optimized_action_only',decoding='G1_omnipicker_mean_std_eps1e-6_once',
        images='uniform_full_native_frame_head_left_right_then_native_resize224_pad',sources=policy.pins))
    for line in sys.stdin:
        request=json.loads(line)
        if request=={'op':'close'}:emit(dict(op='closed'));break
        emit(policy.infer(request))
