"""Optional official LeRobot API caller for the explicit G1 training recipe.

No torch, model code, download, GPU probe or policy construction on import.
"""
import base64
import io

from .codec import REQUEST, actions_response, contract, state
from robotagentbench.interfaces.action_contract import public_observation


class Policy:
    def __init__(self, config):
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import InternVLAA15ChatProcessorTransformFn
        from lerobot.transforms.core import (NormalizeTransformFn, UnNormalizeTransformFn,
            PadStateAndActionTransformFn, RemapImageKeyTransformFn, ResizeImagesWithPadFn, compose)
        from .config import read
        self.np, self.torch = np, torch
        self.config = config
        cfg = PreTrainedConfig.from_pretrained(config['checkpoint'], local_files_only=True)
        if not isinstance(cfg, InternVLAA15Config):
            raise ValueError('Not an InternVLA-A1.5 checkpoint')
        cfg.vlm_model_name_or_path = config['vlm']
        cfg.inference_backend, cfg.action_loss_only = 'optimized', True
        cfg.device = config['device']
        self.dtype = {'float32':torch.float32, 'bfloat16':torch.bfloat16}[config['dtype']]
        stats = read(config['stats'])[config['stats_key']]
        normalized = {key:{k:np.asarray(v,dtype=np.float32) for k,v in row.items()}
                      for key,row in stats.items()}
        mapping = {'observation.images.'+name:'observation.images.image'+str(i)
                   for i,name in enumerate(('head','left','right'))}
        self.transform = compose([
            ResizeImagesWithPadFn(height=224,width=224,mapping=mapping),
            RemapImageKeyTransformFn(mapping=mapping),
            NormalizeTransformFn(selected_keys=['observation.state'],mode='mean_std',
                norm_stats={'observation.state':normalized['observation.state']}),
            InternVLAA15ChatProcessorTransformFn(pretrained_model_name_or_path=config['vlm'],
                mode='eval',tokenize_state=True,max_state_dim=32),
            PadStateAndActionTransformFn(max_state_dim=32,max_action_dim=32),
        ])
        self.unnormalize = UnNormalizeTransformFn(selected_keys=['action'],mode='mean_std',
            norm_stats={'action':normalized['action']})
        self.model = get_policy_class(cfg.type).from_pretrained(
            config['checkpoint'], config=cfg, local_files_only=True)
        self.model = self.model.to(device=config['device'],dtype=self.dtype).eval()
        self.model.reset()

    def infer(self, request):
        from PIL import Image
        if not isinstance(request,dict) or set(request) != {'schema','contract','observation'} or request['schema'] != REQUEST:
            raise ValueError('Exact example HTTP request required')
        contract(request['contract'])
        obs = public_observation(request['observation'])
        torch,np = self.torch,self.np
        sample = {'observation.state':torch.tensor(state(obs),dtype=torch.float32),
                  'action':torch.zeros(50,22), 'task':obs['instruction'],
                  'language_memory':obs.get('context','')}
        for camera in ('head','left','right'):
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs['images'][camera],validate=True))).convert('RGB')).copy()
            sample['observation.images.'+camera] = torch.from_numpy(rgb).permute(2,0,1).float()/255
        sample = self.transform(sample)
        batch = {key:(value.unsqueeze(0).to(device=self.config['device'],
                     dtype=self.dtype if value.dtype.is_floating_point else value.dtype)
                     if isinstance(value,torch.Tensor) else [value]) for key,value in sample.items()}
        # Same audited full-chunk API: no leaked cross-episode temporal queue.
        self.model.reset()
        with torch.inference_mode():
            predicted = self.model.predict_action_chunk(batch)
        if tuple(predicted.shape) != (1,50,32) or not bool(torch.isfinite(predicted).all()):
            raise ValueError('Expected finite batch1 × chunk50 × padded32 prediction')
        # The bound G1 schema, not the generic RoboTwin compacting rule, owns
        # the 22 supervised leading columns. Unnormalize exactly once.
        decoded = self.unnormalize({'action':predicted[0,:,:22].float()})['action']
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError('Nonfinite decoded native22 model action')
        return actions_response(obs,decoded.detach().cpu().tolist())
