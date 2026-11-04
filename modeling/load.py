"""Portable loader for the two trusted, optimizer-free release checkpoints."""
from pathlib import Path
import torch
from delta_layer import DeltaLayer
from training.model import ModelConfig, TCNLanguageModel
from training.streaming import CachedDecoder
from training.train import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]

def load_float(path=None, deployment_tables=False):
    path=Path(path) if path else ROOT/'models/float/qat-final.pt'
    if not path.is_file():raise FileNotFoundError('Download the external checkpoint from https://huggingface.co/heyanzhuo/convora to '+str(path)+'; see README.md.')
    saved=load_checkpoint(path)
    assert saved['kind'] in ('fp','qat') and not saved.get('diagnostic_only',False)
    backbone=TCNLanguageModel(ModelConfig.from_dict(saved['model_config'])).eval()
    backbone.load_state_dict(saved['backbone'],strict=True)
    layer=DeltaLayer(dimension=32,normalize='bounded').eval()
    layer.load_state_dict(saved['layer'],strict=True)
    if deployment_tables:
        # Only the final QAT package is paired with these int8 CPU memory tables.
        assert saved['source_checkpoint_sha256']=='9db7a2fa06b2687bfaa8091b9814bbf6ce52c7aba64d45bd66af933a902adef5'
        import numpy as np
        directory=ROOT/'models/rknn'
        table=np.fromfile(directory/'memory.i8','i1').reshape(-1,64).astype('float32')
        table*=np.fromfile(directory/'memory_scale.f32','<f4')[:,None]
        with torch.no_grad():
            offset=0
            for emb in backbone.engram.tables:
                count=len(emb.weight);emb.weight.copy_(torch.from_numpy(table[offset:offset+count]));offset+=count
    return backbone,layer,saved

class FloatDecoder:
    """FP32 recurrence; grids are present in the checkpoint but not applied here."""
    def __init__(self,backbone,layer):
        self.backbone=backbone;self.layer=layer;self.head=backbone.head
        self.backbone.head=torch.nn.Identity()
        self.cache=CachedDecoder(backbone);self.reset()

    def reset(self):
        self.cache.reset();self.states=[torch.zeros(1,32,1,32) for _ in range(4)]

    @torch.no_grad()
    def step(self,token):
        feature=self.cache.step(token).reshape(1,256,1,1)
        feature,*self.states=self.layer.reference(feature,self.states)
        return self.head(feature.reshape(1,256,1)).squeeze(2)
