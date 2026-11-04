from qat_delta import SequenceDelta
from sft_model import SFTModel

class CalibratedDelta(SequenceDelta):
    def __init__(self,layer):
        super().__init__(layer);self.mode='qat'
        # Both state I/O grids are recalibrated from this trained model.
        self.grids['state_in'].fixed=False;self.grids['state_out'].fixed=False
    def forward(self,features,valid,mode=None,**kwargs):
        return super().forward(features,valid,self.mode if mode is None else mode,**kwargs)

class QATModel(SFTModel):
    def __init__(self,cfg):
        super().__init__(cfg);self.delta=CalibratedDelta(self.delta.layer)
