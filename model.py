import torch
import sys

sys.path.append("sgmse")
from sgmse.model import ScoreModel
from util import evaluate_model_pr

class PRScoreModel(ScoreModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model_pr(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss
    
    def _pA(self, xt, y, eps=1e-8):
        return y.abs()*xt/torch.clamp(xt.abs(), min=eps)