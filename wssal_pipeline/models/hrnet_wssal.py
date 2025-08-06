import torch
import torch.nn as nn
from lib.config import config
from lib.config.models import MODEL_EXTRAS
from lib.models.seg_hrnet import HighResolutionNet  # ✅ FIXED IMPORT

def get_seg_model(cfg):
    model = HighResolutionNet(cfg)  
    return model

class HRNetWSSAL(nn.Module):
    def __init__(self, cfg, use_ocr=True):
        super(HRNetWSSAL, self).__init__()
        self.model = get_seg_model(cfg)
        self.use_ocr = use_ocr

    def forward(self, x, mc_dropout=False):
        if mc_dropout:
            self.train()
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        else:
            self.eval()
        return self.model(x)
