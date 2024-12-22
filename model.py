import warnings
import os
import torch.optim
import torch.nn as nn

import config as c
from hinet import Hinet_stage


# 隐藏第一张图的网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet_stage()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        # split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            # if split[-2] == 'deconv10':
            #     param.data.fill_(0.)

