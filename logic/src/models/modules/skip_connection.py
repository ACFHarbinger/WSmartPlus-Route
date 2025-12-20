import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, module:nn.Module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, *args, **kwargs):
        return input + self.module(input, *args, **kwargs)
