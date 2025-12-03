import math
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, bias:bool=True):
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.linear(input)