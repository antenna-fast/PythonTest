import torch
import torch.nn as nn


class T(nn.Module):
    def __init__(self):
        super(T, self).__init__()

        self.linear = torch.nn.Parameter(torch.FloatTensor(10))

    def edge_func(self, edges):
        a = torch.mm(self.linear, edges.src['a'])
        return {'e': a}

    def reduce_func(self):
        return {'': 0}

    def forward(self, g):

        return 0
