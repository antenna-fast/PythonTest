"""
Author: ANTenna on 2022/1/14 4:05 下午
aliuyaohua@gmail.com

Description:

"""

import torch
import torch.nn as nn


if __name__ == '__main__':
    print('unit test code')
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output.backward()
    # 2D loss example (used, for example, with image inputs)
    N, C = 5, 4
    loss = nn.NLLLoss()
    # input is of size N x C x height x width
    data = torch.randn(N, 16, 10, 10)
    conv = nn.Conv2d(16, C, (3, 3))
    m = nn.LogSoftmax(dim=1)
    # each element in target has to have 0 <= value < C
    target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    output = loss(m(conv(data)), target)
    output.backward()

    # sample test
    a = torch.zeros(1, 5)
    b = torch.ones(1, dtype=torch.long)
    a = nn.LogSoftmax(dim=-1)(a)
    c = loss(a, b)
    print()
