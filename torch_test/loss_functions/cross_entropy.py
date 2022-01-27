"""
Author: ANTenna on 2022/1/14 11:32 上午
aliuyaohua@gmail.com

Description:

"""

import torch
import torch.nn as nn


if __name__ == '__main__':
    print('unit test code')
    # Domo from Pytorch
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

    batch_size = 1
    class_num = 5

    # This criterion combines `log_softmax` and `nll_loss` in a single function.
    criterion = nn.CrossEntropyLoss()
    input = torch.zeros(batch_size, class_num, requires_grad=True)  # [batch, samples, channel]
    target = torch.ones(batch_size, dtype=torch.long)  # [batch, class]
    # target不是以one-hot形式表示的，而是直接用scalar表示，里面的数值是索引!
    # This criterion expects a class index in the range [0, C-1]
    # as the target for each value of a 1D tensor of size minibatch;
    # if ignore_index is specified, this criterion also accepts this class index
    # (this index may not necessarily be in the class range).

    output = criterion(input, target)
    output.backward()

    # print('CE input is \n', input)  # 每个类别的分布
    # print('CE target is \n', target)
    # print('CE output is \n', output)

    # batch points demo:
    # 由于不是对每个sample进行分类，因此，产生了额外的维度
    p = torch.zeros(1, 10, 13)  # batch_size, per_batch_num, class
    p = p.permute(0, 2, 1)  # batch size, class, per_point
    t = torch.ones(1, 10, dtype=torch.long)  # batch_size, per_batch_num
    loss = criterion(p, t)
    print('loss: ', loss)
    print()

    # batch image segmentation demo:
    i = torch.zeros(1, 3, 4, 4)  # batch_size, class, w, h
    t = torch.ones(1, 4, 4, dtype=torch.long)  # batch_size, w, h 后面的wxh, 就存储每个像素的类别
    loss = criterion(i, t)
    print()
