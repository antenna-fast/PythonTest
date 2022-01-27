import torch
import torch.nn as nn
import numpy as np


# Multi-class Loss Function

# NLL loss
# Negative Log Likelihood loss
# 处境尴尬，因为要加log_softmax  不如直接使用CE(方便), 好处是可以显示地看到数据流
# Input Dim: NxD, N is the num of samples, D is D classes
loss = nn.NLLLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()


softmax = nn.Softmax(dim=-1)

# def nll_loss(input, target):
#     res = -np.log2(softmax(input))
#     return 0


# Cross Entropy
# Cross Entropy = NLL(log(softmax(input)))


# print(input[0])
print(sum(input[0]))

# ERROR demo: input dim error: 输入的每个样本，都应当包含各个类别的概率分布
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([1, 0, 1])
# print(nn.CrossEntropyLoss()(a, b))


# 二分类
# BCELoss
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)  # 0 or 1
output = loss(m(input), target)
output.backward()
print('BCE input is \n', input)
print('BCE target is \n', target)


# # 手写BCE
# def BCEloss(input, target):
#     res = -sum(torch.mm(target, np.log2(input)))
#     return res


# BCEWithLogitsLoss = softmax()
target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
output = torch.full([10, 64], 1.5)  # A prediction (logit)
pos_weight = torch.ones([64])  # All weights are equal to 1
loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss(output, target)  # -log(sigmoid(1.5))


# 回归问题

# L1 Loss
# named MAE: mean absolute error
# 用于简单模型（？）
loss = nn.L1Loss()
input = torch.randn(3, 1, requires_grad=True)  # batch_size, feature_dim
target = torch.randn(3, 1)
output = loss(input, target)
output.backward()
print('L1 loss input \n', input)
print('l1 target \n', target)


# 手写实现
def L1loss(input, target):
    res = sum(abs(input - target)) / len(input)
    return res


def L2loss(input, target):
    res = sum((input - target)**2) / len(input)
    return res


a = L1loss(input, target)
print('a is ', a)
print('output is ', output)

# HingeEmbeddingLoss
hinge_loss = nn.HingeEmbeddingLoss()


# focal_loss = nn.
