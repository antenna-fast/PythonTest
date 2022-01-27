import torch
import torch.nn as nn

in_dim = 3
out_dim = 5

l1 = nn.Linear(in_dim, out_dim)
l2 = nn.Linear(out_dim, out_dim)

# 1. 会将parameter注册到网络，使用时需要编写forward
model_list = nn.ModuleList()
model_list.append(l1)
model_list.append(l2)
print('model_list: \n', model_list)

# 2. Sequential 无需编写forward
model_list = []
model_list.append(l1)
model_list.append(l2)
model = nn.Sequential(*model_list)
print('sequential model: \n', model)

# test data
a = torch.randn((2, in_dim))
x = a

# model list, need forward
for l in model_list:
    x = l(x)
print('x1: \n', x)

output = model(a)
print(model)
print('x2: \n', output)

