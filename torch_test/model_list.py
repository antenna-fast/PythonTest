import torch
import torch.nn as nn


l1 = nn.Linear(10, 20)
l2 = nn.Linear(20, 20)

# 1. 会将parameter注册到网络，使用时需要便携forward
model_list = nn.ModuleList()
model_list.append(l1)
model_list.append(l2)
print('model_list: ', model_list)


# 2. Sequential 无需编写forward
model_list = []
model_list.append(l1)
model_list.append(l2)
model = nn.Sequential(*model_list)
print('model: ', model)

# test
a = torch.randn((1, 10))

x = a
for l in model_list:  # forward
    x = l(x)
print(x)

output = model(a)
print(output)


