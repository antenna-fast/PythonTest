# import torch
# import torch.nn as nn
#
#
# class NN(nn.Module):
#     def __init__(self):
#         super(NN).__init__()
#         self.fc = nn.Liner(100, 2)
#
#     def forward(self, x):
#         return self.fc(x)
#
#
# # nn = NN()
# # print(nn)
# #
# # do = nn.Dropout(p=0.2)
# # print(do)
#
#
# N_HIDDEN = 10
#
# net_overfitting = torch.nn.Sequential(
#     torch.nn.Linear(1, N_HIDDEN),       # first hidden layer
#     torch.nn.ReLU(),                    # activation func for first hidden layer
#     torch.nn.Linear(N_HIDDEN, N_HIDDEN), # second hidden layer
#     torch.nn.ReLU(),                     # activation func for second hidden layer
#     torch.nn.Linear(N_HIDDEN, 1)
# )
#
#
# net_dropout = torch.nn.Sequential(
#     torch.nn.Linear(1, N_HIDDEN),
#     torch.nn.Dropout(0.5),           # drop 50% neurons
#     torch.nn.ReLU(),
#     torch.nn.Linear(N_HIDDEN, N_HIDDEN),
#     torch.nn.Dropout(0.5),
#     torch.nn.ReLU(),
#     torch.nn.Linear(N_HIDDEN, 1)
# )
#
# print(net_overfitting)
# print(net_dropout)
#
import torch
import torch.nn as nn


a = torch.randn((10, 2))
print(a)

d = nn.Dropout()
b = d(a)
print(b)
