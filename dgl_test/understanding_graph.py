import numpy as np
import torch
import torch.nn as nn


# 节点特征
X = np.array([[1, 1],
              [2, 2],
              [3, 3]])

# 邻接矩阵
# 列方向：src
# 行方向：dst
A = np.array([[0, 1, 1],
              [0, 0, 1],
              [0, 0, 0]])

# 增加自环 self loop
A = A + np.eye(len(A))

# 度矩阵(用于归一化)
D = np.array(np.sum(A, axis=0))  # axis=0是对每一列求和(列代表入度)(指向dst)
# 因为计算特征聚合的时候使用了dst的：用哪些计算，就对哪些数量进行统计
D = np.matrix(np.diag(D))
print('D is \n', D)

# 邻接矩阵*特征矩阵: 特征聚合
# 相当于 对dst的(邻居)节点特征求和   这就是聚合的过程(没有包含自己的特征)
res = np.dot(A, X)
res = np.dot(D**-1, res)
print('res is \n', res)


# 添加权重 (这个是将来要学习的参数)
# in_feature__dim * out_feature__dim
# 需要适配输入特征维度，且决定输出特征维度
W = np.matrix([[2, 1, 2],
               [3, 4, 2]])
res_w = np.dot(res, W)
print('res_w is \n', res_w)

# 激活函数
# f = nn.ReLU()
f = nn.Softmax(dim=-1)
res_W_r = f(torch.Tensor(res_w))
print('res_W_r is \n', res_W_r)


