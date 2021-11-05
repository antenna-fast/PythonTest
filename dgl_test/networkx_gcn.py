import torch
from networkx import to_numpy_matrix
import networkx as nx
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt

g = nx.karate_club_graph()
order = sorted(list(g.nodes()))
A = to_numpy_matrix(g, nodelist=order)
I = np.eye(g.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

W1 = np.random.normal(loc=0, scale=1, size=(g.number_of_nodes(), 4))
W2 = np.random.normal(loc=0, size=(W1.shape[1], 2))  # output_dim = 2

f = nn.ReLU()


def gcn_layer(A_hat, D_hat, X, W):
    res_A = np.dot(A_hat, X)  # 聚合
    res_D = np.dot(D_hat ** -1, res_A)  # 归一化
    res = np.dot(res_D, W)  # 权重
    return f(torch.tensor(res))  # 激活


H1 = gcn_layer(A_hat, D_hat, I, W1)  # I: 每个样本都是one hot向量
H2 = gcn_layer(A_hat, D_hat, H1, W2)
output = H2

fea = {node: np.array(output)[node] for node in g.nodes()}
print(fea)

res_fea = fea.values()
print('res_fea is \n', res_fea)
res_fea = np.array(list(res_fea), dtype=float)

plt.scatter(res_fea.T[0], res_fea.T[1])
plt.show()
