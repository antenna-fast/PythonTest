import dgl
import torch
import numpy as np

u, v = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 3, 4])
g = dgl.graph((u, v))

g.ndata['a'] = torch.ones(g.number_of_nodes(), 2)
g.ndata['h'] = torch.zeros(g.number_of_nodes(), 10)

print('g is ', g)
print('g.ndata a is \n', g.ndata['a'])
print('g.items is ', g.ndata.items())

for feat_name, feat in g.ndata.items():
    print('feature {} is \n{} '.format(feat_name, feat))


def set_node(g):
    g.ndata['x'] = torch.ones(g.number_of_nodes(), 2)  # 问题：退出函数后，并没有新建x属性
    return g.ndata


f = set_node(g)
# g.ndata['x'] = torch.ones(g.number_of_nodes(), 2)  # 问题：退出函数后，并没有新建x属性
print('f is \n', f['x'])

print('g is ', g)

for i, j in g.ndata.items():
    print(i, j.shape)
