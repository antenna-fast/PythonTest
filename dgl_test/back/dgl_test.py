import dgl
import torch

# 建图
g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))

# 节点特征

# 边特征
# Set and get feature ‘h’ for a graph of a single edge type.
g.edata['h'] = torch.ones(2, 1)  # num edges; edge features

print('g is \n', g)

# Set and get feature ‘h’ for a graph of multiple edge types.
# 异构图
g = dgl.heterograph({
    ('user', 'follows', 'user'): (torch.tensor([1, 2]), torch.tensor([3, 4])),
    ('user', 'plays', 'user'): (torch.tensor([2, 2]), torch.tensor([1, 1])),
    ('player', 'plays', 'game'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
})
g.edata['h'] = {('user', 'follows', 'user'): torch.zeros(2, 1),
                ('user', 'plays', 'user'): torch.ones(2, 1)}
b = g.edata['h']

g.edata['h'] = {('user', 'follows', 'user'): torch.ones(2, 1)}
c = g.edata['h']

print(g)
