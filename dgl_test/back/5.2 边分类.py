# 有时用户希望预测图中边的属性值，这种情况下，用户需要构建一个边分类/回归的模型。

import numpy as np
import dgl
import torch

src = np.random.randint(0, 100, 500)
dst = np.random.randint(0, 100, 500)

# 同时建立反向边
edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
# 建立点和边特征，以及边的标签
edge_pred_graph.ndata['feature'] = torch.randn(100, 10)
edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
edge_pred_graph.edata['label'] = torch.randn(1000)
# 进行训练、验证和测试集划分
edge_pred_graph.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)

# 同样的方法也可以被用于计算任何节点的隐藏表示。 并从边的两个端点的表示，通过计算得出对边属性的预测。

# 对一条边计算预测值最常见的情况是将预测表示为一个函数，函数的输入为两个端点的表示， 输入还可以包括边自身的特征。


