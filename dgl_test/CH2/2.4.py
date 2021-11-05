# 一类常见的图神经网络建模的做法是在消息聚合前使用边的权重
# 比如在 图注意力网络(GAT) 和一些 GCN的变种

#  DGL的处理方法是：
#
# 将权重存为边的特征
# 在消息函数中用边的特征与源节点的特征相乘

import dgl.function as fn

# 假定eweight是一个形状为(E, dim)的张量，E是边的数量, dim 是边特征维度
graph.edata['a'] = eweight
graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                 fn.sum('m', 'ft'))
