# 高效的消息传递

import torch
import torch.nn as nn

import dgl


# 测试用例
# 基本概念于数据结构
g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))

# 节点特征
node_feat_dim = 3
node_feat_dim2 = 3

g.ndata['h'] = torch.ones(g.num_nodes(), node_feat_dim)
g.ndata['m'] = torch.ones(g.num_nodes(), node_feat_dim2)*5  # 不同名字的特征可以具有不同的维度

# 边特征
# Set and get feature ‘h’ for a graph of a single edge type.
edge_feat_dim = 5
g.edata['h'] = torch.ones((g.num_edges(), edge_feat_dim), dtype=torch.int32)  # num edges; edge features, this case, dim=5
# g.edata['h'] = torch.ones(2)  # num edges; edge features  shape=() 标量特征


node_feat_dim = 5

linear = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim * 2)))


def concat_message_function(edges):
     return {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']])}


g.apply_edges(concat_message_function)
g.edata['out'] = g.edata['cat_feat'] * linear


# 数学等价做法
# 避免不必要的从点到边的内存拷贝
# 但效率高得多，因为不需要在边上保存feat_src和feat_dst， 从内存角度来说是高效的
import dgl.function as fn

linear_src = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim)))
linear_dst = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim)))
out_src = g.ndata['feat'] * linear_src
out_dst = g.ndata['feat'] * linear_dst
g.dstdata.update({'out_dst': out_dst})
g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))
