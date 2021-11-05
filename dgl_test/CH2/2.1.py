import torch

import dgl
import dgl.function as fn


# 消息传递：

# 消息函数 message function
#   输入：edges
#   输出：将数据放在dst node 的 mailbox中
#
#   消息函数通过边获得变量
#   用于访问源节点、目标节点和边的特征，以及一些操作，如利用节点的特征(求和或者加权后)生成新的消息

# 聚合函数  reduce_func
#   输入：nodes
#   输出：dst 节点的输出
#   nodes 的成员属性 mailbox 可以用来访问节点收到的消息。
#   一些最常见的聚合操作包括 sum、max、min 等
#   聚合函数通常有两个参数，它们的类型都是字符串。m 用于指定 mailbox 中的字段名， return h 用于指示目标节点特征的字段名
#   具体的传递过程： nodes.mailbox['a'] -> dstdata['x']

# 更新函数
#   nodes
#   对 聚合函数 的聚合结果进行操作， 通常在消息传递的最后一步将其与节点的特征相结合，并将输出作为节点的新特征

# update_all()
# Send messages along all the edges of the specified type and
# update all the nodes of the corresponding destination type.
# 1. msg_func
# 2. reduce_func
# 3. update_func


# 内置函数
#   包含常见的消息函数和聚合函数
#   https://docs.dgl.ai/en/0.6.x/api/python/dgl.function.html

# 用户自定义函数
#   用于实现自定义的消息或聚合函数

# apply_edge:
# Update the features of the specified edges by the provided function.
# 更新边特征！！

# edge_func():
# 输入：edges
# 输出：新的边特征
# 配合apply_edge()使用

# reduce_func():
# 输入：nodes 使用里面的mailbox数据
# 返回：dst节点的更新的特征


# 测试用例
g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))

# 节点特征
node_feat_dim = 5
node_feat_dim2 = 5

g.ndata['h'] = torch.ones(g.num_nodes(), node_feat_dim, node_feat_dim)
g.ndata['m'] = torch.ones(g.num_nodes(), node_feat_dim2)*5  # 不同名字的特征可以具有不同的维度
# g.ndata['m'][1] = torch.ones(node_feat_dim2) * 8
# print('g.ndata h is \n', g.ndata['h'])

# 里面的术语：u是源节点

# 边特征
# Set and get feature ‘h’ for a graph of a single edge type.
edge_feat_dim = 2
g.edata['h'] = torch.ones((g.num_edges(), edge_feat_dim))   # num edges; edge features, this case, dim=5
print('g.edata h is \n', g.edata['h'][1])

# OK
# 根据源节点特征更新节点
# update_all() computing edge-wise features from node-wise features
g.update_all(fn.copy_u('m', 'd'),  # msg function. collect  computes message using source node feature. 从源节点获得特征
             fn.sum('d', 'h_sum'))  # aggregate function  # 将d求和 赋值给 节点特征h_sum
print('g.ndata h_sum is \n', g.ndata['h_sum'])
print('g fn is ', g)

# u_mul_e:
# computes a message on an edge by performing element-wise mul between features of src and edge if the
# features have the same shape; otherwise, it first broadcasts the features to a new shape and performs the
# element-wise operation.

# 如果有边的权重的 节点特征 更新方法
# Send messages through all edges and update all nodes
# OK, valid broadcasting between feature shapes (5, 10) and (10,)
g.update_all(fn.u_mul_e('h', 'h', 'mm'),   # mm是中间量，合理性：后面设计聚合函数的时候更加灵活
             fn.sum('mm', 'h_new')
             )
print('g is ', g)
print('g.ndata h_new is \n', g.ndata['h_new'])  # shape: (g.num_nodes(), 5, 10)

# case 3
g.ndata['h1'] = torch.ones((g.num_nodes(), 1, 10))
g.ndata['h2'] = torch.ones((g.num_nodes(), 5, 1))
# OK, valid broadcasting between feature shapes (1, 10) and (5, 1)
g.apply_edges(fn.u_add_v('h1', 'h2', 'x'))  # apply_edges also supports broadcasting
res = g.edata['x']  # shape: (g.num_edges(), 5, 10)
print('res is \n', res)

# fn test

# 加权图
weights = torch.ones(g.num_edges())  # 假设权重都是1（一维向量 对应链接每一条链接两点的边）
g.edata['w'] = weights
print('g is \n', g)


# 聚合函数
# 对接收到消息求和的用户定义函数
# dgl.function.sum('m', 'h')  等价
def reduce_func(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

# 消息函数, 接受一个edges


# 不涉及消息传递的情况下，通过 apply_edges() 单独调用逐边计算,参数是一个消息函数
# 并且在默认情况下，这个接口将更新所有的边
print('g.edata is \n', g.edata['w'])
g.apply_edges(fn.u_add_v('h', 'm', 'w'))  # 将两端nodes['h']的特征更新到edges['w']  会自动创建之前不存在的边feature
print('g.edata is \n', g.edata['w'])
print('gggg: ', g)
# OK, it works


# 将一条边两端的节点'hu' 和'hv' 的特征相加, 结果保存在边的'he'特征上  和fn.u_add_v等价
def message_func(edges):
    return {'he': edges.src['hu'] + edges.dst['hv']}


# 边特征
# Set and get feature ‘h’ for a graph of a single edge type.
# g.edata['h'] = torch.ones(2, 1)  # num edges; edge features
# message_func(g.edata)


# update_all()
# 在单个API调用里合并了 消息生成、 消息聚合和节点特征更新，这为从整体上进行系统优化提供了空间
# 参数是一个消息函数、一个聚合函数和一个更新函数


# 加权图 demo
# message_func,
# reduce_func,
# apply_node_func=None,
# etype=None
def updata_all_example(graph):
    # 在graph.ndata['ft']中存储结果
    # 将源节点特征 ft 与边特征 a 相乘生成消息 m  -- 消息函数
    # 然后对所有消息求和来更新节点特征 ft   -- 聚合函数
    # 再将 ft 乘以2得到最终结果 final_ft
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))
    # 在update_all外调用更新函数
    final_ft = graph.ndata['ft'] * 2
    return final_ft

