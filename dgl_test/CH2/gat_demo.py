import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GAT, self).__init__()
        self.g = g

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim, 1, bias=False)

    # apply edge: update edge feature
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['x'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e', F.leaky_relu(a)}  # got edge.data['e'] 将edge score放在边e上

    # update_all: update dst nodes feature
    def message_func(self, edges):  # 消息函数, 将消息放到dst nodes的mailbox
        return {'z': edges.src['z'], 'e': edges.data['e']}
        # to dst nodes.mailbox['z']， 将src节点的特征放到dst的mailbox
        # 返回目标节点的mailbox标志

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)  # output
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)  # 计算边特征(score)
        self.g.update_all(self.message_func,  # msg func, 将边两端的节点信息传递到目标节点，发送节点的embedding
                          self.reduce_func)  # 接收mailbox里面的 embedding（feature），并且进行聚合
        return self.g.ndata.pop['h']

