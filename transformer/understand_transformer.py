import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


# self-attention
# 这里演示batch=1的版本
x = torch.ones((5, 3))

# wij
w_ij = torch.mm(x, x.T)  # Nxd x dxN -> (N,N)  wij denote correlation of i_th sample and j_th one
softmax = nn.Softmax(dim=-1)
w_ij = softmax(w_ij)
print('w_ij is \n', w_ij)

# 以上就是self-attention操作，唯一在向量中传播信息的操作

# 计算输出
y = torch.mm(w_ij, x)
print('y is \n', y)


# dot product: 计算q样本和每个k样本的相关性【compatibility】
# permute invariant！！！相当于每个点都考虑全面的信息，没有遗漏，必然是不变的【用于点云处理】

# 1. q、k、v
# 添加三个nxk的矩阵, 用于计算self-attention的三个不同部分(q, k, v)
# wij = qiT kj
# wij = softmax(wij)
# yi = \sum(wij * vj)

# 2. Scaled-dot-product
# 减小向量长度不一致对学习的影响

# 3. Multi-head attention
# 结合几组有不同矩阵的self-attention
# 对于输入xi，每个attention-head 产生不同的输出向量yi

class SelfAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()  # 用于访问父类的init方法, 否则父类的init被当前init覆盖
        self.emb = emb  # embedding feature dimension
        self.heads = heads
        self.in_dim = emb
        self.out_dim = self.emb * self.heads  # compute all heads at once

        # compute all heads at once
        # 线性层，将x变换到多头q k v
        self.l_queries = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.l_keys = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.l_values = nn.Linear(self.in_dim, self.out_dim, bias=False)

        # linear projection from multi heads to output
        self.unifyhead = nn.Linear(self.emb * self.heads, emb)

    def forward(self, x):
        b, n, k = x.size()  # batch, num_sample, feature dim. OK (1, N, 3) for points example
        assert k == self.emb, 'Input dim mismatch!'

        h = self.heads
        # parse multi-head: [b, n, k] to [b, n, h, k]  OK
        keys = self.l_keys(x).view(b, n, h, k)  # batch, sample num, heads, feature dim
        queries = self.l_queries(x).view(b, n, h, k)
        values = self.l_values(x).view(b, n, h, k)

        # compute scaled dot-product
        # fold heads into the batch dimension
        # why transpose?
        keys = keys.transpose(1, 2).contiguous().view(b * h, n, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, n, k)
        values = values.transpose(1, 2).contiguous().view(b * h, n, k)

        # get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(k)  # Q*K^T / sqrt(k)
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, n, k)
        out = out.transpose(1, 2).contiguous().view(b, n, h * k)

        return self.unifyhead(out)


# 将self-attention包装成可以复用的block
# this is a transformer block
class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, ff_hidden_mult=4):
        super().__init__()

        self.attention = SelfAttention(emb=emb, heads=heads)  # self-attention实现单元之间的交互

        self.norm1 = nn.LayerNorm(emb)  # layer normalization
        self.norm2 = nn.LayerNorm(emb)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

    def forward(self, x):
        attended = self.attention(x)  # attention
        x = self.norm1(attended + x)  # add & layer norm
        feedforward = self.feed_forward(x)  # MLP
        x = self.norm2(feedforward + x)  # add & layer norm

        return x


if __name__ == '__main__':
    # classification using transformer
    tb = TransformerBlock(emb=3, heads=1)

    print()
