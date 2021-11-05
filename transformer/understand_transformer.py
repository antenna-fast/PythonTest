import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# dot product 代表了其他各个任务与当前任务的相关性
# 置换不变！！！相当于每个点都考虑全面的信息，没有遗漏，必然是不变的

# 最新的transformer实际使用的技巧

# 1. 查询q、键k、值v
# 添加三个dxd的矩阵, 用于计算self-attention的三个不同部分(q, k, v)
# wij = qiT kj
# wij = softmax(wij)
# yi = \sum(wij * vj)

# 2. 缩放点积
# 减小向量长度不一致对学习的影响

# 3. Multi-head attention
# 结合几组有不同矩阵的self-attention
# 对于输入xi，每个attention-head 产生不同的输出向量yi

class SelfAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()  # 用于访问父类的方法

        self.emb = 3
        self.heads = 1
        self.in_dim = emb
        self.out_dim = emb * heads

        self.l_queries = nn.Linear(self.in_dim, self.out_dim, bias=False)  # 线性层，将x变换到多头q k v
        self.l_keys = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.l_values = nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.unifyhead = nn.Linear(self.emb * self.heads, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        keys = self.l_keys(x).view(b, t, h, e)  # batch size,
        queries = self.l_queries(x).view(b, t, h, e)
        values = self.l_values(x).view(b, t, h, e)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e)
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyhead(out)


# 将self-attention包装成可以复用的block
# 这是一个transformer block
class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, ff_hidden_mult=4):
        super().__init__()

        self.attention = SelfAttention(emb=emb, heads=heads)  # 实现单元之间的交互

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

    def forward(self, x):
        attended = self.attention(x)  # attention
        x = self.norm1(attended + x)  # layer norm & residual
        fedforward = self.ff(x)  # MLP
        x = self.norm2(fedforward + x)  # layer norm & residual

        return x
