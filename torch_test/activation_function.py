import torch
import torch.nn as nn

softmax = nn.Softmax(dim=1)
# dim: 在哪一个维度上计算softmax
# 如dim=0在每一列计算，
# dim=1，就是在每一行计算
# dim=-1 就是在最后一个维度，如，对于图片就是dim=1的时候，对于大部分的分类输出，也是这样的：如 batch_size x dimension

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)
out = softmax(a)
print(out)

a = torch.tensor([1, 2, 3, 4])
b = a.max(dim=-1).indices  # find index of max element
print(b)

