import torch

a = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])
b = a.T

print(a)
print(b)

c = torch.mm(a, b)
print('c is \n', c)

# 带有 batch的矩阵乘法
x = torch.ones((2, 3, 4))

# input and mat2 must be 3-D tensors each containing the same number of matrices.
out = torch.bmm(x, x.transpose(1, 2))
print('out is ', out)
