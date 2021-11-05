import torch

a = torch.tensor([1, 2, 3, 4], requires_grad=True, dtype=torch.float32)
b = a.clone()  # b与a不是同个对象，不是同一块内存

print(b.requires_grad)

c = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(c)
# View tensor shares the same underlying data with its base tensor.
# Supporting View avoids explicit data copy, thus allows us to do fast
# and memory efficient reshaping, slicing and element-wise operations.
d = c.view(-1)
print(d)

print(id(c))
print(id(d))

# torch mm
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 0], [0, 1]])

out = torch.mm(a, b)
print(out)

# cat function
out = torch.cat([a, b], 0)  # cat in raw
# out = torch.cat([a, b], 1)  # cat in col
print('out is \n', out)
