import numpy as np

a = ['1', '2', '3']
b = [1, 2, 3]

dic = {}
dic.keys()

for i, j in enumerate(a):
    print(i, j)

# dic to numpy
d = {'1': 1, '2': 2, '3': 3}
print(d)

d_keys = d.keys()
d_keys = np.array(list(d_keys))
print(d_keys)

d_val = d.values()
d_val = np.array(list(d_val), dtype=float)
print(d_val)

print('d.items() is ', d.items())
