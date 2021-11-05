import scipy.sparse as sp
import numpy as np


a = np.eye(4)
# b = sp.

b = sp.csr_matrix(a)
print(b)

# 问题：转换之后不是对齐的

