import numpy as np
from scipy.sparse import coo_matrix

adj = coo_matrix((np.ones(5), ([3, 4, 0, 2, 1], [0, 2, 1, 4, 3])), shape=(5, 5), dtype=np.float32)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
print(adj)
