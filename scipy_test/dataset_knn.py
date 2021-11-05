import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch


def fast_knns2spmat(knns, thd_sim=0.1):
    # convert knns to symmetric sparse matrix
    n = len(knns)
    nbrs = knns[:, 0, :]
    sims = knns[:, 1, :]
    assert -1e-5 <= sims.min() <= sims.max() <= 1 + 1e-5, "min: {}, max: {}".format(sims.min(), sims.max())
    row, col = np.where(sims >= thd_sim)
    idxs = np.where(row != nbrs[row, col])  # remove the self-loop
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_symmetric_adj(adj, self_loop=True):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


m = [[[1,2], [3,4]],
     [[5,6], [7,8]]]

sp = fast_knns2spmat(m)
print('sp is ', sp)

a = np.ones((3, 3))
print(a)

b = row_normalize(a)
print(b)

c = torch.tensor([[1, 2],
                  [3, 4]])
d = build_symmetric_adj(c)
print(d)
