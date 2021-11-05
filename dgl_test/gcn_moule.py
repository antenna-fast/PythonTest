"""Torch Module for EdgeConv Layer"""

import torch
from torch import nn

from dgl.base import DGLError
from dgl import function as fn
from dgl.utils import expand_as_pair


class EdgeConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, g, feat, weight=None):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))  # e_feature = xj - xi, update edge feature
            g.edata['theta'] = self.theta(g.edata['theta'])  # theta m(xj - xi), encode local information of each edge
            g.dstdata['phi'] = self.phi(g.dstdata['x'])  # phi m(xi), encode global location

            if not self.batch_norm:
                g.update_all(fn.e_add_v('theta', 'phi', 'e'),  # message function, to dst node mailbox
                             fn.max('e', 'x'))  # aggregation (update)  to dstdata
            else:  # batch norm
                g.apply_edges(fn.e_add_v('theta', 'phi', 'e'))  # add local_edge_information and enc center point feature
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'e'),
                             fn.max('e', 'x'))
            return g.dstdata['x']


# Type 4 in 3D MOT
class WeightEdgeConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False):
        super(WeightEdgeConv, self).__init__()

        self.a_linear_1 = nn.Linear(in_feat, out_feat)
        self.a_linear_2 = nn.Linear(out_feat, 1)
        self.linear3 = nn.Linear(in_feat, out_feat)
        self.linear4 = nn.Linear(in_feat, out_feat)

        self.sigmoid = nn.Sigmoid()

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    # Aij
    def edge_regre(self, edges):  # edge feature update: edge reg as dynamic weight
        res = self.sigmoid(self.a_linear_2(nn.ReLU(self.a_linear_1(edges.data['theta']))))
        return {'w': res}

    # edge weight
    def edge_weight(self, edges):  # edge feature update: fixed weight
        return {'w_theta': edges.data['w'] * edges.data['theta']}  # new edge feature

    def reduce_func(self, nodes):  # update
        h = torch.sum(nodes.mailbox['h'], dim=1)
        return {'h': h}  # dst node mailbox

    def forward(self, g, feat, edge_weight=None):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst

            # edge diff
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))  # e_feature = xj - xi, on edge(diff iterm)

            # edge score (aij)
            g.apply_edges(self.edge_regre)

            # edge weight
            g.apply_edges(self.edge_weight)

            # center node linear
            g.dstdata['x'] = self.linear4(g.dstdata['x'])

            # update nodes feature
            # g.update_all(self.msg_func,
            g.update_all(fn.e_add_v('w_theta', 'x', 'h'),
                         self.reduce_func)

            rst = g.dstdata['h']

            # activation
            if self.activation is not None:
                rst = self.activation(rst)

            return rst
