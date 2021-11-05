"""
Written by Yaohua Liu
yaohualiu@aibee.com
6.17 2021
"""

import dgl
import dgl.data
import dgl.function as fn
from dgl.nn.pytorch.conv.sageconv import SAGEConv
from ConvModule import GraphConv
# from loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
import scipy.sparse as sp


from sklearn.metrics import roc_auc_score

# GOAL: Build a GNN-based link prediction model.
# Train and evaluate the model on a small DGL-provided dataset.

# This tutorial shows an example of predicting whether a citation relationship,
# either citing or being cited, between two papers exists in a citation network.

# OK, don't care too much about it for we only want to learn the link predict pipeline

# This tutorial formulates the link prediction problem as a binary classification problem as follows:

# Treat the edges in the graph as positive examples.
# Sample a number of non-existent edges (i.e. node pairs with no edges between them) as negative examples.
# Divide the positive examples and negative examples into a training set and a test set.
# Evaluate the model with any binary classification metric such as Area Under Curve (AUC).


# Loading graph and features

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
print(g)

# Prepare training and testing sets

# Split edge set for training and testing
# Randomly picks 10% of the edges for positive examples in the test set,
# and leave the rest for the training set.
# It then samples the same number of edges for negative examples in both sets.

u, v = g.edges()

eids = np.arange(g.number_of_edges())  # [0 to num-1]
eids = np.random.permutation(eids)  # permutation
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]  # test set of pos edge
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]  # train set of pos edge

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))  # sparse matrix：Links of Nodes


# When training, you will need to remove the edges in the test set
# from the original graph. You can do this via
# dgl.remove_edges

train_g = dgl.remove_edges(g, eids[:test_size])


# NOTICE
# Define a GraphSAGE model
# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model

#
class DGLGCN(nn.Module):
    def __init__(self, reid_feature_dim, st_feature_dim, reid_nhid, st_nhid, nclass,
                 dropout=0., bn=False, gn=False, residual=False):
        super(DGLGCN, self).__init__()

        # BUILD CONV MODULE
        # in_feats, out_feats, aggregator_type,
        # feat_drop=0., bias=True, norm=None, activation=None
        self.reid_conv1 = GraphConv(reid_feature_dim, reid_nhid, 'mean')
        self.reid_conv2 = GraphConv(reid_nhid, reid_nhid, 'mean')

        self.st_conv1 = GraphConv(st_feature_dim, st_nhid, 'mean', dropout)
        self.st_conv2 = GraphConv(st_nhid, st_nhid, 'mean', dropout)

        out_size = (reid_nhid + st_nhid) / 2
        self.cat_conv1 = GraphConv(reid_nhid + st_nhid, out_size, 'mean', dropout)
        self.cat_conv2 = GraphConv(out_size, out_size, 'mean', dropout)

    # FORWARD
    # def forward(self, g, in_feat):
    def forward(self, g, data):

        reid_x, st_x, adj, idx = data[0], data[1], data[2], data[3]

        reid_x = self.conv1(g, reid_x)
        reid_x = F.relu(reid_x)
        reid_x = self.conv1(g, reid_x)

        st_x = self.conv1(g, st_x)
        st_x = F.relu(st_x)
        st_x = self.conv1(g, st_x)

        cat_feature = torch.cat([reid_x, st_x], dim=-1)  # cat app and　st feature
        cat_feature = self.cat_conv1(g, cat_feature)
        cat_feature = F.relu(cat_feature)
        cat_feature = self.cat_conv2(cat_feature)

        return cat_feature


# The model then predicts the probability of existence of an edge
# by computing a score between the representations of both incident
# nodes with a function (e.g. an MLP or a dot product),
# which you will see in the next section.

# Train pos and neg simultaneously (BUT calculate loss respectively)
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())


# nodes feature to edge score
# Calculate edge score using dot product of nodes feature
class DotPredictor(nn.Module):
    def forward(self, g, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


# Calculate edge score using using MLP
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        # h is nodes feature from CONV process (after cat)
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


nclass = 1
reid_nhid = 128
st_nhid = 128
model = DGLGCN(reid_feature_dim=64, st_feature_dim=120, reid_nhid=reid_nhid, st_nhid=st_nhid,
               nclass=nclass, dropout=0., bn=False, gn=False, residual=False)

out_size = (reid_nhid + st_nhid) / 2
pred = MLPPredictor(out_size)  # TODO


def loss_function(loss_type, nclass, Linthd=0.99):
    # input: edge score
    # output: loss function

    if nclass == 1:
        if loss_type == 'BCE':
            criterion = torch.nn.BCEWithLogitsLoss().cuda()
        elif loss_type == 'Focal':
            criterion = FocalLoss(logits=True, reduce=True)
        elif loss_type == 'Lin':
            criterion = LinLoss(thd=Linthd, logits=True, reduce=True)
        else:
            raise KeyError("Unknown loss_type:", loss_type)

    elif nclass == 2:
        criterion = nn.NLLLoss().cuda()
    else:
        raise ValueError('nclass should be 1 or 2')

    return criterion


# Calculate AUC
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

loss_fun = loss_function('BCE', 1)  # loss_type, nclass, Linthd=0.99

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata['feat'])  # 同时包含+-例训练样本
    pos_score = pred(train_pos_g, h)  # pos score
    loss = loss_fun(loss_fun)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    print('AUC', compute_auc(pos_score))

# Thumbnail Courtesy: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
