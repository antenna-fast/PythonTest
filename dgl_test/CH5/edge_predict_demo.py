
import dgl
import dgl.data
import dgl.function as fn
from dgl.nn.pytorch.conv.sageconv import SAGEConv

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
import scipy.sparse as sp

# import sklearn
from sklearn.metrics import roc_auc_score

# GOAL: Build a GNN-based link prediction model.
# Train and evaluate the model on a small DGL-provided dataset.

# This tutorial shows an example of predicting whether a citation relationship,
# either citing or being cited, between two papers exists in a citation network.

# OK, don't care too much about it for we only want to learn a link predict pipeline

# This tutorial formulates the link prediction problem as a binary classification problem as follows:

# Treat the edges in the graph as positive examples.
# Sample a number of non-existent edges (i.e. node pairs with no edges between them) as negative examples.
# Divide the positive examples and negative examples into a training set and a test set.
# Evaluate the model with any binary classification metric such as Area Under Curve (AUC).

# 存在的边作为正例！
# 采样不存在的边作为负例
# 将正负例分为训练和测试集合
# 评估2分类问题 使用AUC


# DATA
# Loading graph and features

dataset = dgl.data.CoraGraphDataset()
# print("dataset: ", dataset)  # CoraGraphDataset object
g = dataset[0]
print(g)

# Prepare training and testing sets

# Split edge set for training and testing
# This tutorial randomly picks 10% of the edges for positive examples in the test set,
# and leave the rest for the training set.
# It then samples the same number of edges for negative examples in both sets.

# 数据集分割：从positive中取出10%的+作为测试，90%的作为训练
u, v = g.edges()  # 存在的边，是正例

eids = np.arange(g.number_of_edges())  # 一个序列  0 - num-1
eids = np.random.permutation(eids)  # 打乱序号顺序
test_size = int(len(eids) * 0.1)  # 测试集大小
train_size = g.number_of_edges() - test_size  # train size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]  # test pos edges
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]  # train pos edges

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))  # 稀疏矩阵：所有的连接关系


# print("test_pos_u : ", test_pos_u.max())
# print("test_pos_v : ", test_pos_v.max())
# print("adj : \n", adj)

# 负样本
# Find all negative edges and split them for training and testing
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)  # 负样本 of edge

neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
# print("test_neg_u : ", test_neg_u)
# print("adj_neg : ", adj_neg)

# 在测试集上，把边去掉
# 由于这个操作会创建拷贝，对于大数据会减慢速度

# When training, you will need to remove the edges in the test set
# from the original graph. You can do this via
# dgl.remove_edges

train_g = dgl.remove_edges(g, eids[:test_size])


# Define a GraphSAGE model
# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model

# train_g.ndata['feat'].shape[1], 16

# RETURN: nodes feature after CONV MODULE
class GraphSAGE(nn.Module):  # Need to rewrite conv
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        # in_feats, out_feats, aggregator_type,
        # feat_drop=0., bias=True, norm=None, activation=None
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# The model then predicts the probability of existence of an edge
# by computing a score between the representations of both incident
# nodes with a function (e.g. an MLP or a dot product),
# which you will see in the next section.


# In previous tutorials you have learned how to compute node representations with a GNN.
# However, link prediction requires you to compute representation of pairs of nodes.

# Train pos and neg simultaneously (BUT calculate loss respectively)
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

#
test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())


# Edge predict function 1 : Dot product
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))  #
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


# Edge predict function 2 : MLP
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)  # For we catted the 2 nodes feature of 1 edge
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
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


# BUILD model
model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)

# You can replace DotPredictor with MLPPredictor.
pred = MLPPredictor(16)  # h_feats
# pred = DotPredictor()


def compute_loss(pos_score, neg_score):
    # input: pos score, neg score
    # output: bce loss
    scores = torch.cat([pos_score, neg_score])
    # pos score = 1; neg score = 0
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)  # <score, label>


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(200):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)  # pos score
    neg_score = pred(train_neg_g, h)  # neg score
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))

# Thumbnail Courtesy: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
